import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

# -------------------- 可配置项 --------------------
BATCH_SIZE = 128
NUM_EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # DataLoader 的 workers
USE_AMP = True  # 是否使用混合精度
STEP_LR_STEP = 30
STEP_LR_GAMMA = 0.1
BEST_MODEL_PATH = "./best_model.pth"


# -------------------------------------------------

# -------------------- 模型定义----------------------
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,
                               stride=strides, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1,
                                   stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_1x1conv = use_1x1conv

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.use_1x1conv:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return nn.Sequential(*blk)


class Resnet(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            resnet_block(64, 64, num_residuals=4, first_block=True),
            resnet_block(64, 128, num_residuals=4),
            resnet_block(128, 256, num_residuals=4),
            resnet_block(256, 512, num_residuals=4),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, X):
        return self.model(X)


# -----------------------------------------------------

# -------------------- 数据增强--------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

train_transform = torchvision.transforms.Compose([
    # 随机填充后裁剪，保持尺寸 32x32
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),  # 小幅旋转
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


# ---------------------------------------------------------

def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, scheduler, num_epochs, device, use_amp=True):
    train_loss_history, train_acc_history = [], []
    test_loss_history, test_acc_history = [], []

    best_test_acc = 0.0
    best_model_state = None

    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for X, y in progress_bar:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(X)
                    loss = loss_fn(outputs, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(X)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()

            epoch_train_loss += loss.item() * y.size(0)
            epoch_train_correct += (outputs.argmax(dim=1) == y).sum().item()

            # 显示当前 batch 的 loss 和显存占用
            gpu_mem = torch.cuda.memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0
            progress_bar.set_postfix({
                "BatchLoss": f"{loss.item():.4f}",
                "GPU_MB": f"{gpu_mem:.1f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        # 每 epoch 结束，学习率 scheduler.step()
        if scheduler is not None:
            scheduler.step()

        # 计算并记录 epoch 平均 loss / acc
        train_loss_epoch = epoch_train_loss / len(train_dataloader)
        train_acc_epoch = epoch_train_correct / len(train_dataloader.dataset)
        train_loss_history.append(train_loss_epoch)
        train_acc_history.append(train_acc_epoch)

        # 验证
        model.eval()
        epoch_test_loss = 0.0
        epoch_test_correct = 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                outputs = model(X)
                loss = loss_fn(outputs, y)
                epoch_test_loss += loss.item() * y.size(0)
                epoch_test_correct += (outputs.argmax(dim=1) == y).sum().item()

        test_loss_epoch = epoch_test_loss / len(test_dataloader)
        test_acc_epoch = epoch_test_correct / len(test_dataloader.dataset)
        test_loss_history.append(test_loss_epoch)
        test_acc_history.append(test_acc_epoch)

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}  "
            f"TrainLoss: {train_loss_epoch:.4f}  TrainAcc: {train_acc_epoch:.4f}  "
            f"TestLoss: {test_loss_epoch:.4f}  TestAcc: {test_acc_epoch:.4f}"
        )

        # 存下最优模型参数
        if test_acc_epoch > best_test_acc and test_acc_epoch > 0.8:
            best_test_acc = test_acc_epoch
            best_model_state = model.state_dict()

    return best_model_state, train_loss_history, train_acc_history, test_loss_history, test_acc_history


def plot_results(train_loss, train_acc, test_loss, test_acc):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, test_acc, label='Test Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.title('Train vs Test Acc')
    plt.legend()

    plt.show()


# -------------------- 主程序入口 --------------------
def main():
    print("Device:", DEVICE, "USE_AMP:", USE_AMP)
    transform_train = train_transform
    transform_test = test_transform

    train_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=transform_train, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=transform_test, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)

    model = Resnet(num_classes=10).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_LR_STEP, gamma=STEP_LR_GAMMA)

    best_state, train_loss, train_acc, test_loss, test_acc = train(
        train_loader, test_loader, model, loss_fn, optimizer, scheduler,
        NUM_EPOCHS, DEVICE, use_amp=USE_AMP
    )

    # 保存最优模型
    if best_state is not None:
        torch.save(best_state, BEST_MODEL_PATH)

    plot_results(train_loss, train_acc, test_loss, test_acc)


if __name__ == "__main__":
    main()
