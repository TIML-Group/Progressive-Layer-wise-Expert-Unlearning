import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import RandAugment
import random
import numpy as np
import time
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CV_DIR = os.path.dirname(BASE_DIR)
if CV_DIR not in sys.path:
    sys.path.insert(0, CV_DIR)

from try_moevit import MoEVisionTransformer  # 假设内部 MoEMLP 已返回 load_balance_loss


DATA_ROOT = "/root/autodl-tmp/SEUF_MCU/data_cv"
MODEL_DIR = os.path.join(BASE_DIR, "model")
CUTMIX_ALPHA = 1.0
FORGET_CLASS_NAME = "bicycle"

# ---------------- reproducibility ----------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_DIR, exist_ok=True)

# ==== 数据增强 ====
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# ==== 数据集：使用原始 fine labels（100类） ====
trainset = torchvision.datasets.CIFAR100(
    root=DATA_ROOT, train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR100(
    root=DATA_ROOT, train=False, download=True, transform=transform_test
)
eval_trainset = torchvision.datasets.CIFAR100(
    root=DATA_ROOT, train=True, download=True, transform=transform_test
)
eval_testset = torchvision.datasets.CIFAR100(
    root=DATA_ROOT, train=False, download=True, transform=transform_test
)

# ==== DataLoader ====
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
eval_trainloader = DataLoader(eval_trainset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

# ==== 模型：输出类别数改为 100 ====
num_classes = 100
model = MoEVisionTransformer(
    img_size=32,
    patch_size=4,
    embed_dim=256,
    depth=12,
    num_heads=8,
    num_classes=num_classes,
).to(device)

# print basic diagnostics
def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
print("Device:", device)
print("Total params: {:.2f}M".format(count_params(model)/1e6))


def rand_bbox(size, lam):
    height = size[2]
    width = size[3]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return x1, y1, x2, y2


def cutmix_data(inputs, targets, alpha=1.0):
    if alpha <= 0.0:
        return inputs, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    targets_a = targets
    targets_b = targets[index]

    x1, y1, x2, y2 = rand_bbox(inputs.size(), lam)
    mixed_inputs = inputs.clone()
    mixed_inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]

    box_area = (x2 - x1) * (y2 - y1)
    lam = 1.0 - box_area / float(inputs.size(2) * inputs.size(3))

    targets_a = targets
    return mixed_inputs, targets_a, targets_b, lam


def cutmix_criterion(criterion_fn, outputs, targets_a, targets_b, lam):
    return lam * criterion_fn(outputs, targets_a) + (1 - lam) * criterion_fn(outputs, targets_b)


def evaluate_loader(model_obj, loader, criterion_fn=None):
    model_obj.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs, aux_loss = model_obj(inputs)
            if criterion_fn is not None:
                loss_sum += criterion_fn(outputs, labels).item() * inputs.size(0)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100.0 * correct / total
    avg_loss = loss_sum / total if criterion_fn is not None else None
    return avg_loss, acc


def evaluate_subset_accuracy(model_obj, dataset, indices, batch_size, desc):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model_obj.eval()
    total = 0
    correct = 0
    index_set = set(indices)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            start = batch_idx * batch_size
            end = start + labels.size(0)
            batch_indices = [idx for idx in range(start, end) if idx in index_set]
            if not batch_indices:
                continue

            local_positions = torch.tensor([idx - start for idx in batch_indices], device=labels.device)
            inputs = inputs[local_positions].to(device, non_blocking=True)
            labels = labels[local_positions].to(device, non_blocking=True)

            outputs, _ = model_obj(inputs)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"{desc}: {acc:.2f}% ({correct}/{total})")
    return acc

# ==== criterion / optimizer / scheduler ====
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

base_lr = 1e-3
optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

epochs = 250
warmup_epochs = 15
alpha = 0.01  # balance loss weight for aux_loss (if model returns it)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

best_acc = 0.0
start_time = time.time()

for epoch in range(epochs):
    # ---- warmup lr ----
    if epoch < warmup_epochs:
        warmup_factor = (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * warmup_factor

    current_lr = optimizer.param_groups[0]['lr']

    # ---- 训练阶段 ----
    model.train()
    running_loss, total = 0.0, 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        mixed_inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, CUTMIX_ALPHA)
        outputs, aux_loss = model(mixed_inputs)
        loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam) + alpha * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)

    train_loss = running_loss / total
    clean_train_loss, clean_train_acc = evaluate_loader(model, eval_trainloader, criterion)

    # ---- 测试阶段 ----
    test_loss, test_acc = evaluate_loader(model, testloader, criterion)

    if epoch >= warmup_epochs:
        scheduler.step()

    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs} | lr={current_lr:.6f} | "
          f"Train loss={train_loss:.4f}, Clean Train loss={clean_train_loss:.4f}, Clean Train Acc={clean_train_acc:.2f}% | "
          f"Test loss={test_loss:.4f}, Test Acc={test_acc:.2f}% | elapsed={elapsed/60:.2f}min")

    # ==== 保存最佳模型 ====
    if test_acc > best_acc:
        best_acc = test_acc
        save_path = os.path.join(MODEL_DIR, "moevit_cifar100_best.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, save_path)
        print(f"==> 保存最佳模型到 {save_path}，Test Acc: {best_acc:.2f}%")

print(f"训练完成，最佳测试集保留准确率: {best_acc:.2f}%")

best_model_path = os.path.join(MODEL_DIR, "moevit_cifar100_best.pth")
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

fine_to_idx = eval_trainset.class_to_idx
if FORGET_CLASS_NAME not in fine_to_idx:
    raise ValueError(
        f"Forget class {FORGET_CLASS_NAME} not found in CIFAR100 classes: {sorted(fine_to_idx.keys())}"
    )
forget_class_idx = fine_to_idx[FORGET_CLASS_NAME]

train_forget_indices = [idx for idx, label in enumerate(eval_trainset.targets) if label == forget_class_idx]
train_retain_indices = [idx for idx, label in enumerate(eval_trainset.targets) if label != forget_class_idx]
test_forget_indices = [idx for idx, label in enumerate(eval_testset.targets) if label == forget_class_idx]
test_retain_indices = [idx for idx, label in enumerate(eval_testset.targets) if label != forget_class_idx]

print(f"\n=== Split Accuracy For Class '{FORGET_CLASS_NAME}' (index={forget_class_idx}) ===")
train_forget_acc = evaluate_subset_accuracy(
    model,
    eval_trainset,
    train_forget_indices,
    batch_size=128,
    desc="Train acc on forget class",
)
train_retain_acc = evaluate_subset_accuracy(
    model,
    eval_trainset,
    train_retain_indices,
    batch_size=128,
    desc="Train acc on non-forget classes",
)
test_forget_acc = evaluate_subset_accuracy(
    model,
    eval_testset,
    test_forget_indices,
    batch_size=100,
    desc="Test acc on forget class",
)
test_retain_acc = evaluate_subset_accuracy(
    model,
    eval_testset,
    test_retain_indices,
    batch_size=100,
    desc="Test acc on non-forget classes",
)

print("\n=== Final 4 Acc Results ===")
print(f"Train forget-class acc: {train_forget_acc:.2f}%")
print(f"Train non-forget acc: {train_retain_acc:.2f}%")
print(f"Test forget-class acc: {test_forget_acc:.2f}%")
print(f"Test non-forget acc: {test_retain_acc:.2f}%")
