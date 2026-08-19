import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torchvision.transforms import RandAugment
import time
import os
from try_moevit import MoEVisionTransformer  # 请替换为你的MoEVisionTransformer路径
import pickle
from PIL import Image

ga_epoches = 10
lr = 0.00005
ckpt_path = "moevit_cifar100_super_best.pth"
zl = 'fine'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/root/autodl-tmp/SEUF_MCU/data_cv/cifar-100-python"

# ===================== 数据增强 =====================
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    RandAugment(num_ops=2, magnitude=9),  # 强化增强
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
    transforms.RandomErasing(p=0.25)  # 随机擦除
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# ===================== MIA 计算 =====================
def compute_mia(net, forget_loader, retain_loader, device):
    net.eval()
    forget_confs, retain_confs = [], []
    with torch.no_grad():
        for inputs, _ in forget_loader:
            inputs = inputs.to(device)
            p1,_=net(inputs)
            probs = torch.softmax(p1, dim=1)
            confs, _ = torch.max(probs, dim=1)
            forget_confs.extend(confs.cpu().numpy())
        for inputs, _ in retain_loader:
            inputs = inputs.to(device)
            p2,_=net(inputs)
            probs = torch.softmax(p2, dim=1)
            confs, _ = torch.max(probs, dim=1)
            retain_confs.extend(confs.cpu().numpy())
    return abs(np.mean(forget_confs) - np.mean(retain_confs))

# ===================== 专家亲和度 =====================
def compute_expert_affinity_per_layer(net, layer_idx, dataloader, device):
    blk = net.blocks[layer_idx]
    blk.eval()
    num_experts = len(blk.mlp.experts)
    expert_softmax_sum = torch.zeros(num_experts, device=device)
    expert_top1_count = torch.zeros(num_experts, device=device)
    total_tokens = 0

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            x = net.patch_embed(inputs)
            for i, b in enumerate(net.blocks[:layer_idx]):
                x = b(x)
                if isinstance(x, tuple):
                    x = x[0]  # 只取 tensor

            gate_logits = blk.mlp.gate(x)
            gate_softmax = torch.softmax(gate_logits, dim=-1)
            expert_softmax_sum += gate_softmax.sum(dim=(0, 1))

            top1_idx = torch.argmax(gate_logits, dim=-1)
            for expert_id in range(num_experts):
                expert_top1_count[expert_id] += (top1_idx == expert_id).sum().float()

            total_tokens += x.shape[0] * x.shape[1]

    softmax_avg = (expert_softmax_sum / total_tokens).cpu().numpy()
    top1_ratio = (expert_top1_count / total_tokens).cpu().numpy()
    return softmax_avg, top1_ratio

def compute_anchor_loss(g, target_expert_idx, num_experts):
    a = torch.zeros_like(g, device=g.device)
    a[:, target_expert_idx] = 1.0
    anchor_loss = torch.norm(g - a, p=2, dim=1).mean()
    return anchor_loss


def load_cifar100_coarse(train_transform, test_transform):
    meta_path = os.path.join(DATA_DIR, "meta")
    train_path = os.path.join(DATA_DIR, "train")
    test_path = os.path.join(DATA_DIR, "test")
    required_files = [meta_path, train_path, test_path]
    missing_files = [path for path in required_files if not os.path.exists(path)]
    if missing_files:
        missing_str = "\n".join(missing_files)
        raise FileNotFoundError(
            "Missing CIFAR100 local files:\n"
            f"{missing_str}"
        )

    with open(train_path, 'rb') as f:
        train_dict = pickle.load(f, encoding='latin1')
    with open(test_path, 'rb') as f:
        test_dict = pickle.load(f, encoding='latin1')

    fine2coarse = {}
    for fine_label, coarse_label in zip(train_dict['fine_labels'], train_dict['coarse_labels']):
        fine2coarse[fine_label] = coarse_label
    for fine_label, coarse_label in zip(test_dict['fine_labels'], test_dict['coarse_labels']):
        fine2coarse.setdefault(fine_label, coarse_label)

    class CIFAR100CoarseLocal(Dataset):
        def __init__(self, split_dict, transform, fine2coarse_map):
            self.images = split_dict['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.fine_labels = split_dict['fine_labels']
            self.fine2coarse = fine2coarse_map
            self.transform = transform

        def __len__(self):
            return len(self.fine_labels)

        def __getitem__(self, index):
            image = Image.fromarray(self.images[index])
            label = self.fine2coarse[self.fine_labels[index]]
            if self.transform is not None:
                image = self.transform(image)
            return image, label

    full_trainset = CIFAR100CoarseLocal(train_dict, train_transform, fine2coarse)
    testset = CIFAR100CoarseLocal(test_dict, test_transform, fine2coarse)
    return full_trainset, testset

# ===================== 主函数 =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==== 数据集划分 ====
    split_dir = os.path.join(BASE_DIR, "split_results_cifar100_" + zl)
    forget_file_train = 'forget_indices_' + zl + '_train.npy'
    retain_file_train = 'retain_indices_' + zl + '_train.npy'
    forget_file_test = 'forget_indices_' + zl + '_test.npy'
    retain_file_test = 'retain_indices_' + zl + '_test.npy'
    ckpt_full_path = os.path.join(BASE_DIR, ckpt_path)

    required_split_files = [
        os.path.join(split_dir, forget_file_train),
        os.path.join(split_dir, retain_file_train),
        os.path.join(split_dir, forget_file_test),
        os.path.join(split_dir, retain_file_test),
    ]
    missing_split_files = [path for path in required_split_files if not os.path.exists(path)]
    if missing_split_files:
        missing_str = "\n".join(missing_split_files)
        raise FileNotFoundError(
            "Missing forget/retain split files:\n"
            f"{missing_str}\n"
            "Please generate them first, e.g. run "
            f"`python {os.path.join(BASE_DIR, 'subclass_100.py')}`."
        )

    # ==== 加载 CIFAR100 coarse labels ====
    full_trainset, testset = load_cifar100_coarse(transform_train, transform_test)

    forget_train = Subset(full_trainset, np.load(os.path.join(split_dir, forget_file_train)))
    retain_train = Subset(full_trainset, np.load(os.path.join(split_dir, retain_file_train)))
    forget_test = Subset(testset, np.load(os.path.join(split_dir, forget_file_test)))
    retain_test = Subset(testset, np.load(os.path.join(split_dir, retain_file_test)))

    forget_train_loader = DataLoader(forget_train, batch_size=128, shuffle=True, num_workers=2)
    retain_train_loader = DataLoader(retain_train, batch_size=128, shuffle=True, num_workers=2)
    forget_test_loader = DataLoader(forget_test, batch_size=128, shuffle=False, num_workers=2)
    retain_test_loader = DataLoader(retain_test, batch_size=128, shuffle=False, num_workers=2)

    # ==== 模型加载 ====
    model = MoEVisionTransformer(
        img_size=32, patch_size=4, embed_dim=128, depth=8, num_heads=4, num_classes=20
    ).to(device)

    if not os.path.exists(ckpt_full_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_full_path} not found")

    # ======= 修改这里，兼容 checkpoint =======
    checkpoint = torch.load(ckpt_full_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 如果类别不匹配，允许 strict=False 加载，其它层权重可以加载
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("Warning: state_dict keys mismatch, trying strict=False...")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    print(f"Loaded initial model from {ckpt_full_path}")

    # ==== 评估函数 ====
    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out,_ = model(x)
                _, pred = out.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        return 100. * correct / total

    # ==== 遗忘训练前评估 ====
    print("=== Pre-GA Evaluation ===")
    train_ra = evaluate(retain_train_loader)
    train_fa = evaluate(forget_train_loader)
    test_ra = evaluate(retain_test_loader)
    test_fa = evaluate(forget_test_loader)
    train_ua = 100. - train_fa
    test_ua = 100. - test_fa
    mia = compute_mia(model, forget_train_loader, retain_train_loader, device)
    print(f"Before GA Training | "
          f"Train RA: {train_ra:.2f}% | Train UA: {train_ua:.2f}% | "
          f"Test RA: {test_ra:.2f}% | Test UA: {test_ua:.2f}% | MIA: {mia:.4f}")

    # ===================== 专家归因 =====================
    target_experts_per_layer = {}
    print("=== Expert Affinity per MoE Layer ===")
    for i, blk in enumerate(model.blocks):
        if hasattr(blk.mlp, 'experts'):
            softmax_avg, top1_ratio = compute_expert_affinity_per_layer(model, i, forget_train_loader, device)
            target_idx = int(np.argmax(top1_ratio))
            target_experts_per_layer[i] = target_idx
            softmax_str = ", ".join([f"{s:.4f}" for s in softmax_avg])
            top1_str = ", ".join([f"{t:.4f}" for t in top1_ratio])
            print(f"Layer {i} | Softmax Avg: {softmax_str} | Top-1 Ratio: {top1_str} | Target Expert -> {target_idx}")

    # ===================== 冻结非目标专家 =====================
    for i, blk in enumerate(model.blocks):
        if hasattr(blk.mlp, 'experts'):
            tgt = target_experts_per_layer[i]
            for j, expert in enumerate(blk.mlp.experts):
                requires_grad = (j == tgt)
                for p in expert.parameters():
                    p.requires_grad = requires_grad

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ===================== 训练函数 =====================
    def train_one_epoch():
        model.train()
        total_loss = 0
        for (x_r, y_r), (x_f, y_f) in zip(retain_train_loader, forget_train_loader):
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f, y_f = x_f.to(device), y_f.to(device)

            optimizer.zero_grad()
            out_r,_ = model(x_r)
            out_f,_ = model(x_f)

            loss_r = criterion(out_r, y_r)
            loss_f = -criterion(out_f, y_f)

            anchor_losses = []
            for i, blk in enumerate(model.blocks):
                if hasattr(blk.mlp, 'experts'):
                    tgt_idx = target_experts_per_layer[i]
                    if isinstance(x_f, tuple):
                        x_f = x_f[0]  # 只取 tensor
                    gate_logits = blk.mlp.gate(x_f)
                    gate_softmax = torch.softmax(gate_logits, dim=-1)
                    anchor_losses.append(compute_anchor_loss(gate_softmax, tgt_idx, len(blk.mlp.experts)))

            loss_anchor = torch.stack(anchor_losses).mean() if len(anchor_losses) > 0 else torch.tensor(0.0, device=device)
            loss = loss_r + loss_f + 0.1 * loss_anchor

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(retain_train_loader), len(forget_train_loader))

    # ===================== 主训练循环 =====================
    start_time = time.time()
    for epoch in range(ga_epoches):
        loss = train_one_epoch()
        train_ra = evaluate(retain_train_loader)
        train_fa = evaluate(forget_train_loader)
        test_ra = evaluate(retain_test_loader)
        test_fa = evaluate(forget_test_loader)
        train_ua = 100. - train_fa
        test_ua = 100. - test_fa
        mia = compute_mia(model, forget_train_loader, retain_train_loader, device)
        print(f"Epoch {epoch + 1}/{ga_epoches} | Loss: {loss:.4f} | "
              f"Train RA: {train_ra:.2f}% | Train UA: {train_ua:.2f}% | "
              f"Test RA: {test_ra:.2f}% | Test UA: {test_ua:.2f}% | MIA: {mia:.4f}")

    rte = (time.time() - start_time) / 60.0
    print(f"=== Final Metrics ===")
    print(f"Train Retain Acc (RA): {train_ra:.2f}%")
    print(f"Train Forget Acc (UA): {train_ua:.2f}%")
    print(f"Test Retain Acc (RA): {test_ra:.2f}%")
    print(f"Test Forget Acc (UA): {test_ua:.2f}%")
    print(f"MIA: {mia:.4f}")
    print(f"RTE: {rte:.2f} minutes")


if __name__ == "__main__":
    main()
