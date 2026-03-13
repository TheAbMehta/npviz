
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

try:
    import torch_pruning as tp
except ImportError:
    os.system("pip install torch-pruning")
    import torch_pruning as tp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from npviz.schema import ArchSnapshot, ImportanceSnapshot, MetricSnapshot, RewireEvent
from npviz.recorder import Recorder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 128
LR = 0.1
PRUNE_RATIO = 0.15
PRUNE_EVERY_N_EPOCHS = 5
LOG_DIR = Path("runs/resnet56_self_pruning")

print(f"Device: {DEVICE}")
print(f"Pruning {PRUNE_RATIO*100:.0f}% of channels every {PRUNE_EVERY_N_EPOCHS} epochs")
print(f"Training for {EPOCHS} epochs")
print()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet56(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 16, 9, stride=1)
        self.layer2 = self._make_layer(16, 32, 9, stride=2)
        self.layer3 = self._make_layer(32, 64, 9, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = ResNet56().to(DEVICE)
total_params_init = sum(p.numel() for p in model.parameters())
print(f"Initial model: {total_params_init:,} parameters")

recorder = Recorder(LOG_DIR)


def get_layer_info(model):
    layer_groups = []
    for name in ["layer1", "layer2", "layer3"]:
        layer_seq = getattr(model, name)
        for i, block in enumerate(layer_seq):
            params = sum(p.numel() for p in block.parameters())
            n_channels = block.conv1.weight.shape[0]
            layer_groups.append((f"{name}.{i}", params, n_channels))
    return layer_groups


def log_snapshot(step):
    info = get_layer_info(model)
    n_layers = len(info)
    heads_per_layer = [ch for _, _, ch in info]
    params_per_layer = [p for _, p, _ in info]
    total = sum(p.numel() for p in model.parameters())
    snap = ArchSnapshot(
        step=step,
        timestamp=time.time(),
        n_layers=n_layers,
        heads_per_layer=heads_per_layer,
        params_per_layer=params_per_layer,
        total_params=total,
        connections=[(i, i + 1) for i in range(n_layers - 1)],
    )
    recorder.log_snapshot_obj(snap)
    return snap


def log_importance(step):
    scores = []
    for name in ["layer1", "layer2", "layer3"]:
        layer_seq = getattr(model, name)
        for block in layer_seq:
            w = block.conv1.weight.data
            channel_scores = w.abs().mean(dim=(1, 2, 3)).cpu().tolist()
            scores.append([round(s, 6) for s in channel_scores])
    recorder.log_importance(ImportanceSnapshot(step=step, scores=scores))


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

log_snapshot(0)
log_importance(0)

global_step = 0
prune_count = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        global_step += 1

    train_loss = running_loss / len(trainloader)
    train_acc = 100.0 * correct / total

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    test_loss /= len(testloader)
    test_acc = 100.0 * test_correct / test_total

    current_params = sum(p.numel() for p in model.parameters())
    print(f"Epoch {epoch:>2}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:.1f}% | "
          f"Params: {current_params:,} ({current_params/total_params_init*100:.1f}%)")

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    recorder.log_metrics(global_step, train_loss, round(grad_norm, 4), scheduler.get_last_lr()[0])
    log_importance(global_step)
    log_snapshot(global_step)

    if epoch % PRUNE_EVERY_N_EPOCHS == 0 and epoch < EPOCHS - 2:
        prune_count += 1
        print(f"  >> PRUNING ROUND {prune_count} (removing {PRUNE_RATIO*100:.0f}% channels)...")

        loss_before = test_loss
        params_before = current_params

        importance = tp.importance.MagnitudeImportance(p=1)
        example_inputs = torch.randn(1, 3, 32, 32).to(DEVICE)

        pruner = tp.pruner.MetaPruner(
            model,
            example_inputs,
            importance=importance,
            pruning_ratio=PRUNE_RATIO,
            global_pruning=True,
            ignored_layers=[model.fc],
        )

        pruned_layers = []
        for group in pruner.step(interactive=True):
            affected = []
            for dep, idxs in group:
                module = dep.target.module
                for name, mod in model.named_modules():
                    if mod is module:
                        affected.append((name, len(idxs)))
                        break
            group.prune()
            pruned_layers.extend(affected)

        optimizer = optim.SGD(model.parameters(), lr=scheduler.get_last_lr()[0],
                              momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - epoch)

        model.eval()
        post_prune_loss = 0.0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                post_prune_loss += criterion(outputs, targets).item()
        post_prune_loss /= len(testloader)

        params_after = sum(p.numel() for p in model.parameters())

        layer_info = get_layer_info(model)
        for name, n_pruned in pruned_layers[:5]:
            layer_idx = 0
            for li, (lname, _, _) in enumerate(layer_info):
                if name.startswith(lname.replace(".", "")) or lname in name:
                    layer_idx = li
                    break

            recorder.log_rewire(RewireEvent(
                step=global_step,
                event_type="prune_head",
                layer_idx=layer_idx,
                reason=f"L1 structured pruning: {n_pruned} channels removed from {name}",
                importance_score=None,
                loss_before=round(loss_before, 4),
                loss_after=round(post_prune_loss, 4),
            ))

        print(f"  >> Pruned: {params_before:,} -> {params_after:,} params "
              f"({(1-params_after/params_before)*100:.1f}% reduction this round)")
        print(f"  >> Loss impact: {loss_before:.4f} -> {post_prune_loss:.4f}")

        log_snapshot(global_step)
        log_importance(global_step)

    scheduler.step()

recorder.close()
final_params = sum(p.numel() for p in model.parameters())
print()
print("=" * 60)
print(f"DONE. Log saved to: {LOG_DIR}")
print(f"Initial params:  {total_params_init:,}")
print(f"Final params:    {final_params:,}")
print(f"Pruned:          {(1 - final_params/total_params_init)*100:.1f}%")
print(f"Final test acc:  {test_acc:.1f}%")
print(f"Pruning rounds:  {prune_count}")
