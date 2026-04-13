"""Profile data loading vs compute for a single client training epoch.

Usage (cluster):
    python profile_training.py --device cuda --resize 224 --hidden_size 64 --rawsmpl 1.0 --bf16
Usage (local):
    python profile_training.py --device mps --resize 112 --hidden_size 12 --rawsmpl 0.05
"""
import os
import time
import argparse
import torch
import torchvision
from tqdm import tqdm
from PIL import Image

from src.datasets.chexpert import CheXpertDataset, CheXpertWrapper
from src.models.resnet import ResNet18

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--resize', type=int, default=224)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--rawsmpl', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--bf16', action='store_true')
parser.add_argument('--data_path', type=str, default='./external-data')
args = parser.parse_args()

DEVICE = args.device
NUM_CLASSES = 14
root = os.path.join(args.data_path, 'CheXpert')

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((args.resize, args.resize)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5]*3, [0.5]*3),
])

print(f'Loading dataset (rawsmpl={args.rawsmpl})...')
ds = CheXpertDataset(
    root=root,
    csv_file=os.path.join(root, 'train.csv'),
    transform=transform,
    raw_fraction=args.rawsmpl,
)
ds = CheXpertWrapper(ds, 'CHEXPERT', 'CLIENT')
loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                      pin_memory='cuda' in DEVICE,
                                      num_workers=0)
print(f'Dataset: {len(ds)} samples, {len(loader)} batches')
print(f'Device: {DEVICE}, bf16: {args.bf16}, hidden_size: {args.hidden_size}')

model = ResNet18(in_channels=3, hidden_size=args.hidden_size, num_classes=NUM_CLASSES).to(DEVICE)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Warmup
print('Warming up...')
for i, (inputs, targets) in enumerate(loader):
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if i >= 4:
        break

# 1) Data loading only
print('\n--- Data loading only ---')
start = time.time()
for inputs, targets in tqdm(loader, desc='Data loading'):
    pass
data_time = time.time() - start
print(f'Data loading: {data_time:.2f}s ({data_time/len(loader)*1000:.1f}ms/batch)')

# 2) Compute only (single pre-loaded batch repeated)
print('\n--- Compute only (data pre-loaded) ---')
inputs, targets = next(iter(loader))
inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
model.train()
n_iters = len(loader)

if 'cuda' in DEVICE:
    torch.cuda.synchronize()
start = time.time()
for _ in tqdm(range(n_iters), desc='Compute only'):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
if 'cuda' in DEVICE:
    torch.cuda.synchronize()
elif DEVICE == 'mps':
    torch.mps.synchronize()
compute_time = time.time() - start
print(f'Compute: {compute_time:.2f}s ({compute_time/n_iters*1000:.1f}ms/batch)')

# 3) Full loop
print('\n--- Full training loop ---')
model.train()
if 'cuda' in DEVICE:
    torch.cuda.synchronize()
start = time.time()
for inputs, targets in tqdm(loader, desc='Full loop'):
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
if 'cuda' in DEVICE:
    torch.cuda.synchronize()
elif DEVICE == 'mps':
    torch.mps.synchronize()
total_time = time.time() - start
print(f'Full loop: {total_time:.2f}s ({total_time/len(loader)*1000:.1f}ms/batch)')

print(f'\n--- Summary ---')
print(f'Data loading:  {data_time:.2f}s ({data_time/total_time*100:.0f}%)')
print(f'Compute:       {compute_time:.2f}s ({compute_time/total_time*100:.0f}%)')
print(f'Overhead:      {total_time - data_time - compute_time:.2f}s (transfer, sync, etc.)')
