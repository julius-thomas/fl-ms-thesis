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
parser.add_argument('--precomputed', type=int, nargs='?', const=224, default=None,
                    help='use CheXpert/png_<size>/ cache; skips runtime Resize')
parser.add_argument('--num_workers', type=int, nargs='+', default=[0],
                    help='DataLoader worker counts to sweep (e.g. --num_workers 0 2 4). Compute-only phase runs once; data-loading and full-loop phases repeat per value.')
args = parser.parse_args()

DEVICE = args.device
NUM_CLASSES = 14
chexpert_root = os.path.join(args.data_path, 'CheXpert')
if args.precomputed is not None:
    args.resize = args.precomputed
    image_root = os.path.join(chexpert_root, f'png_{args.precomputed}')
    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f'--precomputed={args.precomputed} but {image_root} does not exist. '
            f'Run external-data/CheXpert/precompute_resized.py --size {args.precomputed} first.'
        )
else:
    image_root = chexpert_root

transform_stages = []
if args.precomputed is None:
    transform_stages.append(torchvision.transforms.Resize((args.resize, args.resize)))
transform_stages.extend([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5]*3, [0.5]*3),
])
transform = torchvision.transforms.Compose(transform_stages)

print(f'Loading dataset (rawsmpl={args.rawsmpl}, precomputed={args.precomputed})...')
ds = CheXpertDataset(
    root=image_root,
    csv_file=os.path.join(chexpert_root, 'train.csv'),
    transform=transform,
    raw_fraction=args.rawsmpl,
    precomputed=args.precomputed is not None,
)
ds = CheXpertWrapper(ds, 'CHEXPERT', 'CLIENT')

def make_loader(nw):
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                       pin_memory='cuda' in DEVICE,
                                       num_workers=nw)

print(f'Dataset: {len(ds)} samples, batch_size={args.batch_size}')
print(f'Device: {DEVICE}, bf16: {args.bf16}, hidden_size: {args.hidden_size}')
print(f'num_workers sweep: {args.num_workers}')

model = ResNet18(in_channels=3, hidden_size=args.hidden_size, num_classes=NUM_CLASSES).to(DEVICE)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Warmup with workers=0 to isolate from the sweep
warmup_loader = make_loader(0)
print('Warming up...')
for i, (inputs, targets) in enumerate(warmup_loader):
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if i >= 4:
        break
del warmup_loader

# Compute-only is independent of num_workers: measure once.
print('\n--- Compute only (data pre-loaded, independent of num_workers) ---')
probe_loader = make_loader(0)
n_iters = len(probe_loader)
inputs, targets = next(iter(probe_loader))
del probe_loader
inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
model.train()

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

# Sweep num_workers for data-loading and full-loop phases.
results = []
for nw in args.num_workers:
    print(f'\n================ num_workers={nw} ================')
    loader = make_loader(nw)
    n_batches = len(loader)

    print('--- Data loading only ---')
    start = time.time()
    for inputs, targets in tqdm(loader, desc=f'nw={nw} load'):
        pass
    data_time = time.time() - start
    print(f'Data loading: {data_time:.2f}s ({data_time/n_batches*1000:.1f}ms/batch)')

    print('--- Full training loop ---')
    model.train()
    if 'cuda' in DEVICE:
        torch.cuda.synchronize()
    start = time.time()
    for inputs, targets in tqdm(loader, desc=f'nw={nw} full'):
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
    print(f'Full loop: {total_time:.2f}s ({total_time/n_batches*1000:.1f}ms/batch)')
    results.append((nw, data_time, total_time, n_batches))

    del loader

print(f'\n--- Summary (compute {compute_time:.2f}s, {compute_time/n_iters*1000:.1f}ms/batch) ---')
print(f'{"nw":>4} | {"data (s)":>9} | {"data ms/b":>10} | {"full (s)":>9} | {"full ms/b":>10} | {"data %":>7}')
for nw, data_time, total_time, n_batches in results:
    print(f'{nw:>4} | {data_time:>9.2f} | {data_time/n_batches*1000:>10.1f} | '
          f'{total_time:>9.2f} | {total_time/n_batches*1000:>10.1f} | '
          f'{data_time/total_time*100:>6.0f}%')
