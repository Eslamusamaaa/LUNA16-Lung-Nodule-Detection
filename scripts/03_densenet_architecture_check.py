"""Inspect the 3D DenseNet-121: layers, parameter count and feature-map shapes."""
import os

BASE_DIR_DEFAULT = os.environ.get('LUNA16_DIR', 'LUNA16_Project')  # data and checkpoints folder

# %%
# !pip install SimpleITK monai tqdm matplotlib torchsummary -q

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
import os, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from monai.networks.nets import DenseNet121

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR       = BASE_DIR_DEFAULT
STAGE1_PATH    = os.path.join(BASE_DIR, 'best_stage1_detector.pth')
STAGE2_PATH    = os.path.join(BASE_DIR, 'best_stage2_fpr_model.pth')

print(f'Device          : {DEVICE}')
print(f'Stage 1 model   : {os.path.exists(STAGE1_PATH)}')
print(f'Stage 2 model   : {os.path.exists(STAGE2_PATH)}')

# %%
# ── Instantiate the model (same config used in Stage 1 & Stage 2) ─────────────
model = DenseNet121(
    spatial_dims = 3,      # 3-D convolutions throughout
    in_channels  = 1,      # single-channel CT (greyscale HU)
    out_channels = 2,      # binary: negative (0) vs nodule (1)
).to(DEVICE)

print(model)

# %% [markdown]
# ---
# ## Section 2 — Parameter Count & Memory Estimate

# %%
def count_parameters(m):
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_parameters(model)

print('=' * 50)
print('  DenseNet-121 (3-D) — Parameter Summary')
print('=' * 50)
print(f'  Total parameters      : {total:>12,}')
print(f'  Trainable parameters  : {trainable:>12,}')
print(f'  Non-trainable params  : {total - trainable:>12,}')
print()
# Memory estimate: each float32 param = 4 bytes
mem_mb = total * 4 / 1024**2
print(f'  Model weight memory   : {mem_mb:.1f} MB  (fp32)')
print(f'  Model weight memory   : {mem_mb/2:.1f} MB  (fp16 / AMP)')
print('=' * 50)

# %%
# ── Layer-by-layer breakdown ──────────────────────────────────────────────────
print(f'{'Layer name':<50} {'Params':>12}')
print('-' * 65)
total_shown = 0
for name, module in model.named_modules():
    n = sum(p.numel() for p in module.parameters(recurse=False))
    if n > 0:
        print(f'{name:<50} {n:>12,}')
        total_shown += n
print('-' * 65)
print(f'{'TOTAL':<50} {total_shown:>12,}')

# %% [markdown]
# ---
# ## Section 3 — Forward-Pass Smoke Test

# %%
model.eval()

# Simulate a batch of 4 patches — same shape used during training
dummy_input = torch.zeros(4, 1, 64, 64, 64, device=DEVICE)

with torch.no_grad():
    logits = model(dummy_input)
    probs  = torch.softmax(logits, dim=1)

print('Input  shape :', dummy_input.shape)
print('Logits shape :', logits.shape)
print('Probs  shape :', probs.shape)
print()
print('Sample logits (first item):', logits[0].cpu().numpy().round(4))
print('Sample probs  (first item):', probs[0].cpu().numpy().round(4))
print()
assert logits.shape == (4, 2), 'Unexpected output shape!'
print(' Forward pass OK')

# %% [markdown]
# ---
# ## Section 4 — Feature-Map Shape Trace
#
# Registers forward hooks on every Conv3d layer to record the output spatial dimensions as data flows through the network.

# %%
shape_log = []

def _hook(module, inp, out):
    shape_log.append((module.__class__.__name__, tuple(out.shape)))

hooks = []
for name, module in model.named_modules():
    if isinstance(module, nn.Conv3d):
        hooks.append(module.register_forward_hook(_hook))

dummy = torch.zeros(1, 1, 64, 64, 64, device=DEVICE)
with torch.no_grad():
    _ = model(dummy)

for h in hooks:
    h.remove()

print(f'{'Layer':<20} {'Output shape':>30}')
print('-' * 55)
for i, (name, shape) in enumerate(shape_log):
    print(f'{i:>3}. {name:<18} {str(shape):>30}')

# %% [markdown]
# ---
# ## Section 6 — Parameter Distribution Visualisation

# %%
# ── Collect per-block parameter counts ───────────────────────────────────────
block_names, block_params = [], []

for name, module in model.named_children():
    n = sum(p.numel() for p in module.parameters())
    block_names.append(name)
    block_params.append(n)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('DenseNet-121 (3-D) — Parameter Distribution', fontsize=14, fontweight='bold')

# Bar chart
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(block_names)))
bars = axes[0].barh(block_names, block_params, color=colors, edgecolor='white')
for bar, val in zip(bars, block_params):
    axes[0].text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                 f'{val/1e6:.2f}M', va='center', fontsize=9)
axes[0].set_xlabel('Parameters')
axes[0].set_title('Parameters per Top-Level Block', fontweight='bold')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x/1e6:.1f}M'))
axes[0].grid(axis='x', alpha=0.3)

# Pie chart
axes[1].pie(block_params, labels=block_names, autopct='%1.1f%%',
            colors=colors, startangle=140,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.2})
axes[1].set_title('Share of Total Parameters', fontweight='bold')

plt.tight_layout()
plt.savefig('densenet121_param_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved → densenet121_param_distribution.png')

# %% [markdown]
# ---
# ## Section 8 — Stage 1 vs Stage 2 Head Comparison

# %%
def load_model(path, label):
    m = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
    if os.path.exists(path):
        state = torch.load(path, map_location='cpu')
        m.load_state_dict(state)
        print(f'  {label} loaded from {path}')
    else:
        print(f'   {label} not found at {path} — using random weights for demo')
    m.eval()
    return m

model_s1 = load_model(STAGE1_PATH, 'Stage 1 Detector')
model_s2 = load_model(STAGE2_PATH, 'Stage 2 FPR')

# %%
# Compare the final classifier (out.weight)
w1 = None
w2 = None
for name, p in model_s1.named_parameters():
    if 'out' in name and 'weight' in name:
        w1 = p.data.cpu()
for name, p in model_s2.named_parameters():
    if 'out' in name and 'weight' in name:
        w2 = p.data.cpu()

if w1 is not None and w2 is not None:
    print('Classifier head weight shape:', w1.shape)
    print()
    print(f'Stage 1 — mean: {w1.mean():.5f}  std: {w1.std():.5f}')
    print(f'Stage 2 — mean: {w2.mean():.5f}  std: {w2.std():.5f}')
    cosine_sim = torch.nn.functional.cosine_similarity(
        w1.view(1,-1), w2.view(1,-1)).item()
    print(f'Cosine similarity (S1 vs S2 head): {cosine_sim:.4f}')
    print('(A value close to 0 confirms the two heads have diverged via independent training.)')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle('Stage 1 vs Stage 2 — Final Classifier Weight Distribution', fontweight='bold')

for ax, w, label, color in zip(
        axes,
        [w1, w2],
        ['Stage 1 Detector', 'Stage 2 FPR Model'],
        ['#3A7DC9', '#E07B54']):
    ax.hist(w.numpy().flatten(), bins=60, color=color, edgecolor='white', linewidth=0.5)
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Count')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('densenet121_head_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved → densenet121_head_comparison.png')

# %%
print('=' * 58)
print('  DenseNet-121 Architecture Notebook — Complete')
print('=' * 58)
print(f'  Architecture   : 3-D DenseNet-121 (MONAI)')
print(f'  Input patch    : 1 × 64 × 64 × 64')
print(f'  Output         : 2-class logits (neg / nodule)')
total, _ = count_parameters(model)
print(f'  Parameters     : {total:,}  (~{total*4/1024**2:.1f} MB fp32)')
print(f'  Stage 1 model  : best_stage1_detector.pth')
print(f'  Stage 2 model  : best_stage2_fpr_model.pth')
print('=' * 58)

