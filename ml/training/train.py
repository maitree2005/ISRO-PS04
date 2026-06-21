import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm

from ml.data.dataset import RoadSegmentationDataset
from ml.models.losses import CombinedLoss
import albumentations as A

from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import torch.nn.functional as F


def load_config(path: str = 'ml/training/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(p=0.3)
    ])


def train(config_path: str = 'ml/training/config.yaml'):
    cfg = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['training'].get('use_cuda', False) else 'cpu')

    # Dataset
    # build transforms using augmentation parameters from config
    from ml.data.augmentations import get_transforms
    aug_cfg = cfg.get('augmentation', {})
    transforms = get_transforms(occlusion_prob=aug_cfg.get('synthetic_occlusion_prob', 0.4),
                                max_occlusion=aug_cfg.get('synthetic_occlusion_max', 0.25))
    train_ds = RoadSegmentationDataset(cfg['data']['train_images'], cfg['data']['train_masks'], transforms=transforms)
    val_ds = RoadSegmentationDataset(cfg['data']['val_images'], cfg['data']['val_masks'], transforms=None)

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=2)

    # Processor + model from transformers, with fallback to tiny UNet
    processor = None
    model = None
    try:
        processor = AutoImageProcessor.from_pretrained(cfg['model']['name'])
        model = AutoModelForSemanticSegmentation.from_pretrained(cfg['model']['name'], num_labels=cfg['model'].get('num_classes', 2))
        print('Loaded transformers model:', cfg['model']['name'])
    except Exception as e:
        print('Warning: Failed to load SegFormer model from transformers:', e)
        print('Falling back to tiny UNet model for local training (no processor).')
        from ml.models.tiny_unet import get_tiny_unet
        model = get_tiny_unet(num_classes=cfg['model'].get('num_classes', 2), in_channels=3)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=cfg['training']['learning_rate'])

    criterion = CombinedLoss(ce_weight=1.0, dice_weight=cfg['training'].get('dice_weight', 1.0))
    # optional topology continuity loss
    topo_w = cfg['training'].get('topology_weight', 0.0)
    topo_k = cfg['training'].get('topology_kernel', 5)
    if topo_w and topo_w > 0.0:
        criterion.enable_topology(weight=topo_w, kernel_size=topo_k)

    best_val = float('inf')
    ckpt_dir = cfg['training'].get('checkpoint_dir', 'ml/checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    num_epochs = cfg['training']['epochs']
    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Train')
        for images, masks in loop:
            # images: numpy array batch or list; ensure list of HWC uint8
            if isinstance(images, np.ndarray):
                images_list = [img for img in images]
            else:
                images_list = images

            # Prepare pixel_values either via processor or manual tensor conversion
            if processor is not None:
                pixel_values = processor(images=images_list, return_tensors='pt')['pixel_values'].to(device)
            else:
                # images_list: list of HWC uint8 arrays
                batch = np.stack(images_list, axis=0).astype('float32') / 255.0
                # to torch BCHW
                pixel_values = torch.tensor(batch).permute(0, 3, 1, 2).to(device)

            # masks -> tensor (B, H, W)
            masks_tensor = torch.tensor(np.array(masks), dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits  # (B, C, Hout, Wout)

            # Upsample logits to mask size
            logits_up = F.interpolate(logits, size=masks_tensor.shape[1:], mode='bilinear', align_corners=False)

            loss = criterion(logits_up, masks_tensor)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=float(loss.cpu().detach().numpy()))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            vloop = tqdm(val_loader, desc='Validation')
            for images, masks in vloop:
                if isinstance(images, np.ndarray):
                    images_list = [img for img in images]
                else:
                    images_list = images

                if processor is not None:
                    pixel_values = processor(images=images_list, return_tensors='pt')['pixel_values'].to(device)
                else:
                    batch = np.stack(images_list, axis=0).astype('float32') / 255.0
                    pixel_values = torch.tensor(batch).permute(0, 3, 1, 2).to(device)

                masks_tensor = torch.tensor(np.array(masks), dtype=torch.long, device=device)
                # model may expect pixel_values keyword or raw tensor
                try:
                    outputs = model(pixel_values=pixel_values)
                except Exception:
                    outputs = model(pixel_values) if callable(model) else model(pixel_values)
                logits = outputs.logits
                logits_up = F.interpolate(logits, size=masks_tensor.shape[1:], mode='bilinear', align_corners=False)
                l = criterion(logits_up, masks_tensor)
                val_loss += float(l.cpu().numpy())
        val_loss = val_loss / max(1, len(val_loader))
        print(f'Epoch {epoch+1} validation loss: {val_loss:.4f}')

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(ckpt_dir, f"segformer_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print('Saved best checkpoint:', ckpt_path)


if __name__ == '__main__':
    train()
