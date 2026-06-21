SegFormer training helper

1. Prepare data

Place training images and masks under the paths configured in `ml/training/config.yaml`.
Images should be PNG or JPG RGB. Masks should be single-channel PNG with values 0 (background) and 255 (road/foreground).

2. Install dependencies

Recommended packages (add to your environment):

```
pip install torch torchvision transformers albumentations pillow tqdm pyyaml
```

3. Run training

```
python -m ml.training.train
```

Notes:
- The script expects available SegFormer model weights for the `model.name` in the config; you can change this to a local checkpoint.
- The training loop upsamples model logits to the mask size and uses combined CrossEntropy + Dice loss.
- For experimentation, adjust `ml/training/config.yaml`.
