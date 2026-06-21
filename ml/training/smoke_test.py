import os
import yaml
from ml.data.generate_sample_dataset import generate_sample_dataset
from ml.training.train import train


def run_smoke():
    root = 'ml/data/sample'
    generate_sample_dataset(root=root, num_train=8, num_val=2, size=(256, 256))

    # write a small config that points to the sample data
    cfg = {
        'model': {
            'name': 'nvidia/mit-b2',
            'num_classes': 2
        },
        'training': {
            'batch_size': 2,
            'learning_rate': 6e-5,
            'epochs': 1,
            'use_cuda': False,
            'checkpoint_dir': 'ml/checkpoints',
            'dice_weight': 1.0,
            'topology_weight': 0.5,
            'topology_kernel': 5
        },
        'data': {
            'train_images': os.path.join(root, 'images', 'train'),
            'train_masks': os.path.join(root, 'masks', 'train'),
            'val_images': os.path.join(root, 'images', 'val'),
            'val_masks': os.path.join(root, 'masks', 'val')
        },
        'augmentation': {
            'synthetic_occlusion_prob': 0.4,
            'synthetic_occlusion_max': 0.25
        }
    }

    sample_cfg_path = 'ml/training/config_sample.yaml'
    with open(sample_cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f)

    # run one-epoch training on the generated data
    train(config_path=sample_cfg_path)


if __name__ == '__main__':
    run_smoke()
