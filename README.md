# AS-UNet Training Configuration Guide

This project is used for image segmentation model training. Dataset files should be placed in the `inputs` directory. The main training entry is `train.py`.

## Dataset Directory

`train.py` reads the training and validation data from `inputs/<dataset>` according to the `--dataset` argument. The default dataset name is `data_sum_moxizhen+bijie`.

The dataset directory should follow this structure:

```text
inputs/
+-- data_sum_moxizhen+bijie/
    +-- train/
    |   +-- images/
    |   +-- masks/
    +-- val/
        +-- images/
        +-- masks/
```

Directory description:

- `train/images`: training images.
- `train/masks`: training mask images.
- `val/images`: validation images.
- `val/masks`: validation mask images.

By default, both image files and mask files use the `.png` extension. The image and its corresponding mask should use the same file name.

## Run Example

```bash
python train.py
```

You can also modify the configuration from the command line:

```bash
python train.py --dataset data_sum_moxizhen+bijie --arch AS_UNet --epochs 150 --batch_size 16
```

During training, the current configuration is saved to:

```text
models/<name>/config.yml
```

The training log and best model weights are saved as:

```text
models/<name>/log.csv
models/<name>/model.pth
```

## Config Arguments in train.py

The `config` dictionary in `train.py` is generated from `parse_args()`. All arguments can be passed through the command line.

| Argument | Default | Description |
| --- | --- | --- |
| `--name` | `None` | Name of the model output directory. If not specified, it is automatically generated as `<dataset>_<arch>_wDS` or `<dataset>_<arch>_woDS`. |
| `--epochs` | `150` | Total number of training epochs. |
| `--batch_size`, `-b` | `16` | Mini-batch size. |
| `--arch`, `-a` | `AS_UNet` | Model architecture. Available options come from `__all__` in `archs.py`, including `UNet`, `NestedUNet`, and `AS_UNet`. |
| `--deep_supervision` | `False` | Whether to enable deep supervision. The current model creation code mainly uses the standard U-Net style interface. |
| `--input_channels` | `3` | Number of input image channels. RGB images usually use `3`. |
| `--num_classes` | `1` | Number of segmentation classes. Binary segmentation usually uses `1`. |
| `--input_w` | `128` | Width of the resized input image. |
| `--input_h` | `128` | Height of the resized input image. |
| `--loss` | `PolyGHMDiceLoss` | Loss function name. Available options come from `__all__` in `losses.py`, with additional support for `BCEWithLogitsLoss`. |
| `--dataset` | `data_sum_moxizhen+bijie` | Dataset folder name. The corresponding path is `inputs/<dataset>`. |
| `--img_ext` | `.png` | File extension of input images. |
| `--mask_ext` | `.png` | File extension of mask images. |
| `--optimizer` | `SGD` | Optimizer type. Available options are `Adam` and `SGD`. |
| `--lr`, `--learning_rate` | `1e-3` | Initial learning rate. |
| `--momentum` | `0.9` | Momentum parameter for the SGD optimizer. |
| `--weight_decay` | `1e-4` | Weight decay coefficient for regularization. |
| `--nesterov` | `False` | Whether to use Nesterov momentum for SGD. |
| `--scheduler` | `CosineAnnealingLR` | Learning rate scheduler. Available options are `CosineAnnealingLR`, `ReduceLROnPlateau`, `MultiStepLR`, and `ConstantLR`. |
| `--min_lr` | `1e-5` | Minimum learning rate, mainly used by learning rate schedulers. |
| `--factor` | `0.1` | Learning rate decay factor for `ReduceLROnPlateau`. |
| `--patience` | `2` | Patience value for `ReduceLROnPlateau`. |
| `--milestones` | `1,2` | Epoch milestones for `MultiStepLR`. Multiple epochs should be separated by commas. |
| `--gamma` | `2/3` | Learning rate decay ratio for `MultiStepLR`. |
| `--early_stopping` | `-1` | Early stopping patience. `-1` means early stopping is disabled. If set to a value greater than or equal to `0`, training stops when validation IoU does not improve for the specified number of epochs. |
| `--num_workers` | `0` | Number of DataLoader worker processes. `0` is usually more stable on Windows. |

## Common Configuration Examples

Use the default dataset and default AS_UNet:

```bash
python train.py
```

Change the input size:

```bash
python train.py --input_w 256 --input_h 256
```

Use the Adam optimizer:

```bash
python train.py --optimizer Adam --lr 0.001
```

When using another dataset, place it under `inputs/<new_dataset_name>` and specify:

```bash
python train.py --dataset new_dataset_name
```
