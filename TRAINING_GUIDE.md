# Training Custom YOLOX Models - Quick Guide

This guide explains how to train YOLOX on your custom dataset.

---

## Why Each Step is Needed

### 1. **Convert YOLO to COCO Format** (`convert_yolo_to_coco.py`)

**Why?** YOLOX natively supports only COCO and VOC dataset formats, not YOLO format.

**What it does:**
- Converts YOLO annotations (`.txt` files with normalized coordinates) to COCO JSON format
- YOLO format: `class_name x_center y_center width height` (normalized 0-1)
- COCO format: JSON with absolute pixel coordinates and structured metadata

**When to use:** When your dataset has `.txt` annotation files (YOLO format)

---

### 2. **Download Pretrained Weights** (`download_weights.py`)

**Why?** Transfer learning - starting from pretrained weights is much better than training from scratch.

**What it does:**
- Downloads COCO-pretrained weights for YOLOX models
- These weights learned to detect 80 COCO classes from millions of images
- Your custom model will fine-tune these weights for your specific classes

**Benefits:**
- Faster convergence (less training time)
- Better performance (especially with small datasets)
- Required for production-quality results

**Note:** YOLOX automatically handles the head mismatch (80 COCO classes → your N classes)

---

### 3. **Create Custom Exp File** (e.g., `yolox_m_parking.py`)

**Why?** The Exp file is YOLOX's configuration system - it controls EVERYTHING about training.

**What it configures:**
- **Model architecture**: depth/width factors determine model size (S/M/L/X)
  - YOLOX-S: `depth=0.33, width=0.50`
  - YOLOX-M: `depth=0.67, width=0.75`
  - YOLOX-L: `depth=1.00, width=1.00`
  - YOLOX-X: `depth=1.33, width=1.25`

- **Dataset settings**:
  - `num_classes`: Number of your custom classes
  - `data_dir`: Path to your dataset
  - `train_ann/val_ann`: Annotation JSON files

- **Training hyperparameters**:
  - `max_epoch`: How long to train
  - `basic_lr_per_img`: Learning rate
  - `warmup_epochs`: Gradual learning rate warmup
  - `input_size`: Image size (e.g., 640x640)

- **Data augmentation**:
  - `mosaic_prob`: Mosaic augmentation probability
  - `mixup_prob`: MixUp augmentation probability
  - `flip_prob`: Horizontal flip probability

**Instead of:** Passing 20+ command-line arguments, you configure once in Python.

---

## Step-by-Step Training Workflow

### **Setup (One-time)**

```bash
# 1. Install YOLOX
pip install -v -e .

# 2. Organize your dataset
datasets/
  your_dataset/
    image1.jpg
    image1.txt      # YOLO format annotations
    image2.jpg
    image2.txt
    ...
```

### **Prepare Dataset**

**IMPORTANT:** YOLOX expects COCO dataset structure:

```
datasets/your_dataset/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── val2017/
    ├── image_val1.jpg
    └── ...
```

```bash
# 3. Convert YOLO to COCO format and organize into train/val splits
python tools/convert_yolo_to_coco.py --dataset-dir datasets/gt_v1/ground-truth --output-dir datasets/p_1 --classes person car --train-split 0.8

# This will:
# - Create train2017/ and val2017/ folders
# - Split images 80% train, 20% validation (shuffled randomly)
# - Generate annotations/instances_train2017.json
# - Generate annotations/instances_val2017.json
# - Move images to appropriate folders
```

### **Download Pretrained Weights**

```bash
# 4. Download pretrained weights (choose one)
python tools/download_weights.py  # Downloads YOLOX-M by default

# Or manually download from:
# https://github.com/Megvii-BaseDetection/YOLOX/releases
# Available models: yolox_s.pth, yolox_m.pth, yolox_l.pth, yolox_x.pth
```

### **Create Exp File**

```python
# 5. Create exps/example/custom/yolox_m_yourdata.py

from yolox.exp import Exp as MyExp
import os

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # Model size (YOLOX-M)
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # YOUR DATASET CONFIG
        self.num_classes = 2  # Change to your number of classes
        self.data_dir = "datasets/your_dataset"
        self.train_ann = "annotations.json"
        self.val_ann = "annotations.json"  # Can be same file for small datasets

        # Training config (adjust based on dataset size)
        self.max_epoch = 100  # More epochs for small datasets
        self.eval_interval = 10
        self.input_size = (640, 640)
```

### **Start Training**

```bash
# 6. Train the model
python tools/train.py -f exps/example/custom/yolox_m_p1.py -d 1 -b 8 -c weights/yolox_m.pth --fp16

# Arguments:
#   -f: Your custom Exp file
#   -d: Number of GPUs (1 for single GPU/CPU)
#   -b: Batch size (adjust based on GPU memory)
#   -c: Pretrained checkpoint path
#   --fp16: Mixed precision training (faster, less memory)
```

### **Monitor Training**

```bash
# Training outputs go to:
YOLOX_outputs/yolox_m_yourdata/

# View logs with TensorBoard:
tensorboard --logdir YOLOX_outputs/yolox_m_yourdata
```

---

## Common Adjustments

### For Small Datasets (< 100 images)
```python
self.max_epoch = 200  # Train longer
self.mixup_prob = 0.5  # Reduce aggressive augmentation
self.eval_interval = 10
```

### For Large Datasets (> 10,000 images)
```python
self.max_epoch = 300
self.mixup_prob = 1.0
self.eval_interval = 10
self.data_num_workers = 8  # More workers for data loading
```

### For Different Model Sizes
```python
# Nano (mobile)
self.depth = 0.33
self.width = 0.25

# Small
self.depth = 0.33
self.width = 0.50

# Medium
self.depth = 0.67
self.width = 0.75

# Large
self.depth = 1.00
self.width = 1.00

# Extra Large
self.depth = 1.33
self.width = 1.25
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Convert dataset | `python tools/convert_yolo_to_coco.py --dataset-dir datasets/X --output-dir datasets/X --classes A B C --train-split 0.8` |
| Download weights | `python tools/download_weights.py` |
| Train | `python tools/train.py -f exps/example/custom/your_exp.py -d 1 -b 8 -c weights/yolox_m.pth` |
| Metrics | `tensorboard --logdir=YOLOX_outputs/yolox_m_p2/tensorboard --port=6006` |
| Evaluate | `python tools/eval.py -f exps/example/custom/your_exp.py -c YOLOX_outputs/.../best_ckpt.pth` |
| Export ONNX | `python tools/export_onnx.py -f exps/example/custom/your_exp.py -c YOLOX_outputs/.../best_ckpt.pth` |
| Demo | `python tools/demo.py image -f exps/example/custom/your_exp.py -c YOLOX_outputs/.../best_ckpt.pth --path image.jpg` |
| Compare models | `python tools/compare_models.py  -f1 exps/default/yolox_m.py  -c1 weights/yolox_m.pth -f2 exps/example/custom/yolox_m_p1.py -c2 YOLOX_outputs/yolox_m_p1/best_ckpt.pth --ann_file datasets/gt_to_test/annotations/instances_val2017.json --data_dir datasets/gt_to_test/val2017/ --save_error` |
---

## Example: Training on Dataset

This is what we did for your dataset:

```bash
# 1. Convert YOLO annotations to COCO and organize structure
python tools/convert_yolo_to_coco.py \
  --dataset-dir datasets/raw \
  --output-dir datasets/prepared \
  --classes person car \
  --train-split 0.8

# Note: For very small datasets (< 10 images), you can use all data for both:
# --train-split 1.0  (uses all images for training and validation)

# 2. Download YOLOX-M pretrained weights
python tools/download_weights.py

# 3. Created: exps/example/custom/yolox_m_p1.py
#    - num_classes = 2 (person, car)
#    - depth = 0.67, width = 0.75 (YOLOX-M)
#    - max_epoch = 100 (small dataset)

# 4. Train
python tools/train.py \
  -f exps/example/custom/yolox_m_p1.py \
  -d 1 \
  -b 2 \
  -c weights/yolox_m.pth \
  --fp16
```

---

## Troubleshooting

**Out of memory:**
- Reduce batch size: `-b 2` or `-b 1`
- Remove `--fp16` flag
- Use smaller model (YOLOX-S instead of YOLOX-M)

**Training too slow:**
- Add `--fp16` flag
- Increase `data_num_workers` in Exp file
- Use smaller input size: `self.input_size = (416, 416)`

**Poor performance:**
- Train longer: increase `max_epoch`
- Use pretrained weights: `-c weights/yolox_m.pth`
- Collect more training data
- Check data quality and annotations

---

**Remember:** The Exp file is the heart of YOLOX configuration. All training parameters live there!
