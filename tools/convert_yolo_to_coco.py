#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Convert YOLO format annotations to COCO format for YOLOX training
with automatic train/validation split and proper directory structure
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from PIL import Image


def convert_yolo_to_coco(dataset_dir, output_dir, class_names, train_split=0.8, shuffle_seed=42):
    """
    Convert YOLO format dataset to COCO format with train/val split

    Args:
        dataset_dir: Directory containing images and .txt annotation files
        output_dir: Output directory for COCO-structured dataset
        class_names: List of class names in order
        train_split: Fraction of data for training (0.0-1.0)
        shuffle_seed: Random seed for reproducible splits
    """

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    # Create output directories following COCO structure
    train_img_dir = output_dir / "train2017"
    val_img_dir = output_dir / "val2017"
    annotations_dir = output_dir / "annotations"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Initialize COCO format dictionaries
    train_coco = {
        "info": {
            "description": "Custom YOLOX Dataset - Training Set",
            "version": "1.0",
            "year": 2024,
            "contributor": "YOLO to COCO Converter",
            "date_created": "2024"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    val_coco = {
        "info": {
            "description": "Custom YOLOX Dataset - Validation Set",
            "version": "1.0",
            "year": 2024,
            "contributor": "YOLO to COCO Converter",
            "date_created": "2024"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create categories (same for both train and val)
    for idx, class_name in enumerate(class_names):
        category = {
            "id": idx,
            "name": class_name,
            "supercategory": "object"
        }
        train_coco["categories"].append(category)
        val_coco["categories"].append(category)

    # Create class name to ID mapping
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    # Find all image files
    image_files = []
    for ext in ["*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]:
        image_files.extend(list(dataset_dir.glob(ext)))

    if not image_files:
        print(f"ERROR: No image files found in {dataset_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Shuffle and split images
    random.seed(shuffle_seed)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * train_split)
    train_images = image_files[:split_idx] if train_split < 1.0 else image_files
    val_images = image_files[split_idx:] if train_split < 1.0 else image_files

    print(f"Split: {len(train_images)} training, {len(val_images)} validation")

    # Process training images
    print("\nProcessing training images...")
    train_annotation_id = 1
    for train_image_id, image_file in enumerate(train_images, start=1):
        train_annotation_id = process_image(
            image_file, train_image_id, train_annotation_id,
            class_to_id, train_coco, train_img_dir
        )

    # Process validation images
    print("Processing validation images...")
    val_annotation_id = 1
    for val_image_id, image_file in enumerate(val_images, start=1):
        val_annotation_id = process_image(
            image_file, val_image_id, val_annotation_id,
            class_to_id, val_coco, val_img_dir
        )

    # Save COCO format JSON files
    train_json_path = annotations_dir / "instances_train2017.json"
    val_json_path = annotations_dir / "instances_val2017.json"

    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f, indent=2)

    with open(val_json_path, 'w') as f:
        json.dump(val_coco, f, indent=2)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print("Training set:")
    print(f"  Images: {len(train_coco['images'])}")
    print(f"  Annotations: {len(train_coco['annotations'])}")
    print(f"  Output: {train_json_path}")
    print("\nValidation set:")
    print(f"  Images: {len(val_coco['images'])}")
    print(f"  Annotations: {len(val_coco['annotations'])}")
    print(f"  Output: {val_json_path}")
    print("\nDataset structure:")
    print(f"  {output_dir}/")
    print("  ├── annotations/")
    print("  │   ├── instances_train2017.json")
    print("  │   └── instances_val2017.json")
    print(f"  ├── train2017/ ({len(train_coco['images'])} images)")
    print(f"  └── val2017/ ({len(val_coco['images'])} images)")


def process_image(image_file, image_id, annotation_id, class_to_id, coco_dict, output_img_dir):
    """
    Process a single image and its annotations

    Returns:
        Updated annotation_id for next image
    """
    # Get corresponding annotation file
    txt_file = image_file.with_suffix('.txt')

    if not txt_file.exists():
        print(f"Warning: No annotation file for {image_file.name}, skipping...")
        return annotation_id

    # Get image dimensions
    try:
        img = Image.open(image_file)
        img_width, img_height = img.size
        img.close()
    except Exception as e:
        print(f"Error reading image {image_file.name}: {e}, skipping...")
        return annotation_id

    # Copy image to output directory
    output_image_path = output_img_dir / image_file.name
    try:
        if image_file != output_image_path:  # Avoid copying to itself
            shutil.copy2(image_file, output_image_path)
    except Exception as e:
        print(f"Error copying image {image_file.name}: {e}")
        return annotation_id

    # Add image info
    coco_dict["images"].append({
        "id": image_id,
        "file_name": image_file.name,
        "width": img_width,
        "height": img_height
    })

    # Read and convert annotations
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            class_name = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            if class_name not in class_to_id:
                print(f"Warning: Unknown class '{class_name}' in {txt_file.name}, skipping...")
                continue

            # Convert from normalized YOLO format to COCO format
            # YOLO: x_center, y_center, width, height (normalized)
            # COCO: x_min, y_min, width, height (absolute)
            x_min = (x_center - width / 2) * img_width
            y_min = (y_center - height / 2) * img_height
            box_width = width * img_width
            box_height = height * img_height

            coco_dict["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_to_id[class_name],
                "bbox": [x_min, y_min, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0
            })

            annotation_id += 1

    return annotation_id


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO format to COCO format with train/val split',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with 80/20 train/val split
  python convert_yolo_to_coco.py --dataset-dir data/raw --output-dir data/coco --classes person car --train-split 0.8

  # Use all data for both training and validation (small datasets)
  python convert_yolo_to_coco.py --dataset-dir data/raw --output-dir data/coco --classes person car --train-split 1.0

  # Custom split with different seed
  python convert_yolo_to_coco.py --dataset-dir data/raw --output-dir data/coco --classes person car --train-split 0.7 --seed 123
        """
    )

    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Directory containing images and .txt annotation files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for COCO-structured dataset')
    parser.add_argument('--classes', type=str, nargs='+', required=True,
                        help='Class names in order (space-separated)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Fraction of data for training (0.0-1.0). Use 1.0 to use all data for both train and val (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits (default: 42)')

    args = parser.parse_args()

    # Validate arguments
    if args.train_split < 0.0 or args.train_split > 1.0:
        parser.error("--train-split must be between 0.0 and 1.0")

    print(f"{'='*60}")
    print("YOLO to COCO Converter")
    print(f"{'='*60}")
    print(f"Input directory: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Classes: {', '.join(args.classes)}")
    print(f"Train split: {args.train_split:.1%}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*60}\n")

    convert_yolo_to_coco(
        args.dataset_dir,
        args.output_dir,
        args.classes,
        args.train_split,
        args.seed
    )


if __name__ == "__main__":
    main()
