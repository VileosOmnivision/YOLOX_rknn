#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from yolox.exp import Exp as MyExp

def calculate_optimal_epochs(dataset_size, batch_size, target_iterations=120000):
    """
    Calculate optimal number of epochs

    Args:
        dataset_size: Number of training images
        batch_size: Batch size for training
        target_iterations: Target total iterations (default: 120k)

    Returns:
        Recommended number of epochs
    """
    iterations_per_epoch = dataset_size / batch_size
    optimal_epochs = int(target_iterations / iterations_per_epoch)

    print(f"Dataset size: {dataset_size}")
    print(f"Batch size: {batch_size}")
    print(f"Iterations per epoch: {iterations_per_epoch:.0f}")
    print(f"Optimal epochs: {optimal_epochs}")
    print(f"Total iterations: {optimal_epochs * iterations_per_epoch:.0f}")

    return optimal_epochs



class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # Model architecture - YOLOX-M configuration
        self.num_classes = 2  # person, car
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Dataset configuration
        self.data_dir = "datasets/p_2"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # Training configuration
        self.batch_size = 16  # Will be total batch size
        calculate_optimal_epochs(9034, self.batch_size)
        self.max_epoch = 100
        self.data_num_workers = 4  # Reduced for small dataset
        self.eval_interval = 10  # Evaluate every 10 epochs
