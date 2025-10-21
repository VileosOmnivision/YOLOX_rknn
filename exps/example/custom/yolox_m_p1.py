#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # Model architecture - YOLOX-M configuration
        self.num_classes = 2  # person, car
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Dataset configuration
        self.data_dir = "datasets/p_1"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # Training configuration
        self.max_epoch = 100  # Reduced for small dataset
        self.data_num_workers = 4  # Reduced for small dataset
        self.eval_interval = 10  # Evaluate every 10 epochs

        # Batch size (adjust based on your GPU memory)
        # For testing without GPU, use small batch
        self.batch_size = 16  # Will be total batch size
