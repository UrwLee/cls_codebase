from re import T
import torch
import argparse
import os
import sys
import cv2
import time

class BaseConfiguration():
    def __init__(self):
        self.EXP_NAME = 'base'

        self.TRAIN_IMAGE_DIR = '/mnt/bd/dmx-workshop/data/largefinefoodai-iccv-recognition/Train'
        self.TRAIN_LAYOUT_PATH = '/mnt/bd/dmx-workshop/data/largefinefoodai-iccv-recognition/label/train_finetune.txt'
        self.VAL_IMAGE_DIR = '/mnt/bd/dmx-workshop/data/largefinefoodai-iccv-recognition/Val'
        self.VAL_LAYOUT_PATH = '/mnt/bd/dmx-workshop/data/largefinefoodai-iccv-recognition/label/val_finetune.txt'
        self.SAVE_DIR = '/mnt/bd/dmx-workshop/exps/kaggle'
        self.DIR_RESULT = os.path.join(self.SAVE_DIR, 'result', self.EXP_NAME)
        self.DIR_CKPT = os.path.join(self.DIR_RESULT, 'ckpt')
        self.DIR_LOG = os.path.join(self.DIR_RESULT, 'log')
        self.DIR_IMG_LOG = os.path.join(self.DIR_RESULT, 'log', 'img')
        self.DIR_EVALUATION = os.path.join(self.DIR_RESULT, 'eval')

        self.DATA_WORKERS = 2
        self.DATA_MIN_SCALE_FACTOR = 1.
        self.DATA_MAX_SCALE_FACTOR = 1.3
        self.DATA_SHORT_EDGE_LEN = 384
        self.DATA_RANDOMCROP = (384, 384)
        self.DATA_RANDOMFLIP = 0.5

        self.PRETRAIN = True

        self.MODEL_NUM_CLASSES = 1000
        self.MODEL_GCT_BETA_WD = True

        self.TRAIN_TOTAL_STEPS = 500000
        self.TRAIN_START_STEP = 1
        self.TRAIN_EVAL_STEPS = 5000
        self.TRAIN_LR = 0.01
        self.TRAIN_MOMENTUM = 0.9
        self.TRAIN_COSINE_DECAY = False
        self.TRAIN_WARM_UP_STEPS = 1000
        self.TRAIN_WEIGHT_DECAY = 15e-5
        self.TRAIN_POWER = 0.9
        self.TRAIN_GPUS = 4
        self.TRAIN_BATCH_SIZE = 64
        self.TRAIN_LOG_STEP = 20
        self.TRAIN_CLIP_GRAD_NORM = 5.
        self.TRAIN_SAVE_STEP = 2000
        self.TRAIN_RESUME = False
        self.TRAIN_RESUME_CKPT = None
        self.TRAIN_RESUME_STEP = 0
        self.TRAIN_AUTO_RESUME = True
        self.TRAIN_MAX_KEEP_CKPT = 8

        # self.TEST_GPU_ID = 0
        # self.TEST_DATASET = 'youtubevos'
        # self.TEST_DATASET_FULL_RESOLUTION = False
        # self.TEST_DATASET_SPLIT = ['val']
        # self.TEST_CKPT_PATH = None
        # self.TEST_CKPT_STEP = None  # if "None", evaluate the latest checkpoint.
        # self.TEST_FLIP = False
        # self.TEST_MULTISCALE = [1]
        # self.TEST_MIN_SIZE = None
        # self.TEST_MAX_SIZE = 800 * 1.3 if self.TEST_MULTISCALE == [1.] else 800
        # self.TEST_WORKERS = 4

        # dist
        self.DIST_ENABLE = True
        self.DIST_BACKEND = 'nccl'
        self.DIST_URL = "file:///tmp/sharefile"
        self.DIST_START_GPU = 0

        self.__check()

    def __check(self):
        if not torch.cuda.is_available():
                raise ValueError('config.py: cuda is not avalable')
        if self.TRAIN_GPUS == 0:
                raise ValueError('config.py: the number of GPU is 0')
        for path in [self.DIR_RESULT, self.DIR_CKPT, self.DIR_LOG, self.DIR_EVALUATION, self.DIR_IMG_LOG]:
            if not os.path.isdir(path):
                os.makedirs(path)

cfg = BaseConfiguration()
