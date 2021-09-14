import os, sys
import random
import numpy as np
import json
import cv2
import pprint
from PIL import Image

import torch
from torch.utils import data
CURR_DIR = os.path.dirname(__file__)

random.seed(0); np.random.seed(0); torch.manual_seed(0)


class BaseDataset(data.Dataset):
    def __init__(self, image_dir,layout_path, phase, transform=None,cfg = None):
        self.image_dir = image_dir
        self.layout_path = layout_path
        self.phase = phase
        self.cfg = cfg
        assert self.phase in ["train", "val"], f"Phase {phase} must be `train`/`val`!"

        with open(self.layout_path, "r") as f:
            meta = f.readlines()
            self.im_paths = [i.strip().split(' ')[0] for i in meta]
            self.labels = [i.strip().split(' ')[1] for i in meta]

        self.len = len(self.im_paths)
        print('%s: %i images'%(phase,self.len))
        self.transform = transform
    
        pprint.pprint({
            'image_dir': image_dir,
            'layout_path':layout_path,
            'phase': self.phase
        }, indent=1)
    
    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        if self.phase == "train":
            ret = self.get_train_item(index)
        else:
            ret = self.get_val_item(index)
        return ret

    def get_val_item(self, index):
        img_id = self.im_paths[index]
        label = int(self.labels[index])

        img_path = os.path.join(self.image_dir, img_id)
        img = cv2.imread(img_path)

        if self.transform is not None:
            img = self.transform(img)
            label = torch.Tensor([label]).long()

        ret = {
            "image": img,
            "label": label,
            "path": img_path
        }
        return ret

    def get_train_item(self, index):        
        img_id = self.im_paths[index]
        label = int(self.labels[index])
        
        img_path = os.path.join(self.image_dir, img_id)
        img = cv2.imread(img_path)
        if self.transform is not None:
            img = self.transform(img)
            label = torch.Tensor([label]).long()

        ret = {
            "image": img,
            "label": label,
            "path": img_path
        }
        return ret



if __name__ == "__main__":
    from torchvision import transforms
    import custom_transforms as tr
    composed_transforms = transforms.Compose([
            tr.RandomScale(1.,1.3,640),
            tr.RandomCrop(640),
            tr.RandomHorizontalFlip(0.5),
            tr.Resize(512),
            tr.ToTensor()])
    val_transforms = transforms.Compose([
            tr.Resize(512),
            tr.ToTensor()])

    ds_train = BaseDataset(
        "/mnt/bd/dmx-workshop/data/largefinefoodai-iccv-recognition/Val",
        "/mnt/bd/dmx-workshop/data/largefinefoodai-iccv-recognition/label/val_finetune.txt",
        phase='train',
        transform=composed_transforms,
        cfg=None)

    ds_val = BaseDataset(
        "/mnt/bd/dmx-workshop/data/largefinefoodai-iccv-recognition/Val",
        "/mnt/bd/dmx-workshop/data/largefinefoodai-iccv-recognition/label/val_finetune.txt",
        phase='val',
        transform=composed_transforms,
        cfg=None)

    train_loader = data.DataLoader(dataset=ds_train, batch_size=1)

    def th2np(th, dtype='image', transpose=False, rgb_cyclic=False):
        assert dtype in ['image', 'mask']
        if dtype == 'image':
            th = (th + 1.0) / 2.0
            th = th * 255
            npdata = th.detach().cpu().numpy()      # NCHW
            if rgb_cyclic:
                npdata = npdata[:, ::-1, :, :]
            if transpose:
                npdata = npdata.transpose((0, 2, 3, 1)) # NHWC
        else:
            if th.ndim == 3:
                th = th.unsqueeze(1)
            if th.size(1) == 1:
                npdata = th.detach().cpu().repeat(1, 3, 1, 1).numpy()   # NCHW
            else:
                npdata = th.detach().cpu().numpy()
            if transpose:
                npdata = npdata.transpose((0, 2, 3, 1))
        return npdata

    for i, s in enumerate(train_loader):
        image = s["image"]
        label = s["label"]
        
        print(image.shape, label)
        image_np = th2np(image, dtype='image', transpose=True, rgb_cyclic=False)
        image_np = np.transpose(image_np, (1, 0, 2, 3))
        image_np = image_np.reshape(512, -1, 3).copy()
        for ii in range(len(label)):
            cv2.putText(image_np, str(label), (20+ii*512, 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), thickness=2)
        cv2.imwrite(f"image_{i}.png", image_np)

        print(image_np.shape)
        if i > 5:
            break


    exit()






















