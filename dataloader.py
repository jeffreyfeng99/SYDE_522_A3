import torch.utils.data as data
from PIL import Image
import os
import pandas as pd
import numpy as np

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        df = pd.read_csv(data_list)

        self.img_paths = df['dir'].to_list()
        self.n_data = len(self.img_paths)

        if 'label2' in df.columns:
            self.img_labels = df['label2'].to_list()
        else: 
            self.img_labels = ['0' for i in range(len(self.img_paths))]

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item%self.n_data], self.img_labels[item%self.n_data]
        imgs = Image.open(os.path.join(self.root, img_paths).replace("\\","/")).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data