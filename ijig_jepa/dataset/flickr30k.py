import os
from typing import Callable, Union

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class Flickr30k(Dataset):

    ANNOTATIONS = r'D:\Datasets\Flickr30k\flickr_annotations_30k.csv'
    DATA_PATH = r'D:\Datasets\Flickr30k\flickr30k-images'

    def __init__(self, split:str, transforms: Union[v2.Compose, Callable]) -> None:
        super(Flickr30k, self).__init__()
        self.df = self.get_split(split)
        self.transforms = transforms

    def get_split(self, split:str):
        df = pd.read_csv(self.ANNOTATIONS)
        df = df.loc[df['split'] == split]
        df.reset_index(drop=True, inplace=True)
        return df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index:int):
        data = self.df.loc[index]
        captions = data['raw']
        img_path = os.path.join(Flickr30k.DATA_PATH, data['filename'])
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = self.transforms(img)

        return img, captions
    