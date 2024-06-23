# datasets will be put here
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader
import random
import numpy as np
from torch.utils.data import random_split
import pandas as pd

random.seed(42)
np.random.seed(42)

class DataPipeline:

    def __init__(self, **configs):

        self.configs = configs

    
    def get_dataloader(self):
        dataset_name = self.configs['dataset_name'].lower()

        if dataset_name == 'flickr30k':
            return self._load_flickr30k()
        
    
    def _load_flickr30k(self):

        transforms_train = T.Compose([
            T.Resize((224,224)),
            T.ToTensor()]) # add any extra transforms for training
        full_trainset = datasets.CIFAR10(root=self.configs['datapath'], train=True, download=True, transform=transforms_train)

        # calculating the splits
        train_size = int((1-self.configs['val_split']) * len(full_trainset))

        trainset, valset = random_split(full_trainset, [train_size, 1-train_size])
        train_load = DataLoader(trainset, batch_size=self.configs['batch_size'], shuffle=True, num_workers=self.configs['num_workers'], pin_memory=self.configs['pin_memory'])
        val_load = DataLoader(valset, batch_size=self.configs['batch_size'], shuffle=False, num_workers=self.configs['num_workers'], pin_memory=self.configs['pin_memory'])

        return train_load, val_load
    


dataset_name = 'flickr30k'
datapath = r'D:\Datasets\Flickr30k'
val_split = 0.2
batch_size = 64
num_workers = 3
pin_memory = True

data_pipeline = DataPipeline(
    dataset_name=dataset_name,
    datapath=datapath, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    pin_memory=pin_memory, 
    val_split=val_split
    )
train_dl, val_dl, test_dl = data_pipeline.get_dataloader()
print(train_dl, val_dl, test_dl)