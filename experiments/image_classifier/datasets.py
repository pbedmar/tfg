import glob

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from functions import gen_metadata
from torchvision.transforms import ToTensor
import os

class CompoundDataset(Dataset):
    def __init__(self, dir_pos, dir_neg, set_name, transform=None, target_transform=None):
        assert set_name in ["train", "test"]

        if not os.path.exists(dir_pos+"/imclass_"+set_name+".txt"):
            gen_metadata(dir_pos)

        if not os.path.exists(dir_neg+"/imclass_"+set_name+".txt"):
            gen_metadata(dir_neg)

        with open(dir_pos+"/imclass_"+set_name+".txt") as fd:
            self.filenames_pos = fd.read().splitlines()

        with open(dir_neg + "/imclass_" + set_name + ".txt") as fd:
            self.filenames_neg = fd.read().splitlines()

        self.filenames = self.filenames_pos + self.filenames_neg
        self.pos_length = len(self.filenames_pos)
        self.set_name = set_name
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = read_image(self.filenames[index])
        if index < len(self.filenames_pos):
            label = torch.tensor([1])
        else:
            label = torch.tensor([0])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


dataset = CompoundDataset("../../datasets/negev/articles_molecules/preprocess256",
                          "../../datasets/negev/not_molecules/preprocess256",
                          "train",
                          )
datapoint, label = dataset[4]
print(datapoint, label)
print(torch.zeros(2,5))