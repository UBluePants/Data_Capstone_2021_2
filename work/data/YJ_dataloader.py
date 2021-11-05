import torch
import torch.utils.data
import os
import numpy as np
from PIL import Image


from utils.util import *

def loaderAndResize(path):

    return Image.open(path).resize((128, 128))

def loader(path):

    return Image.open(path)

class YJ_Dataloader(torch.utils.data.Dataset):

    def __init__(self, GT_root, Synthetic_root, 
                        non_occ_root, occ_root,
                        transform = None, loader = loader):

        self.GT_root = GT_root
        self.Synthetic_root = Synthetic_root
        self.non_occ_root = non_occ_root
        self.occ_root = occ_root

        GT_lst = os.listdir(GT_root)
        self.synth_lst = os.listdir(Synthetic_root)
        self.GT_lst = []

        for gt in GT_lst:
            num = 0
            for s in self.synth_lst:
                if gt[-8:-4] == s[12:16]:
                    num = num + 1
            for i in range(num): 
                self.GT_lst.append(gt)
        
        self.non_occ_lst = os.listdir(non_occ_root)
        self.occ_lst = os.listdir(occ_root)

        

        self.attr_lst = [0 for i in range(509)]

        self.loader = loader
        self.transform = transform


    def __getitem__(self, index):

        #################### GT
        GT_attr = int(self.GT_lst[index][-8:-4])
        GT_img = self.loader(os.path.join(self.GT_root, self.GT_lst[index]))
        GT_img = GT_img.resize((128,128))
        GT_attr_lst = self.attr_lst
        GT_attr_lst[GT_attr - 1] = 1
        #################### synth

        synth_attr = int(self.synth_lst[index][12:16])
        synth_img = self.loader(os.path.join(self.Synthetic_root, self.synth_lst[index]))
        synth_img = synth_img.resize((128,128))
        synth_attr_lst = self.attr_lst
        synth_attr_lst[synth_attr - 1] = 1

        #################### natural

        index2 = np.random.randint(0, len(self.non_occ_lst)-1)
        non_occ_img = self.loader(os.path.join(self.non_occ_root, self.non_occ_lst[index2]))
        non_occ_img = non_occ_img.resize((128,128))

        ################### occluded

        index3 = np.random.randint(0, len(self.occ_lst)-1)
        occ_img = self.loader(os.path.join(self.occ_root, self.occ_lst[index3]))
        occ_img = occ_img.resize((128,128))

        if self.transform is not None:

            GT_img = self.transform(GT_img)
            synth_img = self.transform(synth_img)
            non_occ_img = self.transform(non_occ_img)
            occ_img  = self.transform(occ_img)


        sample = {
                    'GT_img' : GT_img,
                    'GT_attr' : torch.from_numpy(np.array(GT_attr_lst)),

                    'synth_img' : synth_img,
                    'synth_attr' : torch.from_numpy(np.array(synth_attr_lst)),

                    'natural_img' : non_occ_img,
                    'occ_img' : occ_img
        }

        return sample


    def __len__(self):

        return len(self.GT_lst)
