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

class YJ_Dataloader_train(torch.utils.data.Dataset):

    def __init__(self, GT_root, Synthetic_root, 
                        non_occ_root, occ_root,
                        transform = None, loader = loader):

        self.GT_root = GT_root
        self.Synthetic_root = Synthetic_root
        self.non_occ_root = non_occ_root
        self.occ_root = occ_root

        self.GT_lst = os.listdir(GT_root)
        self.synth_lst = os.listdir(Synthetic_root)
        self.non_occ_lst = os.listdir(non_occ_root)
        self.occ_lst = os.listdir(occ_root)

        self.attr_lst = [0 for i in range(509)]

        self.loader = loader
        self.transform = transform


    def __getitem__(self, index):

        #################### synth
        synth_attr = int(self.synth_lst[index][12:16])
        synth_img = self.loader(os.path.join(self.Synthetic_root, self.synth_lst[index]))
        synth_img = synth_img.resize((128,128))
        synth_attr_lst = self.attr_lst
        synth_attr_lst[synth_attr - 1] = 1 # 0.9 for smoothing

         #################### GT
        GT_name = "image_train_" + self.synth_lst[index][12:16] + ".jpg"
        GT_img = self.loader(os.path.join(self.GT_root, GT_name))
        GT_img = GT_img.resize((128,128))
        GT_attr_lst = self.attr_lst
        GT_attr_lst[synth_attr - 1] = 1 # 0.9 for smoothing

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

        return len(self.synth_lst)

class YJ_Dataloader_test(torch.utils.data.Dataset):

    def __init__(self, GT_root, Synthetic_root, 
                        non_occ_root, occ_root,
                        transform = None, loader = loader):

        self.GT_root = GT_root
        self.Synthetic_root = Synthetic_root
        self.non_occ_root = non_occ_root
        self.occ_root = occ_root

        self.GT_lst = os.listdir(GT_root)
        self.synth_lst = os.listdir(Synthetic_root)
        self.non_occ_lst = os.listdir(non_occ_root)
        self.occ_lst = os.listdir(occ_root)

        self.attr_lst = [0 for i in range(509)]

        self.loader = loader
        self.transform = transform


    def __getitem__(self, index):


        #################### synth

        synth_attr = int(self.synth_lst[index][11:15])
        synth_img = self.loader(os.path.join(self.Synthetic_root, self.synth_lst[index]))
        synth_img = synth_img.resize((128,128))
        synth_attr_lst = self.attr_lst
        synth_attr_lst[synth_attr - 1] = 1 # 0.9 for smoothing

         #################### GT
        GT_name = "image_test_" + self.synth_lst[index][11:15] + ".jpg"
        GT_img = self.loader(os.path.join(self.GT_root, GT_name))
        GT_img = GT_img.resize((128,128))
        GT_attr_lst = self.attr_lst
        GT_attr_lst[synth_attr - 1]= 1 # 0.9 for smoothing


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

        return len(self.synth_lst)