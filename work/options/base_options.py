import argparse
import os
from utils import util
import torch
import easydict 

class BaseOptions():
    def __init__(self):
        self._initialized = False

    def initialize(self):
        self.args = easydict.EasyDict({
            'fake_real_rate' : 10,
            'GT_train_root' : "/content/drive/Othercomputers/내 노트북/data/Synthetic/train/GT_train",
            'Synthetic_train_root' : "/content/drive/Othercomputers/내 노트북/data/Synthetic/train/occ_train",
            'non_occ_train_root' : "/content/drive/Othercomputers/내 노트북/data/Natural/train/non_occ",
            'occ_train_root' : "/content/drive/Othercomputers/내 노트북/data/Natural/train/occ", 
            'GT_test_root' : "/content/drive/Othercomputers/내 노트북/data/Synthetic/test/GT_test",
            'Synthetic_test_root' : "/content/drive/Othercomputers/내 노트북/data/Synthetic/test/occ_test",
            'non_occ_test_root' : "/content/drive/Othercomputers/내 노트북/data/Natural/test/non_occ",
            'occ_test_root' : "/content/drive/Othercomputers/내 노트북/data/Natural/test/occ",

            'load_epoch' : -1,
            'batch_size' : 16,
            'image_size' : 128,
            'gpu_ids' : '0',
            'name' : 'experiment_38_alternate_training_-_10pow(2)_landmark_sig_10_wo_hash',
            'model' : 'FOAFCGAN_alternate_training',
            'n_threads_test' : 1,
            'checkpoints_dir' : '/content/drive/Othercomputers/내 노트북/data/check_points',

            'n_threads_train' : 1,
            'num_iters_validate' : 1,
            'print_freq_s' : 300,
            'display_freq_s' : 600,
            'save_latest_freq_s' : 3600,
            'nepochs_no_decay' : 90,
            'nepochs_decay' : 10,
            'train_G_every_n_iterations' : 5,

            'lr_G': 0.0001,
            'G_adam_b1' : 0.5,
            'G_adam_b2' : 0.999,
            'lr_D' : 0.0001,
            'D_adam_b1' : 0.5,
            'D_adam_b2' : 0.999,

            'attr_nc' : 509,

            'lambda_D_prob' : 1,
            'lambda_D_gp' : 10,
        
            'lambda_D_attr' : 2000, 

            'lambda_mask' : 1, 
            'lambda_mask_smooth': 1e-5,
        
            'lambda_g_style' : 120, 
            'lambda_g_perceptual' : 0.05,
            'lambda_g_syhth_smooth': 1e-5,
            'lambda_g_hole': 6, 
            'lambda_g_valid':1,
            'lambda_g_hash':1
        })
        self._initialized = True
        self.is_train = True

    def parse(self):

        if not self._initialized:
            self.initialize()

        self._opt = self.args

        self._opt.is_train = self.is_train

        self._set_and_check_load_epoch()

        self._get_set_gpus()

        args = vars(self._opt)

        self._print(args)

        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):

        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0

    def _get_set_gpus(self):

        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)

    def _print(self, args):

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        print(expr_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
