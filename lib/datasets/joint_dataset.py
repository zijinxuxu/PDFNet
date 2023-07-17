from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from posixpath import splitdrive

import numpy as np
import os
import pickle
import torch.utils.data as data
from glob import glob

dataset_index = {
  'Joint': 0,
  'FreiHAND': 1,
  'HO3D': 2,
  'H2O': 3,
  'H2O3D': 4,
  'InterHand': 5,
  'RHD': 6,
  'Others': 7,
}

class BaseDataset(data.Dataset):
    num_classes = 2 # hand
    mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(BaseDataset, self).__init__()
        self.split = split
        self.flip_idx = [[0, 1], [3, 4]]
        self.opt = opt
        self.update()

    def update(self):
        self.opt.size_train = self.default_resolution
        self.dataset_index = dataset_index

    def __len__(self):
        pass

class JointDataset(BaseDataset):

    def __init__(self, opt, split):
        super(JointDataset, self).__init__(opt, split)
        self.split = split
        self.min_size, self.max_size = 120, 200
        if split == 'train' or split == 'test' or split == 'val':
            self.data_dict = {}
            if opt.dataset == 'InterHandNew': 
                if split == 'train_3d':
                    split = 'train'
                self.data_path = os.path.join(self.opt.cache_path, 'InterHandNew')
                self.num_samples = len(glob(os.path.join(self.opt.cache_path, 'InterHandNew', split, 'anno', '*.pkl')))
            else: 
                dataname = opt.dataset
                cache_path = os.path.abspath(os.path.join(self.opt.cache_path, dataname + '_{}.pkl'.format(split)))          
                with open(cache_path, 'rb') as fid:
                    data = pickle.load(fid, encoding='latin1')
                for item in data:
                    item['dataset'] = self.dataset_index[dataname]
                    item['imgpath'] = os.path.join(dataname, item['imgpath'])
                    item['depthpath'] = os.path.join(dataname, item['depthpath'])
                self.data_dict[dataname] = data
                print('loaded {} dataset {}: {}'.format(split, dataname, len(data)))
                if split =='train' and opt.dataset == 'H2O': # add val dataset here.
                    cache_path = os.path.abspath(os.path.join(self.opt.cache_path, dataname + '_val.pkl'))          
                    with open(cache_path, 'rb') as fid:
                        data = pickle.load(fid, encoding='latin1')
                    for item in data:
                        item['dataset'] = self.dataset_index[dataname]
                        item['imgpath'] = os.path.join(dataname, item['imgpath'])
                        item['depthpath'] = os.path.join(dataname, item['depthpath'])
                    self.data_dict[dataname] += data
                    print('loaded val dataset {}: {}'.format(dataname, len(data)))
                self.prepare_data()
        else:
            print('wow, nope!')


    def __len__(self):
        return self.num_samples

    def prepare_data(self):
        self.data = []
        self.test_data = []
        for key in self.data_dict:
            if key == 'FreiHAND':
                if self.split == 'val':
                    # self.data += data[65120:75120:20]
                    self.data += self.data_dict[key][:3000]
                    self.data += self.data_dict[key][-3000:]
                elif self.split == 'test':
                    self.data += self.data_dict[key]
                else:
                    self.data += self.data_dict[key][:] 
            elif key == 'HO3D' or key == 'HO3Dv3':
                if self.split == 'val':
                    self.data += self.data_dict[key][:3000]
                    self.data += self.data_dict[key][-3000:]
                elif self.split == 'test':
                    self.data += self.data_dict[key]                    
                else:
                    self.data += self.data_dict[key][3000:-3000]#32560]                                    
            elif key == 'OneHand10K':
                if self.split == 'test':
                    self.data += self.data_dict[key][:1000]
                    self.data += self.data_dict[key][-1000:]
                elif self.split == 'eval':
                    self.data += self.data_dict[key] 
                else:
                    self.data += self.data_dict[key][1000:-1000]#32560]  
            elif key == 'H2O':
                if self.split == 'test':
                    # self.data += data[65120:75120:20]
                    self.data += self.data_dict[key][:]
                    # self.data += self.data_dict[key][-3000:]
                elif self.split == 'eval':
                    self.data += self.data_dict[key][:]
                else:
                    self.data += self.data_dict[key][:] 
            else:
                self.data += self.data_dict[key][:]
        self.batch_size = self.opt.batch_size
        self.num_samples = len(self.data)
        print('loaded joint datasets: {}'.format(self.num_samples))

        # if self.opt.input_res == 512 or self.opt.input_res == 384:
        #     self.max_objs = 2
        # else:
        self.max_objs = 2                  
        self.random_idx = np.zeros([self.num_samples, self.max_objs], dtype=np.int)
        self.random_idx[:, 0] = np.arange(self.num_samples)
        for i in range(1, self.max_objs):
            self.random_idx[:, i] = np.random.permutation(self.num_samples)
