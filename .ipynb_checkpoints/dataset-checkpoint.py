
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset



'''
Synthetic PBD data generated with a predefined control point trajetory
'''
class pandajoints_Dataset(Dataset):
    def __init__(self, root='panda_data/se3_quaternion', train=True):

        self.root = root
        self.datapath = []

        for root, dirs, files in os.walk(self.root):
            for file in files:
                self.datapath.append(os.path.join(root, file))
        self.datapath.sort()
           
        # use 4/5 of the data for training
        total_len = len(self.datapath)
        self.train = train
        if self.train:
            self.datapath = self.datapath[:int(total_len)]
        else:
            self.datapath = self.datapath[int(total_len * 4 / 5):]
        


    def __getitem__(self, index):
        # dir_idx = index // len(self.datapath)
        data1 = np.load(self.datapath[index])
        # data2 = np.load(self.datapath[dir_idx][file_idx2])
        
        pc1, gt, label, joints, ee_pos,quaternion = data1['pointcloud'], data1['GT'],data1['label'], data1['Joints'], data1['EE_pos'], data1['quaternion']
        # pc2, t2 = data2['vertices'], data2['gt']
        
        return pc1, gt, label, joints, ee_pos, quaternion
 

    def __len__(self):
        
            
            return int(len(self.datapath))
        # else:
        #     return int(len(self.datapath)) 