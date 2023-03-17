import os, random, cv2, numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class Covid19Dataset(Dataset):
    def __init__(self, df, train_dict, train_batch=10, img_size = 512, transforms=None):
        self.df = df
        self.train_dict = train_dict
        self.img_size = img_size
        # self.file_names = df['filename'].values
        self.path = df['path'].values
        self.labels = df['label'].values
        self.transforms = transforms
        self.img_batch=train_batch
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        label = self.labels[index]
        img_path = self.path[index]
        img_path_l = os.listdir(img_path)
        img_path_l_ = [file[2:] if file.startswith("._") else file for file in img_path_l]
        
        img_list = [int(i.split('.')[0]) for i in img_path_l_]
        index_sort = sorted(range(len(img_list)), key=lambda k: img_list[k])
        
        ct_len = len(img_list)
        
#         if label==0:
#             start_idx = 0
#             end_idx = ct_len
#         else:
#             start_idx,end_idx=train_dic[img_path]
        
        start_idx,end_idx=self.train_dict[img_path]
        


        img_sample = torch.zeros((self.img_batch, 3, self.img_size, self.img_size))
        label_sample=torch.zeros((self.img_batch, 1))

        
        if (end_idx-start_idx) > self.img_batch:
            sample_idx = random.sample(range(start_idx, end_idx),self.img_batch)
        elif ct_len>20:
            sample_idx = [random.choice(range(start_idx, end_idx)) for _ in range(self.img_batch)]
            
        else:
            sample_idx = [random.choice(range(ct_len)) for _ in range(self.img_batch)]
        
        for count, idx in enumerate(sample_idx):

            img_path_ = os.path.join(img_path, img_path_l_[index_sort[idx]])
            
            img = cv2.imread(img_path_)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = self.transforms(image=img)['image']
            
            
            img_sample[count] = img[:]
            label_sample[count]= label
        return {
            'image': img_sample,
            'label': torch.tensor(label_sample, dtype=torch.long)
        }

class Covid19Dataset_valid(Dataset):
    def __init__(self, df, valid_dic, train_batch=10, img_size = 512, transforms=None):
        self.df = df
        self.valid_dic = valid_dic
        # self.file_names = df['filename'].values
        self.path = df['path'].values
        self.labels = df['label'].values
        self.transforms = transforms
        self.img_batch=train_batch
        self.img_size = img_size
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        label = self.labels[index]
        img_path = self.path[index]
        img_path_l = os.listdir(img_path)
        img_path_l_ = [file[2:] if file.startswith("._") else file for file in img_path_l]
        
        img_list = [int(i.split('.')[0]) for i in img_path_l_]
        index_sort = sorted(range(len(img_list)), key=lambda k: img_list[k])
        ct_len = len(img_list)
        # print(img_path)
        # print(img_list)
        # print(index_sort)
        
        start_idx,end_idx=self.valid_dic[img_path]
        # print(start_idx)
        # print(end_idx)


        img_sample = torch.zeros((self.img_batch, 3, self.img_size, self.img_size))
        label_sample=torch.zeros((self.img_batch, 1))

        
        if (end_idx-start_idx) > self.img_batch:
            sample_idx = random.sample(range(start_idx, end_idx),self.img_batch)
        elif ct_len>20:
            sample_idx = [random.choice(range(start_idx, end_idx)) for _ in range(self.img_batch)]
        else:
            sample_idx = [random.choice(range(ct_len)) for _ in range(self.img_batch)]
        # print(sample_idx)
        for count, idx in enumerate(sample_idx):

            img_path_ = os.path.join(img_path, img_path_l_[index_sort[idx]])
            
            img = cv2.imread(img_path_)
          
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = self.transforms(image=img)['image']
          
            
            img_sample[count] = img[:]
            label_sample[count]= label
        return {
            'image': img_sample,
            'label': torch.tensor(label_sample, dtype=torch.long)
        }

        
def prepare_loaders(CONFIG, train_df, train_dic, valid_df, valid_dlc, data_transforms, world_seed = None, rank=None):
    train_dataset = Covid19Dataset(train_df, train_dic, CONFIG['train_batch'], 
                                    img_size = CONFIG["img_size"] , transforms=data_transforms["train"])
    valid_dataset = Covid19Dataset_valid(valid_df, valid_dlc, CONFIG['train_batch'], 
                                img_size = CONFIG["img_size"] , transforms=data_transforms["valid"])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["train_batch_size"],
                            num_workers=15, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG["valid_batch_size"],
                            num_workers=15, shuffle=False, pin_memory=True)

    return train_loader, valid_loader

def prepare_loaders_eval(CONFIG, train_df, train_dic, valid_df, valid_dlc, data_transforms):
    train_dataset = Covid19Dataset(train_df, train_dic, CONFIG['train_batch'], 
                                   img_size = CONFIG["img_size"], transforms=data_transforms["valid"])
    valid_dataset = Covid19Dataset_valid(valid_df, valid_dlc, CONFIG['train_batch'],
                                        img_size = CONFIG["img_size"], transforms=data_transforms["valid"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["train_batch_size"], 
                              num_workers=5, shuffle=False, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG["valid_batch_size"], 
                              num_workers=5, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader