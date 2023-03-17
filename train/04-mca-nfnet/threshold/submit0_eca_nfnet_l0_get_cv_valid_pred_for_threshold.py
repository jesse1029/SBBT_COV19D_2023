import os, gc, cv2, math, copy, time, random
import pickle
# For data manipulation
import numpy as np, pandas as pd

# Pytorch Imports
import torch, torch.nn as nn, torch.optim as optim
from torch.optim import lr_scheduler

from torch.cuda import amp

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score,roc_auc_score
# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
#from imblearn.metrics import sensitive_score
#from imblearn.metrics import specificity_score
from sklearn.metrics import recall_score

from utils.utils_2dcnn import *
from utils.model_2dcnn_eca_nfnet_l0 import criterion, eca_nfnet_l0
#from utils.model_2dcnn import Net, criterion
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

class Covid19Dataset_valid(Dataset):
    def __init__(self, df, train_batch=16, transforms=None):
        self.df = df
        self.path = df['path'].values

        self.transforms = transforms
        self.img_batch = train_batch

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img_path = self.path[index]
        img_path_l = os.listdir(img_path)
        img_path_l_ = [file[2:] if file.startswith("._") else file for file in img_path_l]

        img_list = [int(i.split('.')[0]) for i in img_path_l_]
        index_sort = sorted(range(len(img_list)), key=lambda k: img_list[k])
        ct_len = len(img_list)

        start_idx, end_idx = total_dic[img_path]

        img_sample = torch.zeros((self.img_batch, 3, 384, 384))

        if (end_idx - start_idx) >= self.img_batch:
            sample_idx = random.sample(range(start_idx, end_idx), self.img_batch)
        elif ct_len > 20:
            sample_idx = [random.choice(range(start_idx, end_idx)) for _ in range(self.img_batch)]
        else:
            sample_idx = [random.choice(range(ct_len)) for _ in range(self.img_batch)]

        for count, idx in enumerate(sample_idx):
            img_path_ = os.path.join(img_path, img_path_l_[index_sort[idx]])

            img = cv2.imread(img_path_)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.transforms(image=img)['image']

            img_sample[count] = img[:]

        return {
            'image': img_sample,
            'id': img_path
        }

def prepare_loaders():
    valid_dataset = Covid19Dataset_valid(df, CONFIG['train_batch'], transforms=data_transforms["valid"])

    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG["valid_batch_size"],
                              num_workers=8, shuffle=False, pin_memory=True)

    return valid_loader


data_transforms = {

    "valid": A.Compose([
        A.Resize(384, 384),

        A.Normalize(),
        ToTensorV2()], p=1.)
}


@torch.inference_mode()
def inference(model, dataloader, device):
    model.eval()

    dataset_size = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    IDS = []
    pred_y = []
    for step, data in bar:
        ids = data["id"]
        ct_b, img_b, c, h, w = data['image'].size()
        data_img = data['image'].reshape(-1, c, h, w)

        images = data_img.to(device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images)

        pred_y.append(torch.sigmoid(outputs).cpu().numpy())
        IDS.append(ids)

    pred_y = np.concatenate(pred_y)
    IDS = np.concatenate(IDS)

    gc.collect()

    pred_y = np.array(pred_y).reshape(-1, 1)
    pred_y = np.array(pred_y).reshape(-1, img_b)

    pred_y = pred_y.mean(axis=1)

    return pred_y, IDS


if __name__ == '__main__':
    # Config
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,7"

    set_seed()
    job=51
    CONFIG = {"seed": 2022,
            "img_size": 384, #512

            "valid_batch_size": 16,
            "learning_rate": 0.0001,

            "weight_decay": 0.0005, 
        
            "n_accumulate": 1, #2
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            
            "train_batch":16,
            }
    # img size = 256; batch=8; f1-score mean: 0.9142
    # Data Augmenation
    data_transforms = {

        "valid": A.Compose([
            A.Resize(384, 384),

            A.Normalize(),
            ToTensorV2()], p=1.)
    }
    
    # Get data dict
    with open('/ssd8/2023COVID19/Train_Valid_dataset/filter_slice_train_dic1_05_challenge.pickle', 'rb') as f:
        train_dic = pickle.load(f)

    with open('/ssd8/2023COVID19/Train_Valid_dataset/filter_slice_valid_dic1_05_challenge.pickle', 'rb') as f:
        valid_dlc = pickle.load(f)

    train_df = pd.read_csv('/ssd8/2023COVID19/Train_Valid_dataset/filter_slice_train_df_challenge.csv')
    valid_df = pd.read_csv('/ssd8/2023COVID19/Train_Valid_dataset/filter_slice_valid_df_challenge.csv')   

    fold_df = pd.read_csv('/ssd8/2023COVID19/Train_Valid_dataset/chih_train_valid_fold_1_5.csv')
    total_dic = {**train_dic, **valid_dlc}
    total_df = pd.concat([train_df, valid_df])  
    # fold loader
    
    lst = [1,2,3,4,5]
    for j in range(5):
        print("fold {}的順序：{}".format(j + 1, lst[j:] + lst[:j]))
        train_lst = (lst[j:] + lst[:j])[0:4]
        train_fold = fold_df[fold_df.fold.isin(train_lst)]
        valid_lst = (lst[j:] + lst[:j])[-1]
        valid_fold = fold_df[fold_df.fold.isin([valid_lst])]
        print(valid_fold.values.tolist()[0])
        print("Train: {} || Valid: {}".format(train_fold.shape, valid_fold.shape))

        bin_save_path = "/ssd8/ming/covid_challenge/model"
        job_name = f"job_{job}_eca_nfnet_l0_size{CONFIG['img_size']}_challenge[DataParallel]-fold{j + 1}" + ".bin"
        weights_path = f'{bin_save_path}/f1/' + f"job_{job}_eca_nfnet_l0_size{CONFIG['img_size']}_challenge[DataParallel]-fold{j + 1}" + ".bin"

        print("=" * 10, "loading *model*", "=" * 10)
        model = eca_nfnet_l0(n_classes=2, pretrained=True)
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3,4,5])
        model = model.to(CONFIG['device'])
        scaler = amp.GradScaler()

        state_dict = torch.load(weights_path)  # 模型可以保存为pth文件，也可以为pt文件。
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = "module." + k[:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
            # load params

        model.load_state_dict(new_state_dict)  # 从新加载这个模型。
            # model.load_state_dict(state_dict) # 从新加载这个模型。

            # model = nn.DataParallel(model)
        model = model.cuda()
        df = pd.DataFrame(valid_fold, columns=["path"])
        test_loader = prepare_loaders()
        total_pred = []

        for i in range(1):
            pred_y, name = inference(model, test_loader, device=CONFIG['device'])
            total_pred.append(pred_y)

        final_pred = np.mean(total_pred, axis=0)
        dict_all = dict(zip(name, final_pred))
        cnn_one_pred_df = pd.DataFrame(list(dict_all.items()),
                                           columns=['path', 'pred'])
        cnn_one_pred_df.to_csv(f"/ssd8/ming/covid_challenge/threshold/cnn_one_pred_{j + 1}df.csv", index=False)

        times_list = [10]
        for times in times_list:
            total_pred = []
            for i in range(times):
                pred_y, name = inference(model, test_loader, device=CONFIG['device'])
                total_pred.append(pred_y)
            final_pred = np.mean(total_pred, axis=0)
            dict_all = dict(zip(name, final_pred))

            cnn_times_pred_df = pd.DataFrame(list(dict_all.items()),
                                                 columns=['path', 'pred'])
            cnn_times_pred_df.to_csv(f"/ssd8/ming/covid_challenge/threshold/cnn_{times}_pred_{j + 1}df.csv",
                                         index=False)
            print("save")
