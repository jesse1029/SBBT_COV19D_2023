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
from sklearn.metrics import f1_score, roc_auc_score
# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
# from imblearn.metrics import sensitive_score
# from imblearn.metrics import specificity_score
from sklearn.metrics import recall_score

from utils.utils_2dcnn import *
from utils.model_2dcnn_eca_nfnet_l0 import criterion, eca_nfnet_l0


# from utils.model_2dcnn import Net, criterion
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


@torch.inference_mode()
def pred_one(model, dataloader, device):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    true_y = []
    pred_y = []
    for step, data in bar:
        ct_b, img_b, c, h, w = data['image'].size()
        data_img = data['image'].reshape(-1, c, h, w)
        data_label = data['label'].reshape(-1, 1)

        images = data_img.to('cuda', dtype=torch.float)
        labels = data_label.to('cuda', dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images)
        loss = criterion(outputs, labels)

        true_y.append(labels.cpu().numpy())
        pred_y.append(torch.sigmoid(outputs).cpu().numpy())

    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)

    gc.collect()

    true_y = np.array(true_y).reshape(-1, 1)
    true_y = np.array(true_y).reshape(-1, img_b)
    true_y = true_y.mean(axis=1)

    pred_y = np.array(pred_y).reshape(-1, 1)
    pred_y = np.array(pred_y).reshape(-1, img_b)

    pred_y = pred_y.mean(axis=1)

    return true_y, pred_y


if __name__ == '__main__':
    # Config
    set_seed()
    job = 51
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    CONFIG = {"seed": 2022,
              "epochs": 100,  # 24
              "img_size": 384,  # 512

              "train_batch_size": 16,  # 16
              "valid_batch_size": 16,
              "learning_rate": 0.0001,

              "weight_decay": 0.0005,

              "n_accumulate": 1,  # 2
              "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

              "train_batch": 16,
              }
    # img size = 256; batch=8; f1-score mean: 0.9142
    # Data Augmenation
    data_transforms = {
        "train": A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.ShiftScaleRotate(shift_limit=0.2,
                               scale_limit=0.2,
                               rotate_limit=30,
                               p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),  # 0.2
                contrast_limit=(-0.2, 0.2),  # 0.2
                p=0.5
            ),
            A.dropout.coarse_dropout.CoarseDropout(p=0.2),
            A.Normalize(),
            ToTensorV2()], p=1.),

        "valid": A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),

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
    # test_df = pd.read_csv('')
    fold_df = pd.read_csv('/ssd8/2023COVID19/Train_Valid_dataset/chih_train_valid_fold_1_5.csv')
    total_dic = {**train_dic, **valid_dlc}
    total_df = pd.concat([train_df, valid_df])
    # fold loader

    lst = [1, 2, 3, 4, 5]
    for i in range(len(lst)):
        print("fold {}的順序：{}".format(i + 1, lst[i:] + lst[:i]))
        train_lst = (lst[i:] + lst[:i])[0:4]
        train_fold = fold_df[fold_df.fold.isin(train_lst)]
        valid_lst = (lst[i:] + lst[:i])[-1]
        valid_fold = fold_df[fold_df.fold.isin([valid_lst])]
        print(valid_fold.values.tolist()[0])
        print("Train: {} || Valid: {}".format(train_fold.shape, valid_fold.shape))
        train_loader, valid_loader = prepare_loaders(CONFIG, train_df, train_dic, valid_df, valid_dlc, data_transforms)
        bin_save_path = "/ssd8/ming/covid_challenge/model"
        job_name = f"job_{job}_eca_nfnet_l0_size{CONFIG['img_size']}_challenge[DataParallel]-fold{i + 1}" + ".bin"
        print("=" * 10, "loading *model*", "=" * 10)
        model = eca_nfnet_l0(n_classes=2, pretrained=True)
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model = model.to(CONFIG['device'])
        scaler = amp.GradScaler()
        print("=" * 10, "*model* setting", "=" * 10)
        train_loader, valid_loader = prepare_loaders(CONFIG, train_fold, total_dic, valid_fold, total_dic,
                                                     data_transforms)

        pred_path = f'{bin_save_path}/f1/' + f"job_{job}_eca_nfnet_l0_size{CONFIG['img_size']}_challenge[DataParallel]-fold{i + 1}" + ".bin"
        checkpoint = torch.load(pred_path)
        
        #model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})

        # original saved file with DataParallel
        state_dict = torch.load(pred_path)  # 模型可以保存为pth文件，也可以为pt文件。
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = "module."+k[:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。 
        # load params
        model.load_state_dict(new_state_dict) # 从新加载这个模型。





        #model.load_state_dict(checkpoint, strict=True)
        model.to('cuda')
        #for param_tensor in model.state_dict():
           # print(param_tensor,"\t",model.state_dict()[param_tensor].size())
        total_pred = []
        for j in range(10):
            true_y, pred_y = pred_one(model, valid_loader, device=CONFIG['device'])
            total_pred.append(pred_y)
            #total_pred_round.append(np.round(pred_y))

            # confusion_matrix_.append(confusion_matrix(np.array(true_y), np.array(total_pred_round)))

            #specificity.append(recall_score(np.array(true_y), np.array(total_pred_round)))
            #sensitivity.append(recall_score(np.array(true_y), np.array(total_pred_round)))

        print("=" * 10, "{} fold outcome (f1-score-10)".format(i + 1), "=" * 10)
        for j in range(len(total_pred)):
            print(f1_score(np.array(true_y), np.round(total_pred[j]), average='macro'))

        print("=" * 10, "{} fold outcome (sensitivity-10)".format(i + 1), "=" * 10)
        for j in range(len(total_pred)):
            # print(sensitive_score(np.array(true_y), np.round(total_pred[j])))
            print(recall_score(np.array(true_y),  np.round(total_pred[j])))
        print("=" * 10, "{} fold outcome (specificity-10)".format(i + 1), "=" * 10)
        for j in range(len(total_pred)):
            # print(specificity_score(np.array(true_y), np.round(total_pred[j])))
            print(recall_score(np.logical_not(np.array(true_y)) , np.logical_not(np.round(total_pred[j]))
                               ))
        print("="*10,"{} fold outcome (macro-f1-score-mean-10)".format(i+1),"="*10)
        print(f1_score(np.array(true_y), np.round(np.mean(total_pred, axis=0)), average='macro'))