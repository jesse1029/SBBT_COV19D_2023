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

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ct_b, img_b, c, h, w = data['image'].size()
        data_img = data['image'].reshape(-1, c, h, w)
        data_label = data['label'].reshape(-1,1)
        images = data_img.to(device, dtype=torch.float)
        labels = data_label.to(device, dtype=torch.float)


        batch_size = images.size(0)

        with amp.autocast(enabled = True):
            outputs = model(images)

            loss = criterion(outputs, labels)

            loss = loss / CONFIG['n_accumulate']

        scaler.scale(loss).backward()

        if (step + 1) % CONFIG['n_accumulate'] == 0:

            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()



            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()

    return epoch_loss

@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    true_y=[]
    pred_y=[]
    for step, data in bar:
        ct_b, img_b, c, h, w = data['image'].size()
        data_img = data['image'].reshape(-1, c, h, w)
        data_label = data['label'].reshape(-1,1)

        images = data_img.to(device, dtype=torch.float)
        labels = data_label.to(device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images)
        loss = criterion(outputs, labels)


        true_y.append(labels.cpu().numpy())
        pred_y.append(torch.sigmoid(outputs).cpu().numpy())

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

    true_y=np.concatenate(true_y)
    pred_y=np.concatenate(pred_y)

    gc.collect()

    true_y=np.array(true_y).reshape(-1,1)
    true_y=np.array(true_y).reshape(-1,img_b)
    true_y=true_y.mean(axis=1)

    pred_y=np.array(pred_y).reshape(-1,1)
    pred_y = torch.nan_to_num(torch.from_numpy(pred_y)).numpy()
    pred_y=np.array(pred_y).reshape(-1,img_b)
#     pred_y2=pred_y.max(axis=1)
    pred_y=pred_y.mean(axis=1)


    assert (true_y.shape==pred_y.shape)
    acc_f1=f1_score(np.array(true_y),np.round(pred_y),average='macro')
    acc_f1_48=f1_score(np.array(true_y),np.where(pred_y>0.48,1,0),average='macro')
    acc_f1_51=f1_score(np.array(true_y),np.where(pred_y>0.51,1,0),average='macro')
    acc_f1_52=f1_score(np.array(true_y),np.where(pred_y>0.52,1,0),average='macro')
    acc_f1_54=f1_score(np.array(true_y),np.where(pred_y>0.54,1,0),average='macro')
    auc_roc=roc_auc_score(np.array(true_y),np.array(pred_y))
    print("acc_f1(mean) : ",round(acc_f1,4),"  auc_roc(mean) : ",round(auc_roc,4))


    return epoch_loss,acc_f1,auc_roc

def run_training(model, optimizer, scheduler, device, num_epochs):


    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    best_epoch_auc = 0
    best_epoch_f1 = 0
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler,
                                           dataloader=train_loader,
                                           device=CONFIG['device'], epoch=epoch)

        val_epoch_loss,acc_f1,auc_roc= valid_one_epoch(model, valid_loader, device=CONFIG['device'],
                                         epoch=epoch)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)


        if val_epoch_loss <= best_epoch_loss:
            print(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss

            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f'{bin_save_path}/loss/'+job_name
            os.makedirs(f'{bin_save_path}/loss/', exist_ok=True)
            torch.save(model.module.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved")

        if auc_roc >= best_epoch_auc:
            print(f"Validation Auc Improved ({best_epoch_auc} ---> {auc_roc})")
            best_epoch_auc = auc_roc

            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f'{bin_save_path}/auc_roc/'+job_name
            os.makedirs(f'{bin_save_path}/auc_roc/', exist_ok=True)
            torch.save(model.module.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved")

        if acc_f1 >= best_epoch_f1:
            print(f"Validation f1 Improved ({best_epoch_f1} ---> {acc_f1})")
            best_epoch_f1 = acc_f1

            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f'{bin_save_path}/f1/'+job_name
            os.makedirs(f'{bin_save_path}/f1/', exist_ok=True)
            torch.save(model.module.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved")

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))
    return model, history

@torch.inference_mode()
def pred_one(model, dataloader, device):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    true_y=[]
    pred_y=[]
    for step, data in bar:
        ct_b, img_b, c, h, w = data['image'].size()
        data_img = data['image'].reshape(-1, c, h, w)
        data_label = data['label'].reshape(-1,1)
        
        images = data_img.to('cuda', dtype=torch.float)
        labels = data_label.to('cuda', dtype=torch.float)
        
        batch_size = images.size(0)

        outputs = model(images)
        loss = criterion(outputs, labels)

        
        true_y.append(labels.cpu().numpy())
        pred_y.append(torch.sigmoid(outputs).cpu().numpy())
        

    

    true_y=np.concatenate(true_y)
    pred_y=np.concatenate(pred_y)
    
    
    
   
    gc.collect()
    
    true_y=np.array(true_y).reshape(-1,1)
    true_y=np.array(true_y).reshape(-1,img_b)
    true_y=true_y.mean(axis=1)
  
  
    pred_y=np.array(pred_y).reshape(-1,1)
    pred_y=np.array(pred_y).reshape(-1,img_b)

    pred_y=pred_y.mean(axis=1)
    
    return true_y,pred_y
    
if __name__ == '__main__':
    # Config
    set_seed()
    job=52
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,5,6"
    CONFIG = {"seed": 2022,
            "epochs": 100,  #24
            "img_size": 384, #512

            "train_batch_size": 16, #16
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
                brightness_limit=(-0.2,0.2), #0.2
                contrast_limit=(-0.2, 0.2),  #0.2
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
    


    #with open('ssd8/2023COVID19/Train_Valid_dataset/test_dic1_05.pickle', 'rb') as f:
    #    test_dlc = pickle.load(f)

    train_df = pd.read_csv('/ssd8/2023COVID19/Train_Valid_dataset/filter_slice_train_df_challenge.csv')
    valid_df = pd.read_csv('/ssd8/2023COVID19/Train_Valid_dataset/filter_slice_valid_df_challenge.csv')   
    #test_df = pd.read_csv('')
    fold_df = pd.read_csv('/ssd8/2023COVID19/Train_Valid_dataset/chih_train_valid_fold_1_5.csv')
    total_dic = {**train_dic, **valid_dlc}
    total_df = pd.concat([train_df, valid_df])  
    # fold loader
    
    lst = [1,2,3,4,5]
    for i in range(len(lst)):
        print("fold {}的順序：{}".format(i+1, lst[i:]+lst[:i]))
        train_lst = (lst[i:]+lst[:i])[0:4] 
        train_fold = fold_df[fold_df.fold.isin(train_lst)]
        valid_lst = (lst[i:]+lst[:i])[-1]
        valid_fold = fold_df[fold_df.fold.isin([valid_lst])]
        print(valid_fold.values.tolist()[0])
        print("Train: {} || Valid: {}".format(train_fold.shape, valid_fold.shape))
        
        train_loader, valid_loader = prepare_loaders(CONFIG, train_df, train_dic, valid_df, valid_dlc, data_transforms)
        bin_save_path = "/ssd8/ming/covid_challenge/model"
        job_name = f"job_{job}_eca_nfnet_l0_size{CONFIG['img_size']}_challenge[DataParallel]-fold{i+1}"+".bin"
        
        print("="*10, "loading *model*", "="*10)
        model=eca_nfnet_l0(n_classes=2,pretrained=True)
        
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
        model = model.to(CONFIG['device'])
        scaler = amp.GradScaler()
        train_loader, valid_loader = prepare_loaders(CONFIG, train_fold, total_dic, valid_fold, total_dic, data_transforms)
        
        print("="*10, "*model* setting", "="*10)
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                        weight_decay=CONFIG['weight_decay'])
        
        print("="*10, "Start Train", "="*10)
        

        model, history= run_training(model, optimizer,None,
                                device=CONFIG['device'],
                                num_epochs=CONFIG['epochs'])
    
    
        '''
        
        pred_path = f'{bin_save_path}/f1/' + f"job_{job}_eca_nfnet_l0_size{CONFIG['img_size']}_challenge[DataParallel]-fold{i+1}" + ".bin"
    
        model.load_state_dict(torch.load(pred_path))
        model.to('cuda')

        total_pred = []
        # total_pred_round = []
        # confusion_matrix_= []
        # specificity = []
        # sensitivity = []
        
        for j in range(10):
            true_y, pred_y = pred_one(model, valid_loader, device=CONFIG['device'])
            total_pred.append(pred_y)
            # total_pred_round.append(np.round(pred_y))

            # confusion_matrix_.append(confusion_matrix(np.array(true_y), np.array(total_pred_round)))

            # specificity.append(recall_score(np.array(true_y), np.array(total_pred_round)))
            # sensitivity.append(recall_score(np.array(true_y), np.array(total_pred_round)))

        print("=" * 10, "{} fold outcome (f1-score-10)".format(i + 1), "=" * 10)
        for j in range(len(total_pred)):
            print(f1_score(np.array(true_y), np.round(total_pred[j]), average='macro'))

        print("=" * 10, "{} fold outcome (sensitivity-10)".format(i + 1), "=" * 10)
        for j in range(len(total_pred)):
            # print(sensitive_score(np.array(true_y), np.round(total_pred[j])))
            print(recall_score(np.array(true_y), np.round(total_pred[j])))
        print("=" * 10, "{} fold outcome (specificity-10)".format(i + 1), "=" * 10)
        for j in range(len(total_pred)):
            # print(specificity_score(np.array(true_y), np.round(total_pred[j])))
            print(recall_score(np.logical_not(np.array(true_y)), np.logical_not(np.round(total_pred[j]))))
        '''