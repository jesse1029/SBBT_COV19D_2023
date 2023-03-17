import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import recall_score
import numpy as np
pd.set_option('mode.chained_assignment', None)
# This code will not complain!
pd.reset_option("mode.chained_assignment")
df_one = pd.read_csv("/ssd8/ming/covid_challenge/threshold/cnn_one_total_pred_df.csv")

threshold = [0.4+i*0.05 for i in range(20)]
print(threshold)
'''
f1_=[]
se_=[]
sp_=[]

best_f1 = 0
best_f1_threshold=0

best_se = 0
best_se_threshold=0

best_sp = 0
best_sp_threshold=0
for th in threshold:
    use_df = df_one
    use_df['pred_'] = 999
    use_df['pred_'][use_df.pred > th] = 1
    use_df['pred_'][use_df.pred <= th] = 0
    true_y=use_df['label']
    pred_y=use_df['pred_']
    f1 = f1_score(np.array(true_y), np.round(pred_y), average='macro')
    sensitivity = recall_score(np.array(true_y), np.round(pred_y))
    specificity = recall_score(np.logical_not(np.array(true_y)), np.logical_not(np.round(pred_y)))
    if f1 >= best_f1:
        best_f1 = f1
        best_f1_threshold=th
    if sensitivity >= best_se:
        best_se = sensitivity
        best_se_threshold=th
    if specificity >= best_sp:
        best_sp = specificity
        best_sp_threshold=th
    f1_.append(f1)
    se_.append(sensitivity)
    sp_.append(specificity)

#print(f1_)
#print(se_)
#print(sp_)

#print(max(f1_))
#print(max(se_))
##print(max(sp_))

print(best_f1)
print(best_f1_threshold)
print(threshold)

use_df.to_csv("/ssd8/ming/covid_challenge/threshold/cnn_one_total_pred_df_binary.csv", header=True, index=False)

############
print("="*10,"one10","*"*10)
print("="*10,"one10","*"*10)
print("="*10,"one10","*"*10)
print("="*10,"one10","*"*10)
'''
df_one = pd.read_csv("/ssd8/ming/covid_challenge/threshold/cnn_10_total_pred_df.csv")

threshold = [0.4+i*0.002 for i in range(100)]
print(threshold)

f1_=[]
se_=[]
sp_=[]

best_f1 = 0
best_f1_threshold=0

best_se = 0
best_se_threshold=0

best_sp = 0
best_sp_threshold=0
index=0
best_f1_index=0
for th in threshold:

    use_df = df_one
    use_df['pred_'] = 999
    use_df['pred_'][use_df.pred > th] = 1
    use_df['pred_'][use_df.pred <= th] = 0
    true_y=use_df['label']
    pred_y=use_df['pred_']
    f1 = f1_score(np.array(true_y), np.round(pred_y), average='macro')
    sensitivity = recall_score(np.array(true_y), np.round(pred_y))
    specificity = recall_score(np.logical_not(np.array(true_y)), np.logical_not(np.round(pred_y)))
    if f1 >= best_f1:
        best_f1 = f1
        best_f1_threshold=th
        best_f1_index=index
    if sensitivity >= best_se:
        best_se = sensitivity
        best_se_threshold=th
    if specificity >= best_sp:
        best_sp = specificity
        best_sp_threshold=th
    f1_.append(f1)
    se_.append(sensitivity)
    sp_.append(specificity)

    index+=1

#print(f1_)
#print(se_)
#print(sp_)

#print(max(f1_))
#print(max(se_))
##print(max(sp_))

print(best_f1)
print(best_f1_threshold)
print(best_f1_index)
print(se_[best_f1_index])
print(sp_[best_f1_index])
print(threshold)
df_one = pd.read_csv("/ssd8/ming/covid_challenge/threshold/cnn_10_total_pred_df.csv")
use_df = df_one
use_df['pred_'] = 999
use_df['pred_'][use_df.pred > best_f1_threshold] = 1
use_df['pred_'][use_df.pred <= best_f1_threshold] = 0
use_df.to_csv("/ssd8/ming/covid_challenge/threshold/cnn_10_total_pred_df_binary.csv", header=True, index=False)
print("="*10,"mean10","*"*10)
print("="*10,"mean10","*"*10)
print("="*10,"mean10","*"*10)
print("="*10,"mean10","*"*10)