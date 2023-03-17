import pandas as pd

valid_df = pd.read_csv('/ssd8/2023COVID19/Train_Valid_dataset/filter_slice_valid_df_challenge.csv')

df_1 = pd.read_csv("/ssd8/ming/covid_challenge/threshold/cnn_10_pred_1df.csv")
df_2 = pd.read_csv("/ssd8/ming/covid_challenge/threshold/cnn_10_pred_2df.csv")
df_3 = pd.read_csv("/ssd8/ming/covid_challenge/threshold/cnn_10_pred_3df.csv")
df_4 = pd.read_csv("/ssd8/ming/covid_challenge/threshold/cnn_10_pred_4df.csv")
df_5 = pd.read_csv("/ssd8/ming/covid_challenge/threshold/cnn_10_pred_5df.csv")

df_total = pd.concat([df_1,df_2,df_3,df_4,df_5])
df_total = df_total.merge(valid_df[["path", "label"]], on='path')

print(df_total.head())
print(df_total.tail())

df_total.to_csv("/ssd8/ming/covid_challenge/threshold/cnn_10_total_pred_df.csv", header=True, index=False)

df_1 = pd.read_csv("/ssd8/ming/covid_challenge/threshold/cnn_one_pred_1df.csv")
df_2 = pd.read_csv("/ssd8/ming/covid_challenge/threshold/cnn_one_pred_2df.csv")
df_3 = pd.read_csv("/ssd8/ming/covid_challenge/threshold/cnn_one_pred_3df.csv")
df_4 = pd.read_csv("/ssd8/ming/covid_challenge/threshold/cnn_one_pred_4df.csv")
df_5 = pd.read_csv("/ssd8/ming/covid_challenge/threshold/cnn_one_pred_5df.csv")

df_total = pd.concat([df_1,df_2,df_3,df_4,df_5])
df_total = df_total.merge(valid_df[["path", "label"]], on='path')

print(df_total.head())
print(df_total.tail())

df_total.to_csv("/ssd8/ming/covid_challenge/threshold/cnn_one_total_pred_df.csv", header=True, index=False)
