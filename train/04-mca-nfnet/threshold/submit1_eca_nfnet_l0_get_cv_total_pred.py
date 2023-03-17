import pandas as pd

df_1=pd.read_csv("/ssd8/ming/covid_challenge/output/cnn_10_pred_1df.csv")
df_2=pd.read_csv("/ssd8/ming/covid_challenge/output/cnn_10_pred_2df.csv")
df_3=pd.read_csv("/ssd8/ming/covid_challenge/output/cnn_10_pred_3df.csv")
df_4=pd.read_csv("/ssd8/ming/covid_challenge/output/cnn_10_pred_4df.csv")
df_5=pd.read_csv("/ssd8/ming/covid_challenge/output/cnn_10_pred_5df.csv")  

df_total=df_1.merge(df_2[["path","pred"]], on='path')
df_total=df_total.merge(df_3[["path","pred"]], on='path')
df_total=df_total.merge(df_4[["path","pred"]], on='path')
df_total=df_total.merge(df_5[["path","pred"]], on='path')

df_total["total_pred"]=(df_total.iloc[:,1].values+df_total.iloc[:,2].values+df_total.iloc[:,3].values+df_total.iloc[:,4].values+df_total.iloc[:,5].values)/5

print(df_total.head())
print(df_total.tail())

df_total.to_csv("/ssd8/ming/covid_challenge/output/cnn_10_total_pred_df.csv",header=True,index=False)
