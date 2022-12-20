import numpy as np
import glob
import pandas as pd
from sklearn import metrics

if __name__ == "__main__":
    csv_ls = glob.glob("./predictions/pred*.csv")
    df = pd.read_csv(csv_ls[0])
    for csv in csv_ls[1:]:
        df = df.merge(pd.read_csv(csv), on="id", how="left")
    print(df.head())
    pred_cols = ["LR", "LR_count", "xgb"]
    for col in pred_cols:
        auc_score = metrics.roc_auc_score(df.sentiment.values, df[col].values)
        print(f"model: {col}, auc_score: {auc_score}")

    avg_preds = np.mean(df[pred_cols].values, axis=1)
    auc_score = metrics.roc_auc_score(df.sentiment.values, avg_preds)
    print(f"model: avg, auc_score: {auc_score}")

    lr_preds, lr_count_preds, xgb_preds = [df[col] for col in pred_cols]
    weighted_avg_preds = (5 * lr_preds + lr_count_preds + xgb_preds) / 7
    auc_score = metrics.roc_auc_score(df.sentiment.values, weighted_avg_preds)
    print(f"model: weighted avg, auc_score: {auc_score}")
