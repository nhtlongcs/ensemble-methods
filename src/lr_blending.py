import numpy as np
import glob
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression

num_folds = 5


def train(fold, df):

    train_df = df[df.fold != fold].reset_index(drop=True)
    val_df = df[df.fold == fold].reset_index(drop=True)

    x_train = train_df[["LR", "LR_count", "xgb"]].values
    x_valid = val_df[["LR", "LR_count", "xgb"]].values

    y_train = train_df.sentiment.values
    y_valid = val_df.sentiment.values

    opt = LinearRegression()
    opt.fit(x_train, y_train)

    y_hat = opt.predict(x_valid)
    auc_score = metrics.roc_auc_score(y_valid, y_hat)
    print(f"fold:{fold} auc: {auc_score}\n")
    return opt.coef_


if __name__ == "__main__":
    csv_ls = glob.glob("./predictions/pred*.csv")
    coefs = []
    pred_cols = ["LR", "LR_count", "xgb"]

    df = pd.read_csv(csv_ls[0])
    for csv in csv_ls[1:]:
        df = df.merge(pd.read_csv(csv), on="id", how="left")
    for fold in range(num_folds):
        coefs.append(train(fold=fold, df=df))
    print(coefs)
    coefs = np.mean(coefs, axis=0)
    lr_preds, lr_count_preds, xgb_preds = [df[col] for col in pred_cols]
    weighted_avg_preds = (
        coefs[0] * lr_preds + coefs[1] * lr_count_preds + coefs[2] * xgb_preds
    )
    auc_score = metrics.roc_auc_score(df.sentiment.values, weighted_avg_preds)
    print(f"model: weighted avg, auc_score: {auc_score}")
