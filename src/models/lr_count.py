import os
from os.path import join

import pandas as pd
from sklearn import linear_model, metrics
from sklearn.feature_extraction.text import CountVectorizer

num_folds = 5
data_dir = "../inputs/"
train_filename = "train_folds.csv"
output_dir = "./predictions/"
out_filename = "pred2.csv"


def train(fold):
    df = pd.read_csv(join(data_dir, train_filename))

    train_df = df[df.fold != fold].reset_index(drop=True)
    val_df = df[df.fold == fold].reset_index(drop=True)

    tfv = CountVectorizer()
    tfv.fit(train_df.review.values)

    x_train = tfv.transform(train_df.review.values)
    x_valid = tfv.transform(val_df.review.values)

    y_train = train_df.sentiment.values
    y_valid = val_df.sentiment.values

    clf = linear_model.LogisticRegression()
    clf.fit(x_train, y_train)

    y_hat = clf.predict_proba(x_valid)[:, 1]
    auc_score = metrics.roc_auc_score(y_valid, y_hat)
    print(f"fold:{fold} auc: {auc_score}\n")
    val_df.loc[:, "LR_count"] = y_hat
    return val_df[["id", "sentiment", "fold", "LR_count"]]


if __name__ == "__main__":
    folds_df = []
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_folds):
        fold_df = train(i)
        folds_df.append(fold_df)
    folds_df = pd.concat(folds_df)
    print(folds_df.head())
    folds_df.to_csv(join(output_dir, out_filename), index=False)

