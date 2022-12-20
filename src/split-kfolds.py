from os.path import join

import pandas as pd
from sklearn import model_selection

SEED = 1245
data_dir = "../inputs/"
train_filename = "labeledTrainData.tsv"

if __name__ == "__main__":
    df = pd.read_csv(join(data_dir, train_filename), sep="\t")
    df.loc[:, "fold"] = -1

    y = df.sentiment.values
    skf = model_selection.StratifiedKFold(n_splits=5, random_state=SEED)

    for fold_id, (train, val) in enumerate(skf.split(X=df, y=y)):
        df.loc[val, "fold"] = fold_id

    df.to_csv(join(data_dir, "train_folds.csv"), index=False)

    print(df.head())
    print(df.fold.value_counts())
