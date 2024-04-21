import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_labels(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            files.append(file[:-4])
    return files

def main():
    data_dir = "./data/"
    data = sorted(create_labels(data_dir))
    direction_num = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    labels = {x: direction_num[x[:-3]] for x in data}
    print(labels, "\n", len(labels))

    df = pd.DataFrame.from_dict(labels, orient='index')
    df.to_csv(data_dir + '/labels/labels.csv')

    df['filename'] = df.index

    train, val = train_test_split(df, test_size=0.2, random_state=1)

    train.to_csv('./data/labels/train.csv', index=False)
    val.to_csv('./data/labels/val.csv', index=False)


if __name__ == "__main__":
    main()