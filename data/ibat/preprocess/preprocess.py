import os
import pandas as pd
import argparse
from argparse import ArgumentParser
from datetime import timedelta
from datetime import datetime
from sklearn.model_selection import train_test_split

ARG_PARSER = ArgumentParser()

ARG_PARSER.add_argument("--test_size", default=0.1, type=float)
ARG_PARSER.add_argument("--val_size", default=0.05, type=float)
ARG_PARSER.add_argument("--seq_len", default=20, type=float)
ARG_PARSER.add_argument("--target_field", default="zigduino-3:temperature", type=str)
ARG_PARSER.add_argument("--path_folder", default='/scratch/stoll/BiGAN/data/ibat', type=str)
ARG_PARSER.add_argument("--dataset", default='raw_results_demo.csv', type=str)


ARGS = ARG_PARSER.parse_args(args=[])


# decay
def decay(data=None, seq_len=ARGS.seq_len, target_field=ARGS.target_field):
    data['interval'] = 0
    j = 0
    for n in range(int(data.shape[0] / seq_len)):
        i = 0
        df_group = data.iloc[n * seq_len:(n * seq_len) + seq_len, :]
        for index, row in df_group.iterrows():  # go over mask
            try:
                if(i == 0):
                    row['interval'] = 0
                    i = 1
                else:
                    if(prev[target_field] == 1):
                        row['interval'] = timedelta.total_seconds(datetime.strptime(str(row['Date'])[:10] + " " + str(row['Time']), "%Y-%m-%d %H:%M:%S")
                                                                  - datetime.strptime(str(prev['Date'])[:10] + " " + str(prev['Time']), "%Y-%m-%d %H:%M:%S"))
                    elif(prev[target_field] == 0):
                        row['interval'] = timedelta.total_seconds(datetime.strptime(str(row['Date'])[:10] + " " + str(row['Time']), "%Y-%m-%d %H:%M:%S")
                                                                  - datetime.strptime(str(prev['Date'])[:10] + " " + str(prev['Time']), "%Y-%m-%d %H:%M:%S")) + prev['interval']
            except ValueError as e:
                print(e)
                print(str(row['Date']) + " " + str(row['Time']))
                break

            prev = row
            data.iloc[j, 3] = row['interval']
            j = j + 1

    data['interval'] = data['interval'].apply(lambda x: abs(x / 60))
    return data


def rdecay(data=None, seq_len=ARGS.seq_len, target_field=ARGS.target_field):
    data['intervalReverse'] = 0
    j = data.shape[0] - 1
    for n in range(int(data.shape[0] / seq_len)):
        i = 0
        df_group = data.iloc[n * seq_len:(n * seq_len) + seq_len, :]
        df_group = df_group[::-1]
        for index, row in df_group.iterrows():  # go over mask
            if(i == 0):
                row['intervalReverse'] = 0
                i = 1
            else:
                if(prev[target_field] == 1):
                    row['intervalReverse'] = timedelta.total_seconds(datetime.strptime(str(row['Date'])[:10] + " " + str(row['Time']), "%Y-%m-%d %H:%M:%S")
                                                                     - datetime.strptime(str(prev['Date'])[:10] + " " + str(prev['Time']), "%Y-%m-%d %H:%M:%S"))
                elif(prev[target_field] == 0):
                    row['intervalReverse'] = timedelta.total_seconds(datetime.strptime(str(row['Date'])[:10] + " " + str(row['Time']), "%Y-%m-%d %H:%M:%S")
                                                                     - datetime.strptime(str(prev['Date'])[:10] + " " + str(prev['Time']), "%Y-%m-%d %H:%M:%S")) + prev['interval']
            prev = row
            data.iloc[j, 4] = row['intervalReverse']
            j = j - 1

    data['intervalReverse'] = data['intervalReverse'].apply(lambda x: abs(x / 60))
    return data


def read_dataset(path_folder=ARGS.path_folder, dataset=ARGS.dataset):

    path_dataset = os.path.join(path_folder, "initial", dataset)
    dataset = pd.read_csv(path_dataset, header=0).fillna(-200)
    dataset["timestamp"] = pd.to_datetime(dataset["date_format"])
    dataset["Date"] = pd.to_datetime(dataset["timestamp"].dt.date, utc=False)
    dataset["Time"] = dataset["timestamp"].dt.time.astype(str)
    dataset["Month"] = dataset.Date.dt.month

    cols = dataset.columns.to_list()

    cols.remove("Date")
    cols.remove("Time")
    cols.remove("Month")
    cols = ["Date", "Time", "Month"] + cols

    dataset = dataset[cols]
    dataset.drop(["date_format", "timestamp"], axis=1, inplace=True)

    return dataset


def add_epoch(dataset=None, mask=None):
    mask[['epoch_format']] = dataset[['epoch_format']].copy()
    return mask



def group_by_seq(dataset=None, seq_len=ARGS.seq_len):

    rows = dataset.groupby('Month').count()['Date'] % seq_len
    rows = pd.DataFrame(rows)
    rows = rows.reset_index()

    final = pd.DataFrame()
    for seq in range(rows.shape[0]):

        temp = dataset[dataset['Month'] == rows.iloc[seq, 0]]
        nrows = temp.shape[0] - rows.iloc[seq, 1]
        temp = temp.iloc[0:nrows, :]
        final = pd.concat([final, temp])

    return final


def split_train_test_val(dataset=None, test_size=ARGS.test_size, val_size=ARGS.val_size):

    x_train, x_temp = train_test_split(dataset, test_size=test_size + val_size, shuffle=False)  # shuffle = False important!
    x_val, x_test = train_test_split(x_temp, test_size=test_size / (test_size + val_size), shuffle=False)  # shuffle = False important!

    assert dataset.shape[0] == x_train.shape[0] + x_test.shape[0] + x_val.shape[0]

    return x_train, x_test, x_val


def get_mask(dataset=None, target_field=ARGS.target_field):
    mask = dataset[[target_field, 'Date', 'Time']].copy()
    mask[mask[target_field] != -200] = 1
    mask[mask[target_field] == -200] = 0
    mask[['Date', 'Time']] = dataset[['Date', 'Time']].copy()
    return mask


def save_files(path_folder=ARGS.path_folder, name_dataset=ARGS.dataset, dataset=None, mask=None, type_dataset=None):

    dir_preprocess = os.path.join(path_folder, "preprocess")
    if not os.path.exists(dir_preprocess):
        os.makedirs(dir_preprocess)

    path_dataset = os.path.join(dir_preprocess, type_dataset + "_" + name_dataset)
    dataset.to_csv(path_dataset, index=False)

    path_mask = os.path.join(path_folder, "preprocess", "mask_" + type_dataset + "_" + name_dataset)
    mask.to_csv(path_mask, index=False)


def preprocess_data(ARGS=ARGS):

    dataset = read_dataset(path_folder=ARGS.path_folder, dataset=ARGS.dataset)

    dataset = group_by_seq(dataset=dataset, seq_len=ARGS.seq_len)

    x_train, x_test, x_val = split_train_test_val(dataset=dataset, test_size=ARGS.test_size, val_size=ARGS.val_size)

    mask_train = get_mask(dataset=x_train, target_field=ARGS.target_field)
    mask_train = decay(mask_train, seq_len=ARGS.seq_len, target_field=ARGS.target_field)
    mask_train = rdecay(mask_train, seq_len=ARGS.seq_len, target_field=ARGS.target_field)

    mask_train = add_epoch(dataset=x_train, mask=mask_train)


    mask_test = get_mask(dataset=x_test, target_field=ARGS.target_field)
    mask_test = decay(mask_test, seq_len=ARGS.seq_len, target_field=ARGS.target_field)
    mask_test = rdecay(mask_test, seq_len=ARGS.seq_len, target_field=ARGS.target_field)

    mask_train = add_epoch(dataset=x_test, mask=mask_test)



    mask_val = get_mask(dataset=x_val, target_field=ARGS.target_field)
    mask_val = decay(mask_val, seq_len=ARGS.seq_len, target_field=ARGS.target_field)
    mask_val = rdecay(mask_val, seq_len=ARGS.seq_len, target_field=ARGS.target_field)

    mask_train = add_epoch(dataset=x_val, mask=mask_val)


    save_files(path_folder=ARGS.path_folder, name_dataset=ARGS.dataset, dataset=x_train, mask=mask_train, type_dataset="train")
    save_files(path_folder=ARGS.path_folder, name_dataset=ARGS.dataset, dataset=x_test, mask=mask_test, type_dataset="test")
    save_files(path_folder=ARGS.path_folder, name_dataset=ARGS.dataset, dataset=x_val, mask=mask_val, type_dataset="val")


if __name__ == "__main__":
    preprocess_data(ARGS=ARGS)
