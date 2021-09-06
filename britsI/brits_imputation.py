import random
import numpy as np
import pandas as pd
import torch.optim as optim
import torch as T
import math
import argparse
import os


from torch.utils.data import DataLoader, Dataset, ConcatDataset, IterableDataset
from argparse import ArgumentParser

from brits2_i_original import BRITS2, run_on_batch


import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

if not os.path.exists("data/saved_models"):
    os.makedirs("data/saved_models")
save_path = "data/saved_models/brits_imputation.pt"



ARG_PARSER = ArgumentParser()

ARG_PARSER.add_argument("--nfeatures", default=609, type=int)
ARG_PARSER.add_argument("--dfeatures", default=43, type=int)
ARG_PARSER.add_argument("--ehidden", default=300, type=int)
ARG_PARSER.add_argument("--model", type=str)

ARG_PARSER.add_argument("--ehr", default=False)
ARG_PARSER.add_argument("--air", default=True)
ARG_PARSER.add_argument("--mimic", default=False)

ARG_PARSER.add_argument("--num_epochs", default=100, type=int)
ARG_PARSER.add_argument("--seq_len", default=3, type=int)
ARG_PARSER.add_argument("--pred_len", default=8, type=int)
ARG_PARSER.add_argument("--batch_size", default=2, type=int)
ARG_PARSER.add_argument("--missingRate", default=10, type=int)
ARG_PARSER.add_argument("--patience", default=30, type=int)
ARG_PARSER.add_argument("--e_lrn_rate", default=0.1, type=float)
ARG_PARSER.add_argument("--g_lrn_rate", default=0.1, type=float)
ARG_PARSER.add_argument("--d_lrn_rate", default=0.001, type=float)
ARG_PARSER.add_argument("--resume_training", default=False)
ARG_PARSER.add_argument("--train", default=True)
ARG_PARSER.add_argument("--evalImp", default=False)
ARG_PARSER.add_argument("--evalPred", default=False)


ARGS = ARG_PARSER.parse_args(args=[])
MAX_SEQ_LEN = ARGS.seq_len
BATCH_SIZE = ARGS.batch_size
EPSILON = 1e-40



# Create Dataset
class CSVDataset(Dataset):
    def __init__(self, path=None, chunksize=0, length=0, seq_len=0, flag=0):
        self.path = path
        self.chunksize = chunksize
        self.len = int(length)  # number of times total getitem is called
        self.seq_len = seq_len
        self.flag = flag
        self.reader = pd.read_csv(self.path, header=0, chunksize=self.chunksize, nrows=20)

    def __getitem__(self, index):
        data = self.reader.get_chunk(self.chunksize)
        data = data.replace(np.inf, 0)
        data = data.replace(np.nan, 0)
        data = data.fillna(0)
        data = data.drop(["date_format"], axis=1)

        if self.flag == 0:
            pids = data["epoch_format"]
            pids = T.as_tensor(pids.values.astype(float), dtype=T.long)

            data = T.as_tensor(data.values.astype(float), dtype=T.float32)
            data = data.view(int(data.shape[0] / self.seq_len), self.seq_len, data.shape[1])
            return data, pids
        else:
            data = T.as_tensor(data.values.astype(float), dtype=T.float32)
            # print("Coucou")
            # print(data.shape[1])
            data = data.view(int(data.shape[0] / self.seq_len), self.seq_len, data.shape[1])

        return data

    def __len__(self):
        return self.len


files = "./data/ibat/initial/raw_results_demo_non_agregated.csv"
# seq_len = 3
# BATCH_SIZE = 2
# dataset = CSVDataset(files, chunksize=int(ARGS.seq_len * BATCH_SIZE), length=1356100, seq_len=3, flag=1)
# def __init__(self, path=None, chunksize=0, length=0, seq_len=0, flag=0):
dataset = CSVDataset(path=files, chunksize=2, length=20, seq_len=1, flag=1)

loader = DataLoader(dataset, batch_size=3, num_workers=0, shuffle=False)  # number of times getitem is called in one iteration



for data in loader:
    print(data)
    input("waiting")