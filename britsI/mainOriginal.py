import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torch as T
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset, ConcatDataset, IterableDataset
import numpy as np
import pandas as pd

import math
import argparse
from brits2_i_original import BRITS2, run_on_batch
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from argparse import ArgumentParser

import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

if not os.path.exists("data/saved_models"):
    os.makedirs("data/saved_models")
# save_path = "data/saved_models/activityReverse.tar"  # vaegan_model - Copy.tar"
save_path = "data/saved_models/brits_original.pt"  # vaegan_model - Copy.tar"


ARG_PARSER = ArgumentParser()

ARG_PARSER.add_argument("--nfeatures", default=609, type=int)
ARG_PARSER.add_argument("--dfeatures", default=43, type=int)
ARG_PARSER.add_argument("--ehidden", default=300, type=int)
ARG_PARSER.add_argument("--model", type=str)

ARG_PARSER.add_argument("--ehr", default=False)
ARG_PARSER.add_argument("--air", default=True)
ARG_PARSER.add_argument("--mimic", default=False)

ARG_PARSER.add_argument("--num_epochs", default=100, type=int)
ARG_PARSER.add_argument("--seq_len", default=40, type=int)
ARG_PARSER.add_argument("--pred_len", default=8, type=int)
ARG_PARSER.add_argument("--batch_size", default=200, type=int)
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
    def __init__(self, path, chunksize, length, seq_len, flag):
        self.path = path
        self.chunksize = chunksize
        self.len = int(length)  # number of times total getitem is called
        self.seq_len = seq_len
        self.flag = flag
        self.reader = pd.read_csv(self.path, header=0, chunksize=self.chunksize)  # ,names=['data']))

    def __getitem__(self, index):
        data = self.reader.get_chunk(self.chunksize)
        data = data.replace(np.inf, 0)
        data = data.replace(np.nan, 0)
        data = data.fillna(0)
        if self.flag == 0:
            pids = data["person_id"]
            pids = T.as_tensor(pids.values.astype(float), dtype=T.long)

            data = T.as_tensor(data.values.astype(float), dtype=T.float32)
            data = data.view(
                int(data.shape[0] / self.seq_len), self.seq_len, data.shape[1]
            )
            return data, pids
        else:
            data = T.as_tensor(data.values.astype(float), dtype=T.float32)
            data = data.view(
                int(data.shape[0] / self.seq_len), self.seq_len, data.shape[1]
            )

        return data

    def __len__(self):
        return self.len


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf  # 11.1179
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, save_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, save_path):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        T.save(
            {"model": model.state_dict(), "trainer": optimizer.state_dict()}, save_path
        )
        self.val_loss_min = val_loss


def pred_test(args, model, predWin):
    model.eval()

    RLoss = 0
    FLoss = 0
    mseLoss = 0
    mseLossF = 0
    TBatches = 0
    oBmi = []
    oBmiF = []
    iBmi = []
    oAge = []
    oSex = []
    imputations = []
    with T.autograd.no_grad():
        # Yannick
        # for i in ["M", "F"]:
        #     files = "cond" + i + "test.csv"
        #     maskFiles = "mask" + i + "test.csv"
        for i in [1]:
            files = "data/air/preprocess/airTest.csv"
            maskFiles = "data/air/preprocess/airTestMask.csv"

            dataset = CSVDataset(
                files, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=0
            )
            maskDataset = CSVDataset(
                maskFiles, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=1
            )

            loader = DataLoader(
                dataset, batch_size=1, num_workers=0, shuffle=False
            )  # number of times getitem is called in one iteration
            maskLoader = DataLoader(
                maskDataset, batch_size=1, num_workers=0, shuffle=False
            )

            loss = {}

            # for every batch
            for batch_idx, allData in enumerate(zip(loader, maskLoader)):
                data, mask = allData
                data = data[0]
                data = data[:, :, :, 1:]

                decay = mask[:, :, :, 6]
                rdecay = mask[:, :, :, 7]
                bmi = mask[:, :, :, 5]
                mask = mask[:, :, :, 4]

                bmi = bmi.unsqueeze(3)

                data = torch.cat((data, bmi), dim=3)

                data = data.squeeze()
                mask = mask.squeeze()
                decay = decay.squeeze()
                rdecay = rdecay.squeeze()
                bmi = bmi.squeeze()

                y = data.clone().detach()
                testMask = mask.clone().detach()
                y = y[:, :, 811]

                for i in range(data.shape[0]):
                    j = 40
                    if predWin == 8:
                        k = 32
                    elif predWin == 7:
                        k = 28
                    elif predWin == 6:
                        k = 24
                    elif predWin == 5:
                        k = 20

                    data[i, j - k: j, :] = 0
                    mask[i, j - k: j] = 0
                    y[i, 0: j - k] = 0
                    testMask[i, 0: j - k] = 0

                ret_f, ret = run_on_batch(
                    model, data, mask, decay, rdecay, args, optimizer=None, epoch=None
                )  # ,bmi_norm)
                RLoss = RLoss + ret["loss"]
                FLoss = FLoss + ret_f["loss"]
                outputBMI = ret["imputations"] * testMask
                outputBMIF = ret_f["imputations"] * testMask
                mseLoss = mseLoss + (torch.sum(torch.abs(outputBMI - y))) / (
                    torch.sum(testMask) + 1e-5
                )
                mseLossF = mseLossF + (torch.sum(torch.abs(outputBMIF - y))) / (
                    torch.sum(testMask) + 1e-5
                )
                outBmi, outBmiF, inBmi = plotBmi(outputBMI, outputBMIF, y, testMask)
                oBmi.extend(outBmi)
                oBmiF.extend(outBmiF)
                iBmi.extend(inBmi)

            TBatches = TBatches + batch_idx + 1
        RLoss = RLoss / TBatches
        mseLoss = mseLoss / TBatches
        mseLossF = mseLossF / TBatches
    oBmi = np.asarray(oBmi)
    iBmi = np.asarray(iBmi)
    loss = oBmi - iBmi
    loss = np.asarray([abs(number) for number in loss])
    variance = sum([((x - mseLoss) ** 2) for x in loss]) / len(loss)
    res = variance ** 0.5
    ci = 1.96 * (res / (math.sqrt(len(loss))))

    print("CI", ci)
    print("MAE Loss Reverse:", mseLoss)
    print("MAE Loss Forward:", mseLossF)
    return oBmi, oBmiF, iBmi


def imputation_test(args, model, missingRate):
    model.eval()

    RLoss = 0
    FLoss = 0
    mseLoss = 0
    mseLossF = 0
    TBatches = 0
    oBmi = []
    oBmiF = []
    iBmi = []
    oAge = []
    oSex = []
    imputations = []
    samples = 0
    pids = 0
    with T.autograd.no_grad():
        # Yannick
        # for i in ["M", "F"]:
        #     files = "cond" + i + "test.csv"
        #     maskFiles = "mask" + i + "test.csv"
        for i in [1]:
            files = "data/air/preprocess/airTest.csv"
            maskFiles = "data/air/preprocess/airTestMask.csv"

            dataset = CSVDataset(
                files, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=0
            )
            maskDataset = CSVDataset(
                maskFiles, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=1
            )

            loader = DataLoader(
                dataset, batch_size=1, num_workers=0, shuffle=False
            )  # number of times getitem is called in one iteration
            maskLoader = DataLoader(
                maskDataset, batch_size=1, num_workers=0, shuffle=False
            )

            loss = {}

            # for every batch
            for batch_idx, allData in enumerate(zip(loader, maskLoader)):
                data, mask = allData
                data = data[0]
                data = data[:, :, :, 1:]

                decay = mask[:, :, :, 6]
                rdecay = mask[:, :, :, 7]
                bmi = mask[:, :, :, 5]
                mask = mask[:, :, :, 4]

                bmi = bmi.unsqueeze(3)

                data = torch.cat((data, bmi), dim=3)

                data = data.squeeze()
                mask = mask.squeeze()
                decay = decay.squeeze()
                rdecay = rdecay.squeeze()
                bmi = bmi.squeeze()

                # values to be predicted
                y = data.clone().detach()
                testMask = mask.clone().detach()

                y = y[:, :, 811]

                bmi = 811

                for i in range(data.shape[0]):
                    idxs = torch.nonzero(mask[i, :] == 1)
                    samples = samples + list(idxs.size())[0]
                    if (missingRate == 50) & (list(idxs.size())[0] > 4):
                        idxs = random.sample(set(idxs), 5)
                        data[i, idxs[0], bmi] = 0
                        data[i, idxs[1], bmi] = 0
                        data[i, idxs[2], bmi] = 0
                        data[i, idxs[3], bmi] = 0
                        data[i, idxs[4], bmi] = 0
                        mask[i, idxs[0]] = 0
                        mask[i, idxs[1]] = 0
                        mask[i, idxs[2]] = 0
                        mask[i, idxs[3]] = 0
                        mask[i, idxs[4]] = 0
                        pids = pids + 5
                    elif (missingRate >= 40) & (list(idxs.size())[0] > 3):
                        idxs = random.sample(set(idxs), 4)
                        data[i, idxs[0], bmi] = 0
                        data[i, idxs[1], bmi] = 0
                        data[i, idxs[2], bmi] = 0
                        data[i, idxs[3], bmi] = 0
                        mask[i, idxs[0]] = 0
                        mask[i, idxs[1]] = 0
                        mask[i, idxs[2]] = 0
                        mask[i, idxs[3]] = 0
                        pids = pids + 4
                    elif (missingRate >= 30) & (list(idxs.size())[0] > 2):
                        idxs = random.sample(set(idxs), 3)
                        data[i, idxs[0], bmi] = 0
                        data[i, idxs[1], bmi] = 0
                        data[i, idxs[2], bmi] = 0
                        mask[i, idxs[0]] = 0
                        mask[i, idxs[1]] = 0
                        mask[i, idxs[2]] = 0
                        pids = pids + 3
                    elif (missingRate >= 20) & (list(idxs.size())[0] > 1):
                        idxs = random.sample(set(idxs), 2)
                        data[i, idxs[0], bmi] = 0
                        data[i, idxs[1], bmi] = 0
                        mask[i, idxs[0]] = 0
                        mask[i, idxs[1]] = 0
                        pids = pids + 2
                    elif (missingRate >= 10) & (list(idxs.size())[0] > 0):
                        if i % 2 == 0:
                            idxs = random.sample(set(idxs), 1)
                            data[i, idxs, bmi] = 0
                            mask[i, idxs] = 0
                            pids = pids + 1

                    testMask[i, :] = testMask[i, :] - mask[i, :]
                    y[i, :] = y[i, :] * testMask[i, :]

                ret_f, ret = run_on_batch(
                    model, data, mask, decay, rdecay, args, optimizer=None, epoch=None
                )  # ,bmi_norm)
                RLoss = RLoss + ret["loss"]
                FLoss = FLoss + ret_f["loss"]
                outputBMI = ret["imputations"] * testMask
                outputBMIF = ret_f["imputations"] * testMask
                mseLoss = mseLoss + (torch.sum(torch.abs(outputBMI - y))) / (
                    torch.sum(testMask) + 1e-5
                )
                mseLossF = mseLossF + (torch.sum(torch.abs(outputBMIF - y))) / (
                    torch.sum(testMask) + 1e-5
                )
                outBmi, outBmiF, inBmi = plotBmi(outputBMI, outputBMIF, y, testMask)
                oBmi.extend(outBmi)
                oBmiF.extend(outBmiF)
                iBmi.extend(inBmi)

            TBatches = TBatches + batch_idx + 1
        RLoss = RLoss / TBatches
        mseLoss = mseLoss / TBatches
        mseLossF = mseLossF / TBatches
    oBmi = np.asarray(oBmi)
    iBmi = np.asarray(iBmi)
    loss = oBmi - iBmi
    loss = np.asarray([abs(number) for number in loss])
    variance = sum([((x - mseLoss) ** 2) for x in loss]) / len(loss)
    res = variance ** 0.5
    ci = 1.96 * (res / (math.sqrt(len(loss))))

    print("CI", ci)
    print("MAE Loss Reverse:", mseLoss)
    print("Total BMI values", samples)
    print("Deleted BMIs", pids)
    print("Missing%", pids / samples)
    return oBmi, oBmiF, iBmi


def plotBmi(outBmi, outBmiF, inBmi, testMask):

    outBmi = outBmi.cpu().detach().numpy()
    outBmiF = outBmiF.cpu().detach().numpy()
    inBmi = inBmi.cpu().detach().numpy()
    testMask = testMask.cpu().detach().numpy()

    outBmi = outBmi[np.nonzero(testMask)]
    outBmiF = outBmiF[np.nonzero(testMask)]
    inBmi = inBmi[np.nonzero(testMask)]

    return outBmi, outBmiF, inBmi


def run_evalFull(args, model):
    model.eval()

    RLoss = 0
    TBatches = 0
    oBmi = []
    iBmi = []
    oAge = []
    oSex = []

    with T.autograd.no_grad():
        # Yannick
        # for i in ["M", "F"]:
        #     files = "cond" + i + "val.csv"
        #     maskFiles = "mask" + i + "val.csv"
        for i in [1]:
            files = "data/air/preprocess/airTest.csv"
            maskFiles = "data/air/preprocess/airTestMask.csv"

            dataset = CSVDataset(files, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=0)
            maskDataset = CSVDataset(maskFiles, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=1)

            loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)  # number of times getitem is called in one iteration
            maskLoader = DataLoader(maskDataset, batch_size=1, num_workers=0, shuffle=False)

            loss = {}

            # for every batch
            for batch_idx, allData in enumerate(zip(loader, maskLoader)):
                data, mask = allData
                pids = data[1]
                data = data[0]
                data = data[:, :, :, 1:]

                decay = mask[:, :, :, 6]
                rdecay = mask[:, :, :, 7]
                bmi = mask[:, :, :, 5]
                mask = mask[:, :, :, 4]
                bmi = bmi.unsqueeze(3)
                data = torch.cat((data, bmi), dim=3)
                data = data.squeeze()
                mask = mask.squeeze()
                decay = decay.squeeze()
                rdecay = rdecay.squeeze()
                bmi = bmi.squeeze()

                ret_f, ret = run_on_batch(model, data, mask, decay, rdecay, args, optimizer=None)  # ,bmi_norm)
                RLoss = RLoss + ret["loss"]

            TBatches = TBatches + batch_idx + 1
        RLoss = RLoss / TBatches
    print("Val R Loss:", RLoss)
    return RLoss


def run_epoch(args, model):
    """Run a single epoch"""

    trainLoss = []
    valLoss = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    if args.resume_training:
        checkpoint = T.load(save_path)
        optimizer.load_state_dict(checkpoint["trainer"])
        early_stopping(16.972769, model, optimizer, save_path)

    # for evrey epoch
    for epoch in range(args.num_epochs):
        model.train()

        # Running Losses
        RLoss = 0
        TBatches = 0
        print("=============EPOCH=================")
        # Yannick
        # for i in ["M", "F"]:
        #     files = "cond" + i + "train.csv"
        #     maskFiles = "mask" + i + "train.csv"

        for i in [1]:
            files = "data/air/preprocess/airTest.csv"
            maskFiles = "data/air/preprocess/airTestMask.csv"

            dataset = CSVDataset(files, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=0)
            maskDataset = CSVDataset(maskFiles, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=1)

            loader = DataLoader(
                dataset, batch_size=1, num_workers=0, shuffle=False
            )  # number of times getitem is called in one iteration
            maskLoader = DataLoader(
                maskDataset, batch_size=1, num_workers=0, shuffle=False
            )
            # for every batch
            for batch_idx, allData in enumerate(zip(loader, maskLoader)):
                data, mask = allData
                pids = data[1]
                data = data[0]
                data = data[:, :, :, 1:]

                decay = mask[:, :, :, 6]
                rdecay = mask[:, :, :, 7]
                bmi = mask[:, :, :, 5]
                mask = mask[:, :, :, 4]

                bmi = bmi.unsqueeze(3)
                data = torch.cat((data, bmi), dim=3)

                data = data.squeeze()
                mask = mask.squeeze()
                decay = decay.squeeze()
                rdecay = rdecay.squeeze()
                bmi = bmi.squeeze()

                ret_f, ret = run_on_batch(
                    model, data, mask, decay, rdecay, args, optimizer
                )  # ,bmi_norm)
                RLoss = RLoss + ret["loss"].item()

            TBatches = TBatches + batch_idx + 1
        RLoss = RLoss / TBatches

        print("EPOCH:", epoch, "loss_R:", "%.4f" % RLoss)

        trainLoss.append(RLoss)

        valid_loss = run_evalFull(args, model)

        valLoss.append(valid_loss)

        if not (T.isnan(valid_loss)):
            early_stopping(valid_loss, model, optimizer, save_path)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # plot_grad_flow(model['e'].named_parameters())
        # plot_grad_flow(model['g'].named_parameters())
        # plot_grad_flow(model['d'].named_parameters())
    return trainLoss, valLoss


def run(args):

    train_on_gpu = T.cuda.is_available()
    if train_on_gpu:
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")

    model = BRITS2(args)

    if torch.cuda.is_available():
        model = model.cuda()

    if args.resume_training:
        checkpoint = T.load(save_path)
        model.load_state_dict(checkpoint["model"])
        trainLoss, valLoss = run_epoch(args, model)
        return trainLoss, valLoss

    elif args.train:
        trainLoss, valLoss = run_epoch(args, model)
        return trainLoss, valLoss

    elif args.evalImp:
        # load Model
        checkpoint = T.load(save_path)
        model.load_state_dict(checkpoint["model"])
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        optimizer.load_state_dict(checkpoint["trainer"])
        oBmi, oBmiF, iBmi = imputation_test(args, model, args.missingRate)

    elif args.evalPred:
        # load Model
        checkpoint = T.load(save_path)
        model.load_state_dict(checkpoint["model"])
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        optimizer.load_state_dict(checkpoint["trainer"])
        oBmi, oBmiF, iBmi = pred_test(args, model, args.pred_len)

        # return oBmi, oBmiF, iBmi


# trainLoss, valLoss = run(ARGS)
# oBmi, oBmiF, iBmi, oAge, oSex = run(ARGS)
run(ARGS)
