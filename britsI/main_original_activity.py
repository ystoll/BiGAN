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
import import_ipynb
from brits2_i_original import BRITS2, run_on_batch
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from argparse import ArgumentParser

# save_path = "data/saved_models/airReverse.tar"  # vaegan_model - Copy.tar"
save_path = "data/saved_models/essai.pt"
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

if not os.path.exists("data/saved_models"):
    os.makedirs("data/saved_models")


ARG_PARSER = ArgumentParser()

ARG_PARSER.add_argument("--nfeatures", default=609, type=int)
ARG_PARSER.add_argument("--dfeatures", default=43, type=int)
ARG_PARSER.add_argument("--ehidden", default=300, type=int)
ARG_PARSER.add_argument("--model", type=str)

ARG_PARSER.add_argument("--air", default=True)
ARG_PARSER.add_argument("--mimic", default=False)
ARG_PARSER.add_argument("--ehr", default=False)

ARG_PARSER.add_argument("--num_epochs", default=3, type=int)
ARG_PARSER.add_argument("--seq_len", default=20, type=int)
ARG_PARSER.add_argument("--pred_len", default=8, type=int)
ARG_PARSER.add_argument("--missingRate", default=10, type=int)
ARG_PARSER.add_argument("--patience", default=200, type=int)
ARG_PARSER.add_argument("--e_lrn_rate", default=0.1, type=float)
ARG_PARSER.add_argument("--g_lrn_rate", default=0.1, type=float)
ARG_PARSER.add_argument("--d_lrn_rate", default=0.001, type=float)
ARG_PARSER.add_argument("--resume_training", default=False)
ARG_PARSER.add_argument("--train", default=False)
ARG_PARSER.add_argument("--evalImp", default=False)
ARG_PARSER.add_argument("--evalPred", default=True)


ARGS = ARG_PARSER.parse_args(args=[])
MAX_SEQ_LEN = ARGS.seq_len
# Yannick
# BATCH_SIZE = ARGS.batch_size
BATCH_SIZE = 64
EPSILON = 1e-40


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
            print( f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..." )
        print(save_path)
        input("waiting")
        T.save({"model": model.state_dict(), "trainer": optimizer.state_dict()}, save_path)
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
        if args.air:
            data = pd.read_csv("./data/air/preprocess/airTest.csv", header=0)
            mask = pd.read_csv("./data/air/preprocess/airTestMask.csv", header=0)
            data = data[
                [
                    "Date",
                    "Time",
                    "Month",
                    "PT08.S1(CO)",
                    "CO(GT)",
                    "NMHC(GT)",
                    "C6H6(GT)",
                    "PT08.S2(NMHC)",
                    "NOx(GT)",
                    "PT08.S3(NOx)",
                    "NO2(GT)",
                    "PT08.S4(NO2)",
                    "PT08.S5(O3)",
                    "T",
                    "RH",
                    "AH",
                ]
            ]
            del data["Time"]
            del data["Date"]
            del mask["Time"]
            del mask["Date"]

        if args.mimic:
            data = pd.read_csv(".../aaai/data/mimic/preprocess/mimicTest.csv", header=0)
            mask = pd.read_csv(
                ".../aaai/data/mimic/preprocess/mimicTestMask.csv", header=0
            )
            del data["subject_id"]
            del data["charttime"]
            del mask["subject_id"]
            del mask["charttime"]
            print(data.shape)
            data = data[
                [
                    "ALBUMIN",
                    "ANION GAP",
                    "WBC",
                    "BANDS",
                    "BICARBONATE",
                    "BILIRUBIN",
                    "BUN",
                    "CHLORIDE",
                    "CREATININE",
                    "GLUCOSE",
                    "HEMATOCRIT",
                    "HEMOGLOBIN",
                    "INR",
                    "LACTATE",
                    "PaCO2",
                    "PLATELET",
                    "POTASSIUM",
                    "PT",
                    "PTT",
                    "SODIUM",
                ]
            ]

        data = T.as_tensor(data.values.astype(float), dtype=T.float32)
        data = data.view(int(data.shape[0] / args.seq_len), args.seq_len, data.shape[1])

        mask = T.as_tensor(mask.values.astype(float), dtype=T.float32)
        mask = mask.view(int(mask.shape[0] / args.seq_len), args.seq_len, mask.shape[1])

        loss = {}

        decay = mask[:, :, 1]
        rdecay = mask[:, :, 2]
        mask = mask[:, :, 0]

        data = data.squeeze()
        mask = mask.squeeze()
        decay = decay.squeeze()
        rdecay = rdecay.squeeze()
        # print(data.shape)
        # print(mask.shape)
        # print(decay.shape)

        # values to be predicted
        y = data.clone().detach()
        testMask = mask.clone().detach()

        y = y[:, :, 2]
        # ------------remove last 5 timestamps------------------
        # print(data[0:10,8:,653])
        for i in range(data.shape[0]):
            # if(data[i,])
            j = 20
            if predWin == 8:
                k = 16
                decay[i, j - k : j] = T.tensor(
                    [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
                )
                rdecay[i, j - k : j] = T.tensor(
                    [8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5]
                )
            elif predWin == 7:
                k = 14
                decay[i, j - k : j] = T.tensor(
                    [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
                )
                rdecay[i, j - k : j] = (
                    T.tensor([7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5])
                    * 120
                )
            elif predWin == 6:
                k = 12
                decay[i, j - k : j] = T.tensor(
                    [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
                )
                rdecay[i, j - k : j] = T.tensor(
                    [6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5]
                )
            elif predWin == 5:
                k = 10
                decay[i, j - k : j] = T.tensor([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
                rdecay[i, j - k : j] = T.tensor(
                    [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5]
                )

            data[i, j - k : j, :] = 0
            mask[i, j - k : j] = 0
            y[i, 0 : j - k] = 0
            testMask[i, 0 : j - k] = 0

        ret_f, ret = run_on_batch(
            model, data, mask, decay, rdecay, args, optimizer=None
        )
        RLoss = RLoss + ret["loss"]
        FLoss = FLoss + ret_f["loss"]
        testMask = testMask.cuda()
        y = y.cuda()
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

        RLoss = RLoss
        mseLoss = mseLoss
        mseLossF = mseLossF
    # print("===================================")
    # print("Val R Loss:",RLoss)
    oBmi = np.asarray(oBmi)
    iBmi = np.asarray(iBmi)
    loss = oBmi - iBmi
    loss = np.asarray([abs(number) for number in loss])
    variance = sum([((x - mseLoss) ** 2) for x in loss]) / len(loss)
    res = variance ** 0.5
    ci = 1.96 * (res / (math.sqrt(len(loss))))
    print("CI", ci)
    print("MAE Loss Reverse:", mseLoss)
    # print("MAE Loss Forward:",mseLossF)
    # print(outputBMI)
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
        if args.air:
            data = pd.read_csv("./data/air/preprocess/airTest.csv", header=0)
            mask = pd.read_csv("./data/air/preprocess/airTestMask.csv", header=0)
            data = data[
                [
                    "Date",
                    "Time",
                    "Month",
                    "PT08.S1(CO)",
                    "CO(GT)",
                    "NMHC(GT)",
                    "C6H6(GT)",
                    "PT08.S2(NMHC)",
                    "NOx(GT)",
                    "PT08.S3(NOx)",
                    "NO2(GT)",
                    "PT08.S4(NO2)",
                    "PT08.S5(O3)",
                    "T",
                    "RH",
                    "AH",
                ]
            ]
            del data["Time"]
            del data["Date"]
            del mask["Time"]
            del mask["Date"]

        if args.mimic:
            data = pd.read_csv("./data/mimic/preprocess/mimicTest.csv", header=0)
            mask = pd.read_csv("./data/mimic/preprocess/mimicTestMask.csv", header=0)
            del data["subject_id"]
            del data["charttime"]
            del mask["subject_id"]
            del mask["charttime"]
            print(data.shape)
            data = data[
                [
                    "ALBUMIN",
                    "ANION GAP",
                    "WBC",
                    "BANDS",
                    "BICARBONATE",
                    "BILIRUBIN",
                    "BUN",
                    "CHLORIDE",
                    "CREATININE",
                    "GLUCOSE",
                    "HEMATOCRIT",
                    "HEMOGLOBIN",
                    "INR",
                    "LACTATE",
                    "PaCO2",
                    "PLATELET",
                    "POTASSIUM",
                    "PT",
                    "PTT",
                    "SODIUM",
                ]
            ]

        data = T.as_tensor(data.values.astype(float), dtype=T.float32)
        data = data.view(int(data.shape[0] / args.seq_len), args.seq_len, data.shape[1])

        mask = T.as_tensor(mask.values.astype(float), dtype=T.float32)
        mask = mask.view(int(mask.shape[0] / args.seq_len), args.seq_len, mask.shape[1])

        loss = {}

        decay = mask[:, :, 1]
        rdecay = mask[:, :, 2]
        mask = mask[:, :, 0]

        data = data.squeeze()
        mask = mask.squeeze()
        decay = decay.squeeze()
        rdecay = rdecay.squeeze()

        # values to be predicted
        y = data.clone().detach()
        testMask = mask.clone().detach()
        y = y[:, :, 2]

        # ------------remove last 5 timestamps------------------
        # print(data[0:10,8:,653])
        for i in range(data.shape[0]):
            # if(data[i,])
            j = 20
            k = 16
            # mask[i,:].loc[mask[i,:].query('value == 1').sample(frac=.1).index,'value'] = 0
            idxs = torch.nonzero(mask[i, :] == 1)
            # print(idxs)
            samples = samples + list(idxs.size())[0]
            if missingRate == 50:
                if list(idxs.size())[0] > 4:
                    idxs = random.sample(set(idxs), 5)
                    data[i, idxs[0], 2] = 0
                    data[i, idxs[1], 2] = 0
                    data[i, idxs[2], 2] = 0
                    data[i, idxs[3], 2] = 0
                    data[i, idxs[4], 2] = 0
                    mask[i, idxs[0]] = 0
                    mask[i, idxs[1]] = 0
                    mask[i, idxs[2]] = 0
                    mask[i, idxs[3]] = 0
                    mask[i, idxs[4]] = 0
                    pids = pids + 5
                    break
            if missingRate >= 40:
                if list(idxs.size())[0] > 3:
                    idxs = random.sample(set(idxs), 4)
                    data[i, idxs[0], 2] = 0
                    data[i, idxs[1], 2] = 0
                    data[i, idxs[2], 2] = 0
                    data[i, idxs[3], 2] = 0
                    mask[i, idxs[0]] = 0
                    mask[i, idxs[1]] = 0
                    mask[i, idxs[2]] = 0
                    mask[i, idxs[3]] = 0
                    pids = pids + 4
                    break
            if missingRate >= 30:
                if list(idxs.size())[0] > 2:
                    idxs = random.sample(set(idxs), 3)
                    data[i, idxs[0], 2] = 0
                    data[i, idxs[1], 2] = 0
                    data[i, idxs[2], 2] = 0
                    mask[i, idxs[0]] = 0
                    mask[i, idxs[1]] = 0
                    mask[i, idxs[2]] = 0
                    pids = pids + 3
                    break
            if missingRate >= 20:
                if list(idxs.size())[0] > 1:
                    idxs = random.sample(set(idxs), 2)
                    data[i, idxs[0], 2] = 0
                    data[i, idxs[1], 2] = 0
                    mask[i, idxs[0]] = 0
                    mask[i, idxs[1]] = 0
                    pids = pids + 2
                    break
            if missingRate >= 10:
                if list(idxs.size())[0] > 0:
                    idxs = random.sample(set(idxs), 1)
                    data[i, idxs, 2] = 0
                    mask[i, idxs] = 0
                    pids = pids + 1

            testMask[i, :] = testMask[i, :] - mask[i, :]

            y[i, :] = y[i, :] * testMask[i, :]

        ret_f, ret = run_on_batch(
            model, data, mask, decay, rdecay, args, optimizer=None
        )
        RLoss = RLoss + ret["loss"]
        FLoss = FLoss + ret_f["loss"]
        testMask = testMask.cuda()
        y = y.cuda()
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

        RLoss = RLoss
        mseLoss = mseLoss
        mseLossF = mseLossF
    oBmi = np.asarray(oBmi)
    iBmi = np.asarray(iBmi)
    loss = oBmi - iBmi
    loss = np.asarray([abs(number) for number in loss])
    variance = sum([((x - mseLoss) ** 2) for x in loss]) / len(loss)
    res = variance ** 0.5
    ci = 1.96 * (res / (math.sqrt(len(loss))))

    print("CI", ci)
    print("MAE Loss Reverse:", mseLoss)
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
        if args.air:
            data = pd.read_csv("./data/air/preprocess/airVal.csv", header=0)
            mask = pd.read_csv("./data/air/preprocess/airValMask.csv", header=0)
            data = data[
                [
                    "Date",
                    "Time",
                    "Month",
                    "PT08.S1(CO)",
                    "CO(GT)",
                    "NMHC(GT)",
                    "C6H6(GT)",
                    "PT08.S2(NMHC)",
                    "NOx(GT)",
                    "PT08.S3(NOx)",
                    "NO2(GT)",
                    "PT08.S4(NO2)",
                    "PT08.S5(O3)",
                    "T",
                    "RH",
                    "AH",
                ]
            ]
            del data["Time"]
            del data["Date"]
            del mask["Time"]
            del mask["Date"]

        if args.mimic:
            data = pd.read_csv("./data/mimic/preprocess/mimicVal.csv", header=0)
            mask = pd.read_csv(
                "./data/mimic/preprocess/mimicValMask.csv", header=0
            )
            del data["subject_id"]
            del data["charttime"]
            del mask["subject_id"]
            del mask["charttime"]
            print(data.shape)
            data = data[
                [
                    "ALBUMIN",
                    "ANION GAP",
                    "WBC",
                    "BANDS",
                    "BICARBONATE",
                    "BILIRUBIN",
                    "BUN",
                    "CHLORIDE",
                    "CREATININE",
                    "GLUCOSE",
                    "HEMATOCRIT",
                    "HEMOGLOBIN",
                    "INR",
                    "LACTATE",
                    "PaCO2",
                    "PLATELET",
                    "POTASSIUM",
                    "PT",
                    "PTT",
                    "SODIUM",
                ]
            ]

        data = T.as_tensor(data.values.astype(float), dtype=T.float32)
        data = data.view(int(data.shape[0] / args.seq_len), args.seq_len, data.shape[1])

        mask = T.as_tensor(mask.values.astype(float), dtype=T.float32)
        mask = mask.view(int(mask.shape[0] / args.seq_len), args.seq_len, mask.shape[1])

        loss = {}

        decay = mask[:, :, 1]
        rdecay = mask[:, :, 2]
        mask = mask[:, :, 0]

        data = data.squeeze()
        mask = mask.squeeze()
        decay = decay.squeeze()
        rdecay = rdecay.squeeze()
        # print(data.shape)
        # print(mask.shape)
        # print(decay.shape)

        ret_f, ret = run_on_batch(
            model, data, mask, decay, rdecay, args, optimizer=None
        )
        RLoss = RLoss + ret["loss"]

        RLoss = RLoss
    print("Val R Loss:", RLoss)
    return RLoss


def run_epoch(args, model):
    """Run a single epoch"""

    trainLoss = []
    valLoss = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    if args.resume_training:
        early_stopping(133, model, optimizer, save_path)

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # for evrey epoch
    for epoch in range(args.num_epochs):
        model.train()

        # Running Losses
        RLoss = 0
        TBatches = 0
        print("=============EPOCH=================")

        if args.air:
            data = pd.read_csv("./data/air/preprocess/airTrain.csv", header=0)
            mask = pd.read_csv(
                "./data/air/preprocess/airTrainMask.csv", header=0
            )
            data = data[
                [
                    "Date",
                    "Time",
                    "Month",
                    "PT08.S1(CO)",
                    "CO(GT)",
                    "NMHC(GT)",
                    "C6H6(GT)",
                    "PT08.S2(NMHC)",
                    "NOx(GT)",
                    "PT08.S3(NOx)",
                    "NO2(GT)",
                    "PT08.S4(NO2)",
                    "PT08.S5(O3)",
                    "T",
                    "RH",
                    "AH",
                ]
            ]
            del data["Time"]
            del data["Date"]
            del mask["Time"]
            del mask["Date"]

        if args.mimic:
            data = pd.read_csv(
                "./data/mimic/preprocess/mimicTrain.csv", header=0
            )
            mask = pd.read_csv(
                "./data/mimic/preprocess/mimicTrainMask.csv", header=0
            )
            del data["subject_id"]
            del data["charttime"]
            del mask["subject_id"]
            del mask["charttime"]
            print(data.shape)
            data = data[
                [
                    "ALBUMIN",
                    "ANION GAP",
                    "WBC",
                    "BANDS",
                    "BICARBONATE",
                    "BILIRUBIN",
                    "BUN",
                    "CHLORIDE",
                    "CREATININE",
                    "GLUCOSE",
                    "HEMATOCRIT",
                    "HEMOGLOBIN",
                    "INR",
                    "LACTATE",
                    "PaCO2",
                    "PLATELET",
                    "POTASSIUM",
                    "PT",
                    "PTT",
                    "SODIUM",
                ]
            ]

        # print(data.head())
        data = T.as_tensor(data.values.astype(float), dtype=T.float32)
        data = data.view(int(data.shape[0] / args.seq_len), args.seq_len, data.shape[1])

        mask = T.as_tensor(mask.values.astype(float), dtype=T.float32)
        mask = mask.view(int(mask.shape[0] / args.seq_len), args.seq_len, mask.shape[1])

        decay = mask[:, :, 1]
        rdecay = mask[:, :, 2]
        mask = mask[:, :, 0]

        data = data.squeeze()
        mask = mask.squeeze()
        decay = decay.squeeze()
        rdecay = rdecay.squeeze()

        ret_f, ret = run_on_batch(
            model, data, mask, decay, rdecay, args, optimizer
        )
        RLoss = RLoss + ret["loss"].item()

        RLoss = RLoss

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
        model.load_state_dict(checkpoint)
        optimizer.load_state_dict(checkpoint)
        output = run_epoch(args, model)
        # return model, output

    elif args.train:
        trainLoss, valLoss = run_epoch(args, model)
        # return trainLoss, valLoss

    elif args.evalImp:
        # load Model
        checkpoint = T.load(save_path)
        model.load_state_dict(checkpoint["model"])
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(checkpoint["trainer"])
        oBmi, oBmiF, iBmi = imputation_test(args, model, args.missingRate)

    elif args.evalPred:
        # load Model
        checkpoint = T.load(save_path)
        model.load_state_dict(checkpoint["model"])
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(checkpoint["trainer"])
        oBmi, oBmiF, iBmi = pred_test(args, model, args.pred_len)

        # return oBmi, oBmiF, iBmi


# trainLoss, valLoss = run(ARGS)
# oBmi, oBmiF, iBmi = run(ARGS)
run(ARGS)
