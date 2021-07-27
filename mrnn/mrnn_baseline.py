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
# import import_ipynb
from mrnn import TemporalDecay, MRNN, run_on_batch
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from argparse import ArgumentParser

save_path = "data/aaai/mrnn.tar"  # vaegan_model - Copy.tar"
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

ARG_PARSER = ArgumentParser()

ARG_PARSER.add_argument("--nfeatures", default=609, type=int)
ARG_PARSER.add_argument("--dfeatures", default=43, type=int)
ARG_PARSER.add_argument("--ehidden", default=300, type=int)
ARG_PARSER.add_argument("--model", type=str)

ARG_PARSER.add_argument("--ehr", default=False)
ARG_PARSER.add_argument("--air", default=False)
ARG_PARSER.add_argument("--mimic", default=True)

ARG_PARSER.add_argument("--num_epochs", default=5, type=int)
ARG_PARSER.add_argument("--seq_len", default=40, type=int)
ARG_PARSER.add_argument("--pred_len", default=5, type=int)
ARG_PARSER.add_argument("--batch_size", default=200, type=int)
ARG_PARSER.add_argument("--missingRate", default=10, type=int)
ARG_PARSER.add_argument("--patience", default=40, type=int)
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
        self.reader = pd.read_csv(
            self.path, header=0, chunksize=self.chunksize
        )  # ,names=['data']))

    def __getitem__(self, index):
        data = self.reader.get_chunk(self.chunksize)
        # sex=pd.read_csv('C:\\Users/mehak/Desktop/demo.csv',header=0)
        # sex=sex[['person_id','Sex']]
        # data = pd.merge(data, sex, how='left', on=['person_id'])
        # print(data.shape)
        # data=data.sort_values(by=['RANDOM_PATIENT_ID','VISIT_YEAR','VISIT_MONTH'])
        # print(data['RANDOM_PATIENT_ID'].unique())
        #         del data['person_id']
        #         print(data.columns.get_loc('BMI'))
        # print(data.columns)

        data = data.replace(np.inf, 0)
        data = data.replace(np.nan, 0)
        data = data.fillna(0)
        # print(data.shape)
        if self.flag == 0:
            #             data['Age']=data['Age'].apply(lambda x: ((x*12)/3)-81)
            # Yannick
            pids = data["subject_id"]
            #             age=data['age']
            #             del data['age']
            pids = T.as_tensor(pids.values.astype(float), dtype=T.long)
            #             age = T.as_tensor(age.values.astype(float), dtype=T.long)
            #             print("age",data['Age'])
            #             print("pids",list(pids))
            #             print("========================================================")

            data = T.as_tensor(data.values.astype(float), dtype=T.float32)
            #         print(list(data[:,0]))
            #         print("========================================================")
            # data=T.from_numpy(data)
            # data=data.double()
            data = data.view(
                int(data.shape[0] / self.seq_len), self.seq_len, data.shape[1]
            )
            # print(data.shape)
            # print("age",data[0,:,203])
            # mask=pd.DataFrame()
            # mask = data.loc[data['LABS_LDL_MEAN']>0,'LABS_LDL_MEAN']=1
            # df[df['LABS_LDL_MEAN']<0].count()
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

    def __init__(self, patience=7, verbose=True, delta=0):
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
        for i in ["M", "F"]:
            files = "cond" + i + "test.csv"

            #         drugFiles = 'drug'+G+'val.csv'

            maskFiles = "mask" + i + "test.csv"
            # print(files)

            dataset = CSVDataset(
                files, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=0
            )
            # orig = CSVDataset('C:\\Users/mehak/Desktop/testganAggOrig.csv', int(args.seq_len*500),1356100,args.seq_len)
            maskDataset = CSVDataset(
                maskFiles, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=1
            )

            loader = DataLoader(
                dataset, batch_size=1, num_workers=0, shuffle=False
            )  # number of times getitem is called in one iteration
            # origLoader = DataLoader(orig,batch_size=1,num_workers=0, shuffle=False)
            maskLoader = DataLoader(
                maskDataset, batch_size=1, num_workers=0, shuffle=False
            )

            loss = {}

            # for every batch
            for batch_idx, allData in enumerate(zip(loader, maskLoader)):
                # bmi_norm=dataset.bmi_norm
                # print('batch: {}'.format(batch_idx))
                data, mask = allData
                data = data[0]
                data = data[:, :, :, 1:]

                decay = mask[:, :, :, 6]
                rdecay = mask[:, :, :, 7]
                bmi = mask[:, :, :, 5]
                mask = mask[:, :, :, 4]

                #             print(data.shape)
                bmi = bmi.unsqueeze(3)
                #             print(bmi.shape)
                #             print(bmi[0,0,:,0])

                data = torch.cat((data, bmi), dim=3)
                #             print(data.shape)
                #             print(data[0,0,:,228])

                data = data.squeeze()
                mask = mask.squeeze()
                decay = decay.squeeze()
                rdecay = rdecay.squeeze()
                bmi = bmi.squeeze()

                # values to be predicted
                y = data.clone().detach()
                #                 print(y.shape)
                #                 print(data.shape)
                testMask = mask.clone().detach()
                #             sex=y[:,:,608]
                #             age=y[:,:,607]
                # print(sex.shape,age.shape)
                y = y[:, :, 811]

                # ------------remove last 5 timestamps------------------
                # print(data[0:10,8:,653])
                for i in range(data.shape[0]):
                    # if(data[i,])
                    j = 40
                    if predWin == 8:
                        k = 32
                    elif predWin == 7:
                        k = 28
                    elif predWin == 6:
                        k = 24
                    elif predWin == 5:
                        k = 20

                    data[i, j - k : j, :] = 0
                    mask[i, j - k : j] = 0
                    y[i, 0 : j - k] = 0
                    #                 age[i,0:j-k]=0
                    #                 sex[i,0:j-k]=0
                    # print(sex.shape,age.shape)
                    # yOrig[i,0:j-k]=0
                    testMask[i, 0 : j - k] = 0

                ret = run_on_batch(
                    model, data, mask, decay, rdecay, args, optimizer=None
                )  # ,bmi_norm)
                # print("Input",data.shape)
                # print(data[0,:,316])
                # print("Reverse",ret['imputations'][0,:])
                # print("ForwardOnly",ret_f['imputations'][0,:])
                # print("Original",y.shape)
                # print(y[0,:])
                # print("Mask",testMask.shape)
                # print(testMask[0,:])
                RLoss = RLoss + ret["loss"]
                #                 testMask=testMask.cuda()
                #                 y=y.cuda()
                outputBMI = ret["imputations"] * testMask
                mseLoss = mseLoss + (torch.sum(torch.abs(outputBMI - y))) / (
                    torch.sum(testMask) + 1e-5
                )

                # print("RMSELoss Revrese: ",mseLoss)
                # print("RMSELoss Forward: ",mseLossF)
                outBmi, inBmi = plotBmi(outputBMI, y, testMask)
                oBmi.extend(outBmi)

                iBmi.extend(inBmi)

                # T.cuda.empty_cache()
                # paramsE=list(model['e'].parameters())
                # paramsG=list(model['g'].parameters())
                # print("AFTER PARAM",paramsE[0][20],paramsG[8][0][0])
            TBatches = TBatches + batch_idx + 1
        RLoss = RLoss / TBatches
        mseLoss = mseLoss / TBatches

    # print("===================================")
    oBmi = np.asarray(oBmi)
    iBmi = np.asarray(iBmi)
    loss = oBmi - iBmi
    loss = np.asarray([abs(number) for number in loss])
    variance = sum([((x - mseLoss) ** 2) for x in loss]) / len(loss)
    res = variance ** 0.5
    ci = 1.96 * (res / (math.sqrt(len(loss))))

    # print("Val R Loss:",RLoss)
    print("CI", ci)
    print("MAE Loss Reverse:", mseLoss)
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
        for i in ["M", "F"]:
            files = "cond" + i + "test.csv"

            #         drugFiles = 'drug'+G+'val.csv'

            maskFiles = "mask" + i + "test.csv"
            # print(files)

            dataset = CSVDataset(
                files, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=0
            )
            # orig = CSVDataset('C:\\Users/mehak/Desktop/testganAggOrig.csv', int(args.seq_len*500),1356100,args.seq_len)
            maskDataset = CSVDataset(
                maskFiles, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=1
            )

            loader = DataLoader(
                dataset, batch_size=1, num_workers=0, shuffle=False
            )  # number of times getitem is called in one iteration
            # origLoader = DataLoader(orig,batch_size=1,num_workers=0, shuffle=False)
            maskLoader = DataLoader(
                maskDataset, batch_size=1, num_workers=0, shuffle=False
            )

            loss = {}

            # for every batch
            for batch_idx, allData in enumerate(zip(loader, maskLoader)):
                # bmi_norm=dataset.bmi_norm
                # print('batch: {}'.format(batch_idx))
                data, mask = allData
                data = data[0]
                data = data[:, :, :, 1:]

                decay = mask[:, :, :, 6]
                rdecay = mask[:, :, :, 7]
                bmi = mask[:, :, :, 5]
                mask = mask[:, :, :, 4]

                #             print(data.shape)
                bmi = bmi.unsqueeze(3)
                #             print(bmi.shape)
                #             print(bmi[0,0,:,0])

                data = torch.cat((data, bmi), dim=3)
                #             print(data.shape)
                #             print(data[0,0,:,228])

                data = data.squeeze()
                mask = mask.squeeze()
                decay = decay.squeeze()
                rdecay = rdecay.squeeze()
                bmi = bmi.squeeze()

                # values to be predicted
                y = data.clone().detach()
                #                 print(y.shape)
                #                 print(data.shape)
                testMask = mask.clone().detach()
                #             sex=y[:,:,608]
                #             age=y[:,:,607]
                # print(sex.shape,age.shape)
                y = y[:, :, 811]

                bmi = 811

                # ------------remove last 5 timestamps------------------
                # print(data[0:10,8:,653])
                for i in range(data.shape[0]):
                    # if(data[i,])
                    # mask[i,:].loc[mask[i,:].query('value == 1').sample(frac=.1).index,'value'] = 0
                    idxs = torch.nonzero(mask[i, :] == 1)
                    samples = samples + list(idxs.size())[0]
                    if (missingRate == 50) & (list(idxs.size())[0] > 4):
                        idxs = random.sample(set(idxs), 5)
                        data[i, idxs[0], bmi] = 0
                        data[i, idxs[1], bmi] = 0
                        data[i, idxs[2], bmi] = 0
                        data[i, idxs[3], bmi] = 0
                        data[i, idxs[4], bmi] = 0
                        # print(mask[i,:])
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
                        # print(mask[i,:])
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
                        # print(mask[i,:])
                        mask[i, idxs[0]] = 0
                        mask[i, idxs[1]] = 0
                        mask[i, idxs[2]] = 0
                        pids = pids + 3
                    elif (missingRate >= 20) & (list(idxs.size())[0] > 1):
                        idxs = random.sample(set(idxs), 2)
                        data[i, idxs[0], bmi] = 0
                        data[i, idxs[1], bmi] = 0
                        # print(mask[i,:])
                        mask[i, idxs[0]] = 0
                        mask[i, idxs[1]] = 0
                        pids = pids + 2
                    elif (missingRate >= 10) & (list(idxs.size())[0] > 0):
                        idxs = random.sample(set(idxs), 1)
                        data[i, idxs, bmi] = 0
                        mask[i, idxs] = 0
                        pids = pids + 1

                    # data[i,idxs,316]=0
                    # print(mask[i,:])
                    # mask[i,idxs]=0
                    # print(mask[i,:])
                    testMask[i, :] = testMask[i, :] - mask[i, :]
                    # print(testMask[i,:])
                    # print(y[i,:])
                    y[i, :] = y[i, :] * testMask[i, :]

                # print("Input Data",data.shape)
                # print("Input Mask",mask.shape)
                ret = run_on_batch(
                    model, data, mask, decay, rdecay, args, optimizer=None
                )  # ,bmi_norm)
                # print("Input",data.shape)
                # print(data[0,:,316])
                # print("Reverse",ret['imputations'][0,:])
                # print("ForwardOnly",ret_f['imputations'][0,:])
                # print("Original",y.shape)
                # print(y[0,:])
                # print("Mask",testMask.shape)
                # print(testMask[0,:])
                RLoss = RLoss + ret["loss"]

                #             testMask=testMask.cuda()
                #             y=y.cuda()
                # print("Output",ret['imputations'].shape)
                # print("Output",ret_f['imputations'].shape)
                # print("Output mask",testMask.shape)
                outputBMI = ret["imputations"] * testMask

                # print("outputBMI",outputBMI.shape)
                # print(outputBMI[0])
                # print("outputBMIF",outputBMIF.shape)
                # print(outputBMIF[0])
                mseLoss = mseLoss + (torch.sum(torch.abs(outputBMI - y))) / (
                    torch.sum(testMask) + 1e-5
                )

                # print("RMSELoss Revrese: ",mseLoss)
                # print("RMSELoss Forward: ",mseLossF)
                outBmi, inBmi = plotBmi(outputBMI, y, testMask)
                oBmi.extend(outBmi)

                iBmi.extend(inBmi)

            TBatches = TBatches + batch_idx + 1
        RLoss = RLoss / TBatches
        mseLoss = mseLoss / TBatches
        mseLossF = mseLossF / TBatches
    # print("===================================")
    oBmi = np.asarray(oBmi)
    iBmi = np.asarray(iBmi)
    loss = oBmi - iBmi
    loss = np.asarray([abs(number) for number in loss])
    variance = sum([((x - mseLoss) ** 2) for x in loss]) / len(loss)
    res = variance ** 0.5
    ci = 1.96 * (res / (math.sqrt(len(loss))))

    # print("Val R Loss:",RLoss)
    print("CI", ci)
    print("MAE Loss Reverse:", mseLoss)
    print("Total BMI values", samples)
    print("Deleted BMIs", pids)
    print("Missing%", pids / samples)
    # print("MAE Loss Forward:",mseLossF)
    # print(outputBMI)
    return oBmi, iBmi


def plotBmi(outBmi, inBmi, testMask):

    outBmi = outBmi.cpu().detach().numpy()
    inBmi = inBmi.cpu().detach().numpy()
    testMask = testMask.cpu().detach().numpy()

    # import matplotlib.pyplot as plt
    #%matplotlib inline
    # from matplotlib.ticker import MultipleLocator
    outBmi = outBmi[np.nonzero(testMask)]
    inBmi = inBmi[np.nonzero(testMask)]

    # print(outBmi)
    # print(inBmi)
    # print(outAge)
    # print(outSex)

    # print(sex.shape,age.shape)

    return outBmi, inBmi


def run_evalFull(args, model):
    model.eval()

    RLoss = 0
    TBatches = 0
    oBmi = []
    iBmi = []
    oAge = []
    oSex = []

    with T.autograd.no_grad():
        for i in ["M", "F"]:
            files = "cond" + i + "val.csv"

            #         drugFiles = 'drug'+G+'val.csv'

            maskFiles = "mask" + i + "val.csv"
            # print(files)

            dataset = CSVDataset(
                files, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=0
            )
            # orig = CSVDataset('C:\\Users/mehak/Desktop/testganAggOrig.csv', int(args.seq_len*500),1356100,args.seq_len)
            maskDataset = CSVDataset(
                maskFiles, int(args.seq_len * BATCH_SIZE), 1356100, args.seq_len, flag=1
            )

            loader = DataLoader(
                dataset, batch_size=1, num_workers=0, shuffle=False
            )  # number of times getitem is called in one iteration
            # origLoader = DataLoader(orig,batch_size=1,num_workers=0, shuffle=False)
            maskLoader = DataLoader(
                maskDataset, batch_size=1, num_workers=0, shuffle=False
            )

            loss = {}

            # for every batch
            for batch_idx, allData in enumerate(zip(loader, maskLoader)):
                # bmi_norm=dataset.bmi_norm
                # print('batch: {}'.format(batch_idx))
                data, mask = allData
                pids = data[1]
                data = data[0]
                data = data[:, :, :, 1:]

                decay = mask[:, :, :, 6]
                rdecay = mask[:, :, :, 7]
                bmi = mask[:, :, :, 5]
                mask = mask[:, :, :, 4]

                #             print(data.shape)
                bmi = bmi.unsqueeze(3)
                #             print(bmi.shape)
                #             print(bmi[0,0,:,0])

                data = torch.cat((data, bmi), dim=3)
                #             print(data.shape)
                #             print(data[0,0,:,228])

                data = data.squeeze()
                mask = mask.squeeze()
                decay = decay.squeeze()
                rdecay = rdecay.squeeze()
                bmi = bmi.squeeze()
                # print("Data:",data[0,0,:])
                # print(decay.shape)

                ret = run_on_batch(
                    model, data, mask, decay, rdecay, args, optimizer=None
                )  # ,bmi_norm)
                RLoss = RLoss + ret["loss"]

                # T.cuda.empty_cache()
                # paramsE=list(model['e'].parameters())
                # paramsG=list(model['g'].parameters())
                # print("AFTER PARAM",paramsE[0][20],paramsG[8][0][0])
            TBatches = TBatches + batch_idx + 1
        RLoss = RLoss / TBatches
    # print("===================================")
    print("Val R Loss:", RLoss)
    # print(outputBMI)
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
        early_stopping(14.155674, model, optimizer, save_path)

    # for evrey epoch
    for epoch in range(args.num_epochs):
        model.train()

        # Running Losses
        RLoss = 0
        TBatches = 0
        print("=============EPOCH=================")

        for i in ["M", "F"]:

            # Yannick
            # files = "cond" + i + "train.csv"
            files = "data/mimic/preprocess/mimicTrain.csv"

            #         drugFiles = 'drug'+G+'train.csv'

            # maskFiles = "mask" + i + "train.csv"
            maskFiles = "data/mimic/preprocess/mimicTrainMask.csv"

            # print(files)
            # print(maskFiles)
            # print("====================New File========================")
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
            # for every batch
            for batch_idx, allData in enumerate(zip(loader, maskLoader)):
                # bmi_norm=dataset.bmi_norm
                # print('batch: {}'.format(batch_idx))
                data, mask = allData
                pids = data[1]
                data = data[0]
                data = data[:, :, :, 1:]
                #             if args.female:
                #                 data=data[:,:,:,1:229]#Female = 1:229, Male = 1:334]

                #             if args.male:
                #                 data=data[:,:,:,1:611]#Female = 1:229, Male = 1:334]

                decay = mask[:, :, :, 6]
                rdecay = mask[:, :, :, 7]
                bmi = mask[:, :, :, 5]
                mask = mask[:, :, :, 4]

                #             print(data.shape)
                bmi = bmi.unsqueeze(3)
                #             print(bmi.shape)
                #             print(bmi[0,0,:,0])

                data = torch.cat((data, bmi), dim=3)
                #                 print(data.shape)
                #             print(data[0,0,:,228])

                data = data.squeeze()
                mask = mask.squeeze()
                decay = decay.squeeze()
                rdecay = rdecay.squeeze()
                bmi = bmi.squeeze()
                #                 print("Data:",data.shape)
                #                 print(decay.shape)
                #                 print(mask.shape)

                ret = run_on_batch(
                    model, data, mask, decay, rdecay, args, optimizer
                )  # ,bmi_norm)
                RLoss = RLoss + ret["loss"].item()

                # T.cuda.empty_cache()
                # paramsE=list(model['e'].parameters())
                # paramsG=list(model['g'].parameters())
                # print("AFTER PARAM",paramsE[0][20],paramsG[8][0][0])

            TBatches = TBatches + batch_idx + 1
        # print(TBatches)
        # print("File:", i, "loss_R:", "%.4f"%RLoss/(batch_idx+1), "loss_G:", "%.4f"%GLoss/(batch_idx+1), "loss_D:", "%.4f"%DLoss/(batch_idx+1))
        # print(len(encoded))
        RLoss = RLoss / TBatches

        print("EPOCH:", epoch, "loss_R:", "%.4f" % RLoss)

        trainLoss.append(RLoss)

        valid_loss = run_evalFull(args, model)

        valLoss.append(valid_loss)
        # plotBmi(outBmi , inBmi)

        # if epoch<1 or epoch >5:
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

    model = MRNN(args)

    # print("Model",model)

    if torch.cuda.is_available():
        model = model.cuda()

    if args.resume_training:
        checkpoint = T.load(save_path)
        model.load_state_dict(checkpoint["model"])
        trainLoss, valLoss = run_epoch(args, model)
        # trainLoss, valLoss = run_evalFull(args, model)
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
