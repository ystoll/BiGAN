import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import argparse

from sklearn import metrics

SEQ_LEN = 20


class TemporalDecay(nn.Module):
    def __init__(self, input_size, RNN_HID_SIZE):
        super(TemporalDecay, self).__init__()
        self.build(input_size, RNN_HID_SIZE)

    def build(self, input_size, RNN_HID_SIZE):
        self.W = Parameter(torch.Tensor(RNN_HID_SIZE, input_size))
        self.b = Parameter(torch.Tensor(RNN_HID_SIZE))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS2(nn.Module):
    def __init__(self, args):
        super(RITS2, self).__init__()
        if args.air:
            self.RNN_HID_SIZE = 10
            self.NFEATURES = 14
            self.var = 2
        if args.mimic:
            self.RNN_HID_SIZE = 10
            self.NFEATURES = 20
            self.var = 2
        if args.ehr:
            self.RNN_HID_SIZE = 400
            self.NFEATURES = 812
            self.var = 811
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.NFEATURES + 1, self.RNN_HID_SIZE)

        # self.regression = nn.Linear(RNN_HID_SIZE, 35)
        self.regression = nn.Linear(self.RNN_HID_SIZE, 1)
        self.temp_decay = TemporalDecay(input_size=self.NFEATURES, RNN_HID_SIZE=self.RNN_HID_SIZE)

        # self.out = nn.Linear(RNN_HID_SIZE, 1)

    def forward(self, values, masks, deltas, args, direct):
        # Original sequence with 24 time steps
        deltas = deltas.unsqueeze(dim=2).repeat(1, 1, self.NFEATURES)
        # print("deltas",deltas[0])

        # evals = data[direct]['evals']
        # eval_masks = data[direct]['eval_masks']

        # labels = data['labels'].view(-1, 1)
        # is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], self.RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], self.RNN_HID_SIZE)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
            values, masks, deltas = values.cuda(), masks.cuda(), deltas.cuda()

        x_loss = 0.0
        # y_loss = 0.0

        imputations = []
        if direct == "forward":

            for t in range(SEQ_LEN):
                # print("===============",t,"======================")
                x = values[:, t, :]
                # print("SEQ_LEN: ", SEQ_LEN)
                # print("masks.shape: ", masks.shape)
                # input("waiting")

                m = masks[:, t]
                d = deltas[:, t]
                # print("d",d[:,0].unsqueeze(dim=1).size())
                # print("d",d[7,:])
                # print(d[:,0])

                gamma = self.temp_decay(d)
                # print("Gamma",gamma.size())
                h = h * gamma
                # print("h",h.size())
                x_h = self.regression(h)
                # print("Regression output",x_h[0,:])

                # print("Output regression",x_h.size())
                # print("Mask",m.size())

                # x_c =  m * x +  (1 - m) * x_h
                # x[:,316] =  x[:,316]*m + (1-m)*x_h[:,0]
                x[:, self.var] = x[:, self.var] * m + (1 - m) * x_h[:, 0]
                x_c = x
                # print("Complement Vector",x_c.size())
                # print("Complement Vector",x_c[0,316])

                # x_loss += torch.sum(torch.abs(x[:,316] - x_h[:,0]) * m) / (torch.sum(m) + 1e-5)
                x_loss += torch.sum(torch.abs(x[:, self.var] - x_h[:, 0]) * m) / (torch.sum(m) + 1e-5)

                # print("X_loss",x_loss)
                m = m.unsqueeze(dim=1)

                inputs = torch.cat([x_c, m], dim=1)

                # print("Next input",inputs.size())

                h, c = self.rnn_cell(inputs, (h, c))

                # imputations.append(x_c[:,316].unsqueeze(dim = 1))
                imputations.append(x_c[:, self.var].unsqueeze(dim=1))
                # print("to be appended",m.size())
                # print("Imputations",len(imputations))
                # print("Imputations",combFactor[0].size())

        elif direct == "backward":
            # print("BACKWARD")
            for t in range(SEQ_LEN - 1, -1, -1):
                # print("===============",t,"======================")
                x = values[:, t, :]
                m = masks[:, t]
                d = deltas[:, t]
                # print("Input",x.size())

                gamma = self.temp_decay(d)
                # print("Gamma",gamma[0])
                h = h * gamma
                # print("h",h.size())
                x_h = self.regression(h)

                # print("Output regression",x_h.size())
                # print("Mask",m.size())

                # print("Regression output",x_h[0,:])

                # x_c =  m * x +  (1 - m) * x_h
                # x[:,316] =  x[:,316]*m + (1-m)*x_h[:,0]
                x[:, self.var] = x[:, self.var] * m + (1 - m) * x_h[:, 0]
                x_c = x
                # print("Complement Vector",x_c.size())
                # print("Complement Vector",x_c[0,316])

                # x_loss += torch.sum(torch.abs(x[:,316] - x_h[:,0]) * m) / (torch.sum(m) + 1e-5)
                x_loss += torch.sum(torch.abs(x[:, self.var] - x_h[:, 0]) * m) / (torch.sum(m) + 1e-5)

                # print("X_loss",x_loss)
                m = m.unsqueeze(dim=1)

                inputs = torch.cat([x_c, m], dim=1)

                # print("Next input",inputs.size())

                h, c = self.rnn_cell(inputs, (h, c))

                # imputations.append(x_c[:,316].unsqueeze(dim = 1))
                imputations.append(x_c[:, self.var].unsqueeze(dim=1))
                # print("Imputations",imputations[0].size())

        imputations = torch.cat(imputations, dim=1)
        # print("Final Imputations",imputations.size())

        return {"loss": x_loss / SEQ_LEN, "imputations": imputations}
