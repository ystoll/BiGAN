import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import argparse

from sklearn import metrics


SEQ_LEN = 40


class TemporalDecay(nn.Module):
    def __init__(self, input_size, RNN_HID_SIZE, diag=False):
        super(TemporalDecay, self).__init__()
        self.diag = diag
        self.build(input_size, RNN_HID_SIZE)

    def build(self, input_size, RNN_HID_SIZE):
        self.W = Parameter(torch.Tensor(RNN_HID_SIZE, input_size))
        self.b = Parameter(torch.Tensor(RNN_HID_SIZE))

        if self.diag == True:
            assert input_size == RNN_HID_SIZE
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class MRNN(nn.Module):
    def __init__(self, args):
        super(MRNN, self).__init__()
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
            self.NFEATURES = 813
            self.var = 811
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.NFEATURES + 1, self.RNN_HID_SIZE)

        # self.regression = nn.Linear(RNN_HID_SIZE, 35)
        self.regression = nn.Linear(self.RNN_HID_SIZE * 2, 1)

    def get_hidden(self, values, masks, deltas, args, direct):

        # deltas=deltas.unsqueeze(dim=2).repeat(1,1,self.NFEATURES)
        #         print("deltas",deltas.shape)

        hiddens = []

        h = Variable(torch.zeros((values.size()[0], self.RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], self.RNN_HID_SIZE)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
            values, masks, deltas = values.cuda(), masks.cuda(), deltas.cuda()
        if direct == "forward":
            for t in range(SEQ_LEN):
                hiddens.append(h)

                x = values[:, t, :]
                m = masks[:, t]
                d = deltas[:, t]

                #                 print("x",x.shape)
                #                 print("m",m.shape)
                #                 print("d",d.shape)

                m = m.unsqueeze(dim=1)
                d = d.unsqueeze(dim=1)

                inputs = torch.cat([x, m, d], dim=1)
                #                 print("inputs",inputs.shape)

                h, c = self.rnn_cell(inputs, (h, c))
        elif direct == "backward":
            # print("BACKWARD")
            for t in range(SEQ_LEN - 1, -1, -1):
                hiddens.append(h)

                x = values[:, t, :]
                m = masks[:, t]
                d = deltas[:, t]

                m = m.unsqueeze(dim=1)
                d = d.unsqueeze(dim=1)

                inputs = torch.cat([x, m, d], dim=1)

                h, c = self.rnn_cell(inputs, (h, c))

        return hiddens

    def forward(self, values, masks, decay, rdecay, args):
        # Original sequence with 24 time steps

        x_loss = 0.0
        # y_loss = 0.0

        imputations = []
        hidden_forward = self.get_hidden(values, masks, decay, args, direct="forward")
        hidden_backward = self.get_hidden(
            values, masks, rdecay, args, direct="backward"
        )[::-1]

        #         print("hidden_forward",hidden_forward[0].shape)
        #         print("hidden_backward",hidden_backward[0].shape)
        deltas = decay

        for t in range(SEQ_LEN):
            # print("===============",t,"======================")
            x = values[:, t, :]
            m = masks[:, t]
            d = deltas[:, t]
            # print("d",d[:,0].unsqueeze(dim=1).size())
            # print("d",d[7,:])
            # print(d[:,0])

            hf = hidden_forward[t]
            hb = hidden_backward[t]
            h = torch.cat([hf, hb], dim=1)
            #             print("h",h.size())
            x_h = self.regression(h)
            # print("Regression output",x_h[0,:])

            #             print("Output regression",x_h.size())
            # print("Mask",m.size())

            # x_loss += torch.sum(torch.abs(x[:,316] - x_h[:,0]) * m) / (torch.sum(m) + 1e-5)
            x_loss += torch.sum(torch.abs(x[:, self.var] - x_h[:, 0]) * m) / (
                torch.sum(m) + 1e-5
            )

            # print("X_loss",x_loss)
            m = m.unsqueeze(dim=1)

            # imputations.append(x_c[:,316].unsqueeze(dim = 1))
            imputations.append(x_h[:, 0].unsqueeze(dim=1))
            # print("to be appended",m.size())
            # print("Imputations",len(imputations))
            # print("Imputations",combFactor[0].size())

        imputations = torch.cat(imputations, dim=1)
        #         print("Final Imputations",imputations.size())

        return {"loss": x_loss / SEQ_LEN, "imputations": imputations}


def run_on_batch(model, data, mask, decay, rdecay, args, optimizer):
    ret = model(data, mask, decay, rdecay, args)
    # print("BATCH LOSS",ret['loss'])
    # print("one batch done")

    if optimizer is not None:
        # print("OPTIMIZE")
        optimizer.zero_grad()
        ret["loss"].backward()
        optimizer.step()

    return ret
