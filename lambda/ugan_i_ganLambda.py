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
D_HID_SIZE = 10


class Combination(nn.Module):
    def __init__(self, input_size):
        super(Combination, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W1 = Parameter(torch.Tensor(5, input_size))
        self.b1 = Parameter(torch.Tensor(5))

        self.W2 = Parameter(torch.Tensor(1, 5))
        self.b2 = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.W1.size(0))
        self.W1.data.uniform_(-stdv1, stdv1)

        stdv2 = 1. / math.sqrt(self.W2.size(0))
        self.W2.data.uniform_(-stdv2, stdv2)

        if self.b1 is not None:
            self.b1.data.uniform_(-stdv1, stdv1)

        if self.b2 is not None:
            self.b2.data.uniform_(-stdv2, stdv2)

    def forward(self, d):
        #print(d.size())
        gamma = F.relu(F.linear(d, self.W1, self.b1))
        #print(gamma.size())
        gamma = F.relu(F.linear(gamma, self.W2, self.b2))
        #print(gamma.size())
        gamma = torch.exp(-gamma)
        return gamma

class TemporalDecay(nn.Module):
    def __init__(self, input_size,RNN_HID_SIZE):
        super(TemporalDecay, self).__init__()
        self.build(input_size,RNN_HID_SIZE)

    def build(self, input_size,RNN_HID_SIZE):
        self.W = Parameter(torch.Tensor(RNN_HID_SIZE, input_size))
        self.b = Parameter(torch.Tensor(RNN_HID_SIZE))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(1, D_HID_SIZE)
        self.regression1 = nn.Linear(D_HID_SIZE, 5)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.regression2 = nn.Linear(5, 1)
        self.sig = nn.Sigmoid()

    def forward(self, values, masks, args, direct):

        h = Variable(torch.zeros((values.size()[0], D_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], D_HID_SIZE)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
            values, masks = values.cuda(), masks.cuda()

        scoresSig=[]
        scoresReg=[]
        missing=[]
        if(direct=="forward"):

            for t in range(SEQ_LEN):
                #print("===============",t,"======================")
                x = values[:, t]
                x=x.unsqueeze(dim=1)
                m = masks[:, t]
                #print("Input",x.size())
                #print("Input",x[0])

                x_h = self.regression1(h)
                x_h = self.leaky(x_h)
                x_h = self.regression2(x_h)
                x_h2 = self.sig(x_h)
                #print("Discriminator output",x_h.shape)

                #print("Output regression",x_h.size())
                #print("Mask",m.size())

                m=m.unsqueeze(dim=1)

                #print("i am here")

                h, c = self.rnn_cell(x, (h, c))
                #print("i am here")

                #imputations.append(x_c[:,316].unsqueeze(dim = 1))
                scoresSig.append(x_h2[:,0].unsqueeze(dim = 1))
                scoresReg.append(x_h[:,0].unsqueeze(dim = 1))
                #print("i am here")
                missing.append(m)
                #print("i am here")
                #print("to be appended",m.size())
                #print("Imputations",len(imputations))
                #print("Scores",scores[0].size())

        elif(direct=="backward"):

            for t in range(SEQ_LEN-1,-1,-1):
                #print("===============",t,"======================")
                x = values[:, t]
                x=x.unsqueeze(dim=1)
                m = masks[:, t]
                #print("Input",x.size())
                #print("Input",x[0])

                x_h = self.regression1(h)
                x_h = self.leaky(x_h)
                x_h = self.regression2(x_h)
                x_h2 = self.sig(x_h)
                #print("Discriminator output",x_h.shape)

                #print("Output regression",x_h.size())
                #print("Mask",m.size())

                m=m.unsqueeze(dim=1)

                #print("d",d[:,0].unsqueeze(dim=1).size())

                h, c = self.rnn_cell(x, (h, c))

                #imputations.append(x_c[:,316].unsqueeze(dim = 1))
                scoresSig.append(x_h2[:,0].unsqueeze(dim = 1))
                scoresReg.append(x_h[:,0].unsqueeze(dim = 1))
                missing.append(m)
                #print("to be appended",m.size())
                #print("Imputations",len(imputations))
                #print("Scores",scores[0].size())

        scoresSig = torch.cat(scoresSig, dim = 1)
        scoresReg = torch.cat(scoresReg, dim = 1)
        missing = torch.cat(missing, dim = 1)
        #print("Scores",len(scores),scores[0].size())
        return {'scoresSig': scoresSig, 'scoresReg': scoresReg, 'missing':missing}

class UGAN(nn.Module):
    def __init__(self,args):
        super(UGAN, self).__init__()
        if args.air:
            self.RNN_HID_SIZE = 10
            self.NFEATURES = 14
            self.var=2
        if args.mimic:
            self.RNN_HID_SIZE = 10
            self.NFEATURES = 20
            self.var=2
        if args.ehr:
            self.RNN_HID_SIZE = 400
            self.NFEATURES = 812
            self.var=811
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.NFEATURES + 1, self.RNN_HID_SIZE)

        #self.regression = nn.Linear(RNN_HID_SIZE, 35)
        self.regression = nn.Linear(self.RNN_HID_SIZE, 1)
        self.temp_decay = TemporalDecay(input_size = self.NFEATURES,RNN_HID_SIZE = self.RNN_HID_SIZE)
        #self.comb_factor = Combination(input_size = 1)


        #self.out = nn.Linear(RNN_HID_SIZE, 1)

    def forward(self, values, masks, deltas, args, direct):
        # Original sequence with 24 time steps
        deltas=deltas.unsqueeze(dim=2).repeat(1,1,self.NFEATURES)
        #print("deltas",deltas[0])

        #evals = data[direct]['evals']
        #eval_masks = data[direct]['eval_masks']

        #labels = data['labels'].view(-1, 1)
        #is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], self.RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], self.RNN_HID_SIZE)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
            values, masks, deltas = values.cuda(), masks.cuda(), deltas.cuda()

        x_loss = 0.0
        #y_loss = 0.0

        imputations = []
        combFactor=[]
        missing=[]
        originals=[]
        if(direct=="forward"):

            for t in range(SEQ_LEN):
                #print("===============",t,"======================")
                x = values[:, t, :]
                m = masks[:, t]
                d = deltas[:, t]
                #print("d",d[:,0].unsqueeze(dim=1).size())
                #print("d",d[7,:])
                #print(d[:,0])

                gamma = self.temp_decay(d)
                #print("Gamma",gamma.size())
                h = h * gamma
                #print("h",h.size())
                x_h = self.regression(h)
                #print("Regression output",x_h[0,:])

                #print("Output regression",x_h.size())
                #print("Mask",m.size())

                #x_c =  m * x +  (1 - m) * x_h
                x[:,self.var] =  x[:,self.var]*m + (1-m)*x_h[:,0]
                x_c=x
                #print("Complement Vector",x_c.size())
                #print("Complement Vector",x_c[0,316])

                #comb=self.comb_factor(d[:,0].unsqueeze(dim=1))
                #print("Comb Output",comb.size())

                x_loss += torch.sum(torch.abs(x[:,self.var] - x_h[:,0]) * m) / (torch.sum(m) + 1e-5)

                #print("X_loss",x_loss)
                m=m.unsqueeze(dim=1)

                inputs = torch.cat([x_c, m], dim = 1)

                #print("Next input",inputs.size())

                h, c = self.rnn_cell(inputs, (h, c))

                #imputations.append(x_c[:,316].unsqueeze(dim = 1))
                imputations.append(x_h[:,0].unsqueeze(dim = 1))
                originals.append(x_c[:,self.var].unsqueeze(dim = 1))
                #combFactor.append(comb)
                missing.append(m)
                #print("to be appended",m.size())
                #print("Imputations",len(imputations))
                #print("Imputations",combFactor[0].size())

        elif(direct=="backward"):
            #print("BACKWARD")
            for t in range(SEQ_LEN-1,-1,-1):
                #print("===============",t,"======================")
                x = values[:, t, :]
                m = masks[:, t]
                d = deltas[:, t]
                #print("Input",x.size())

                gamma = self.temp_decay(d)
                #print("Gamma",gamma[0])
                h = h * gamma
                #print("h",h.size())
                x_h = self.regression(h)

                #print("Output regression",x_h.size())
                #print("Mask",m.size())

                #print("Regression output",x_h[0,:])

                #x_c =  m * x +  (1 - m) * x_h
                x[:,self.var] =  x[:,self.var]*m + (1-m)*x_h[:,0]
                x_c=x
                #print("Complement Vector",x_c.size())
                #print("Complement Vector",x_c[0,316])

                #comb=self.comb_factor(d[:,0].unsqueeze(dim=1))

                x_loss += torch.sum(torch.abs(x[:,self.var] - x_h[:,0]) * m) / (torch.sum(m) + 1e-5)

                #print("X_loss",x_loss)
                m=m.unsqueeze(dim=1)

                inputs = torch.cat([x_c, m], dim = 1)

                #print("Next input",inputs.size())

                h, c = self.rnn_cell(inputs, (h, c))

                #imputations.append(x_c[:,316].unsqueeze(dim = 1))
                imputations.append(x_h[:,0].unsqueeze(dim = 1))
                originals.append(x_c[:,self.var].unsqueeze(dim = 1))
                #combFactor.append(comb)
                missing.append(m)
                #print("Imputations",imputations[0].size())

        imputations = torch.cat(imputations, dim = 1)
        originals = torch.cat(originals, dim = 1)
        #combFactor = torch.cat(combFactor, dim = 1)
        missing = torch.cat(missing, dim = 1)
        #print("Final Imputations",imputations.size())
        #print("Final Combs",combFactor.size())
        #print("Final Missing",missing.size())

        return {'loss': x_loss / SEQ_LEN , 'originals':originals, 'imputations': imputations, 'missing':missing}