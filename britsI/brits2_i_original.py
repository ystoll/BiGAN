import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import argparse

# import import_ipynb
from rits2_i_original import *
from sklearn import metrics

SEQ_LEN = 36
RNN_HID_SIZE = 64


class BRITS2(nn.Module):
    def __init__(self, args):
        super(BRITS2, self).__init__()
        self.build(args)

    def build(self, args):
        self.rits_f = RITS2(args)
        self.rits_b = RITS2(args)

    def forward(self, data, mask, decay, rdecay, args):
        ret_f = self.rits_f(data, mask, decay, args, "forward")

        # print("=====================================REVERSE===================================================")
        ret_b = self.reverse(self.rits_b(data, mask, rdecay, args, "backward"))

        # print("going to merge results")
        ret = self.merge_ret(ret_f, ret_b)

        return ret_f, ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f["loss"]
        loss_b = ret_b["loss"]
        loss_c = self.get_consistency_loss(ret_f["imputations"], ret_b["imputations"])

        # print(loss_f,loss_b,loss_c)
        # print("Foward Imputation",ret_f['imputations'][0,:])
        # print("Backward Imputation",ret_b['imputations'][0,:])

        loss = loss_f + loss_b + loss_c

        imputations = (ret_f["imputations"] + ret_b["imputations"]) / 2

        ret_b["loss"] = loss
        # ret_f['predictions'] = predictions
        ret_b["imputations"] = imputations

        return ret_b

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.pow(pred_f - pred_b, 2.0).mean()
        return loss

    def reverse(self, ret):
        # print("in Reverse")
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                # print("dim <= 1")
                return tensor_
            # print("dim > 1")
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret


def run_on_batch(model, data, mask, decay, rdecay, args, optimizer):
    ret_f, ret = model(data, mask, decay, rdecay, args)
    # print("BATCH LOSS",ret['loss'])
    # print("one batch done")

    if optimizer is not None:
        # print("OPTIMIZE")
        optimizer.zero_grad()
        ret["loss"].backward()
        optimizer.step()

    return ret_f, ret
