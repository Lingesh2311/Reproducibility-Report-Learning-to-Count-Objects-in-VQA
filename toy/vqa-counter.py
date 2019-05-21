#!/usr/bin/env python
# coding: utf-8

# In[1]:

# -*- coding: utf-8 -*-
# Ref: https://github.com/Cyanogenoid/vqa-counting/tree/master/toy

import sys
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from torchbearer.callbacks.decorators import on_end_epoch

import torchbearer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Accelerator used: "+str(device))
with open("print.txt", "w+") as f:
    f.write(str(device) + '\n')


class VQACntrModule(nn.Module):
    def __init__(self, objects, already_sigmoided=False):
        super().__init__()
        self.objects = objects
        self.already_sigmoided = already_sigmoided
        self.f = nn.ModuleList([PiecewiseLin(16) for _ in range(16)])

    def forward(self, X):
        (attnwts, bbox) = X  # attnwts: 1024 x 10
        bbox = bbox.transpose(1, 2)  # bbox: 1024 x 10 x 4 --> 1024 x 4 x 10
        attnwts, bbox = self.top_n_attnwts(self.objects, attnwts, bbox)

        # normalise the attention weights to be in [0, 1]
        if not self.already_sigmoided:
            attnwts = torch.sigmoid(attnwts)

        # eqn 1
        A = self.batch_outer_product(attnwts)  # attnwts --> 1024 x 10 x 10

        # eqn 2
        D = 1 - self.iou(bbox, bbox)

        # eqn 3
        score = self.f[0](A) * self.f[1](D)

        # eqn 4
        dedup_score = self.f[3](A) * self.f[4](D)

        # eqn 5
        dedup_per_entry, dedup_per_row = self.deduplicate(dedup_score, attnwts)
        score = score / dedup_per_entry

        # eqn 6
        correction = self.f[0](attnwts * attnwts) / dedup_per_row
        score = score.sum(dim=2).sum(
            dim=1, keepdim=True) + correction.sum(
                dim=1, keepdim=True)

        # eqn 7
        score = (score + 1e-20).sqrt()

        # eqn 8
        one_hot = self.to_one_hot(score)

        # eqn 9
        att_conf = (self.f[5](attnwts) - 0.5).abs()

        # eqn 10
        D_conf = (self.f[6](D) - 0.5).abs()

        # eqn 11
        conf = self.f[7](att_conf.mean(dim=1, keepdim=True) + D_conf.mean(
            dim=2).mean(dim=1, keepdim=True))

        return one_hot * conf

    def deduplicate(self, dedup_score, att):
        # using outer-diffs
        att_diff = self.batch_outer_diff(att)
        score_diff = self.batch_outer_diff(dedup_score)
        sim = self.f[2](1 - score_diff).prod(dim=1) * self.f[2](1 - att_diff)
        # similarity for each row
        row_sims = sim.sum(dim=2)
        # similarity for each entry
        all_sims = self.batch_outer_product(row_sims)
        return all_sims, row_sims

    def to_one_hot(self, scores):
        """ Turn a bunch of non-negative scalar values into a one-hot encoding.
    E.g. with self.objects = 3, 0 -> [1 0 0 0], 2.75 -> [0 0 0.25 0.75].
    """
        # sanity check, I don't think this ever does anything (it certainly shouldn't)
        scores = scores.clamp(min=0, max=self.objects)
        # compute only on the support
        i = scores.long().data
        f = scores.frac()
        # target_l is the one-hot if the score is rounded down
        # target_r is the one-hot if the score is rounded up
        target_l = scores.data.new(i.size(0), self.objects + 1).fill_(0)
        target_r = scores.data.new(i.size(0), self.objects + 1).fill_(0)

        target_l.scatter_(dim=1, index=i.clamp(max=self.objects), value=1)
        target_r.scatter_(
            dim=1, index=(i + 1).clamp(max=self.objects), value=1)
        # interpolate between these with the fractional part of the score
        return (1 - f) * Variable(target_l) + f * Variable(target_r)

    def top_n_attnwts(self, n, attnwts, bbox):
        # Function creates the top n attention weights for the matrix of attention
        attnwts, idx = attnwts.topk(n, dim=1, sorted=False)
        idx = idx.unsqueeze(dim=1).expand(
            bbox.size(0), bbox.size(1), idx.size(1))
        bbox = bbox.gather(2, idx)
        return attnwts, bbox

    def outer(self, x):
        size = tuple(x.size()) + (x.size()[-1], )
        a = x.unsqueeze(dim=-1).expand(*size)
        b = x.unsqueeze(dim=-2).expand(*size)
        return a, b

    def batch_outer_product(self, x):
        # pytorch version for executing batch outer product
        return torch.bmm(x.unsqueeze(2), x.unsqueeze(1))

    def batch_outer_diff(self, x):
        a, b = self.outer(x)
        return (a - b).abs()

    def iou(self, a, b):
        # this is just the usual way to IoU from bounding boxes
        # Returns the intersection over union of two bouding regions
        inter = self.intersection(a, b)
        area_a = self.area(a).unsqueeze(2).expand_as(inter)
        area_b = self.area(b).unsqueeze(1).expand_as(inter)
        return inter / (area_a + area_b - inter + 1e-12)

    def area(self, box):
        # Returns the area of the box
        x = (box[:, 2, :] - box[:, 0, :]).clamp(min=0)
        y = (box[:, 3, :] - box[:, 1, :]).clamp(min=0)
        return x * y

    def intersection(self, a, b):
        size = (a.size(0), 2, a.size(2), b.size(2))
        min_point = torch.max(
            a[:, :2, :].unsqueeze(dim=3).expand(*size),
            b[:, :2, :].unsqueeze(dim=2).expand(*size),
        )
        max_point = torch.min(
            a[:, 2:, :].unsqueeze(dim=3).expand(*size),
            b[:, 2:, :].unsqueeze(dim=2).expand(*size),
        )
        inter = (max_point - min_point).clamp(min=0)
        area = inter[:, 0, :, :] * inter[:, 1, :, :]
        return area


class PiecewiseLin(nn.Module):
    '''
  This class is used to intialize the weights of the activation functions
  This is discussed in the reproducibility report
  '''

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.weight = nn.Parameter(torch.ones(n + 1))
        # the first weight here is always 0 with a 0 gradient
        self.weight.data[0] = 0

    def forward(self, x):
        # all weights are positive -> function is monotonically increasing
        w = self.weight.abs()
        # make weights sum to one -> f(1) = 1
        w = w / w.sum()
        w = w.view([self.n + 1] + [1] * x.dim())
        # keep cumulative sum for O(1) time complexity
        csum = w.cumsum(dim=0)
        csum = csum.expand((self.n + 1, ) + tuple(x.size()))
        w = w.expand_as(csum)

        # figure out which part of the function the input lies on
        y = self.n * x.unsqueeze(0)
        idx = Variable(y.long().data)
        f = y.frac()

        # contribution of the linear parts left of the input
        x = csum.gather(0, idx.clamp(max=self.n))
        # contribution within the linear segment the input falls into
        x = x + f * w.gather(0, (idx + 1).clamp(max=self.n))
        return x.squeeze(0)


class Net(nn.Module):
    def __init__(self, cf):
        super(Net, self).__init__()
        self.cf = cf
        self.ctr = VQACntrModule(cf, already_sigmoided=True)
        self.cfr = nn.Linear(cf + 1, cf + 1)
        init.eye_(self.cfr.weight)

    def forward(self, k):
        x = self.ctr(k)
        return self.cfr(x)


batch_size = 1024


class ToyTask(data.Dataset):
    """
    This toy task is intended to test the robustness of the approach, not so much to be "fair" to other baselines.
    """

    def __init__(self, max_objects, coord, noise):
        super().__init__()
        self.max_objects = max_objects
        self.max_proposals = self.max_objects
        self.max_coord = max(coord, 1e-6)
        self.weight_noise = noise

    def __getitem__(self, item):
        # generate random object positions
        objects = torch.rand(self.max_proposals, 2) * (1 - self.max_coord)
        # generate object boxes, to make sure that all objects are covered
        boxes = torch.cat([objects, objects + self.max_coord], dim=1)
        # determine selected objects
        count = random.randint(0, self.max_objects)
        if count > 0:
            true_boxes = boxes[:count]
            # find the iou distance to the true objects
            iou = self.iou(boxes.t().contiguous(), true_boxes.t().contiguous())
        else:
            # no true objects, so no true overlaps to compute
            iou = torch.zeros(self.max_proposals, 1)
        # determine weighting by using each box' most overlapping true box
        weights = self.weight(iou.max(dim=1)[0])

        return (weights, boxes), count  # attn, boxes, count

    def weight(self, x):
        noise = torch.rand(x.size())
        # linear interpolation between signal and noise
        x = (1 - self.weight_noise) * x + self.weight_noise * noise
        return x

    def iou(self, a, b):
        inter = self.intersection(a, b)
        area_a = self.area(a).unsqueeze(1).expand_as(inter)
        area_b = self.area(b).unsqueeze(0).expand_as(inter)
        return inter / (area_a + area_b - inter)

    def area(self, box):
        x = (box[2, :] - box[0, :]).clamp(min=0)
        y = (box[3, :] - box[1, :]).clamp(min=0)
        return x * y

    def intersection(self, a, b):
        size = (2, a.size(1), b.size(1))
        min_point = torch.max(
            a[:2, :].unsqueeze(dim=2).expand(*size),
            b[:2, :].unsqueeze(dim=1).expand(*size),
        )
        max_point = torch.min(
            a[2:, :].unsqueeze(dim=2).expand(*size),
            b[2:, :].unsqueeze(dim=1).expand(*size),
        )
        inter = (max_point - min_point).clamp(min=0)
        area = inter[0, :, :] * inter[1, :, :]
        return area

# Each epoch is run for batch_size.
# We could not find a way to work with infinite dataset using torchbearer.
# Another option is to have len = batch_size * epochs (with train_epochs=1).
# however was taking very long to train.
    def __len__(self):
        return batch_size


parameters = {
    'easy': {
        'objects': 10,
        'coord': 0.0,
        'noise': 0.0,
    },
    'hard': {
        'objects': 10,
        'coord': 0.5,
        'noise': 0.5,
    },
}

# Setting the Resolution and ranges of parameter

resolution = 100 + 1
param_ranges = {
    'coord': torch.linspace(
        0, 1, resolution),  # Initial coordinate --> l --> The x and y
    'noise': torch.linspace(
        0, 1, resolution
    ),  # Noise --> q --> Increasing q also indirectly simulates imprecise placements of bounding boxes
}

# This implements the model training

# fixing random seed for reproducibility
seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

data = []  # Stores the weights of the activation functions
acc_plot = []  # Stores the accuracies
## Epoch is the number of times a training code is executed for the entire dataset
train_epochs = 1000  # Actual value used by original paper
val_epochs = 200  # Actual value used by original paper

## These can be set to 50 and 10 to see the how the results are


def extract_weights(net):
    '''
    Function to extract the weights fromt the activation functions
    '''
    weights = [net.ctr.f[i] for i in range(8)]
    return [p.weight.data.cpu().clone() for p in weights]


### Setting the main program ###

configuration = ['easy', 'hard']  # Configuration setting - easy and hard

#@on_end_epoch
#def activation_weights(state):
#    global data
#    if state is not None:
#        data.append(extract_weights(model))

def main_wrap(objects, **kwargs):
    global acc_plot, data
    model = Net(params['objects'])  # change to different model here
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.01)
    trial = torchbearer.Trial(
        model,
        optimiser,
        loss_function,
        metrics=['accuracy'],
        verbose=0).to(device)
        #callbacks=[activation_weights],
	
    dataset = ToyTask(objects, **kwargs)  # change to different dataset here
    loader = DataLoader(dataset, batch_size=batch_size)
    trial.with_generators(train_generator=loader, test_generator=loader)
    print("\nTraining:")
#    activation_weights.on_end_epoch(None)
    trial.run(epochs=train_epochs, verbose=1)
    
    data.append(extract_weights(model))
    
    acc = []
    for i in tqdm(range(val_epochs), desc='Testing'):
        acc.append(
            trial.evaluate(verbose=0,
                           data_key=torchbearer.TEST_DATA)['test_acc'])

    acc_plot.append(np.mean(acc))

for config in configuration:
    params = parameters[config]
    print('\nStarting (config: '+config+')')
    for name, ran in param_ranges.items():
        for x in ran:
            start = time.time()
            px = dict(params)
            px[name] = x
            main_wrap(**px)
            end = time.time()
            time_used = (end - start)
            with open("print.txt", "a") as f:
                f.write(str(px) + '     time used: ' + str(time_used) + '\n')

with open("resultdata.txt", "wb") as fp:  #Pickling
    pickle.dump(data, fp)
with open("resultacc.txt", "wb") as fp:  #Pickling
    pickle.dump(acc_plot, fp)

#END
