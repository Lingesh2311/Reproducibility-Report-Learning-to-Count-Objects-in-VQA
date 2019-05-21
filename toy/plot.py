import torch
import torch.utils.data as data
import random

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

import pickle
with open("resultdata.txt", "rb") as fp:   # Unpickling
    data = pickle.load(fp)
print(len(data))
with open("resultacc.txt", "rb") as fp:   # Unpickling
    acc_plot = pickle.load(fp)
print(len(acc_plot))
with open("resultacc-baseline.txt", "rb") as fp:   # Unpickling
    acc_plot_baseline = pickle.load(fp)
print(len(acc_plot_baseline))

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import colorcet as cc

import sys
import numpy as np
import random
import torch

cm = cc.m_rainbow
cnorm = colors.Normalize(vmin=0, vmax=1)
scalar_map = cmx.ScalarMappable(norm=cnorm, cmap=cm)

resolution = int(len(acc_plot)/4) #100
print("Detected plot resolution: "+ str(resolution))
x = np.linspace(0, 1, 17, endpoint=True)
params_list = np.linspace(0, 1, resolution)

strlist = ['q=0','l=0','q=0.5','l=0.5']
namelist = ['easy', 'hard']
color_label = ['$l$','$q$','$l$','$q$']

for stri in range(4):
    fig = plt.figure(figsize=(16, 8))
    for fi in range(8):
        plt.subplot(2,4, fi+1)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        if  fi % 4 == 0:
            plt.ylabel('$f(x)$')
        plt.xlabel('$x$')
        plt.title('$f_{}, {}$'.format(fi+1, strlist[stri]))
        for resi in range(resolution):
            col = scalar_map.to_rgba(params_list[resi])
            datai = data[(stri+1)*(resi+1) - 1][fi]
            datai = datai.abs()
            datai = datai.cumsum(dim = 0) / datai.sum()
            plt.plot(x, datai.numpy(),color=col, alpha = 0.35,linewidth=1.5)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.03, 0.6])
    cbar_ax.set_title(color_label[stri])
    ticks = np.linspace(0, 1, 5, endpoint=True)
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cm, norm=cnorm, orientation='vertical', ticks=ticks)

    fig.subplots_adjust(hspace=0.3,bottom=0.1)
    taski = 0 if stri<2 else 1
    plt.savefig('activationfuctions-{}-{}.png'.format(namelist[taski],strlist[stri]))
# plt.show()

# comment if running on server or in .py form
fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(18,4))
x = np.linspace(0, 1, resolution)
for i in range(4):
    ax[i].plot(x,acc_plot[i*resolution:(i+1)*resolution],label = 'Counting module')
    ax[i].plot(x,acc_plot_baseline[i*resolution:(i+1)*resolution],label = 'Baseline')
    ax[i].grid()
    ax[i].set_ylim(0,1)
    ax[i].set_xlim(0,1)

ax[0].set_ylabel("accuracy")
ax[0].set_title("$q = 0$")
ax[0].set_xlabel("$l$")
ax[1].set_xlabel("$q$")
ax[1].set_title("$l = 1e-6$")
ax[2].set_xlabel("$l$")
ax[2].set_title("$q = 0.5$")
ax[3].set_xlabel("$q$")
ax[3].set_title("$l = 0.5$")
fig.legend(['Counting module', 'Baseline'],
           ncol=2,
           loc='lower center',
           frameon=False,
           fontsize = 12)
plt.subplots_adjust(bottom=0.2, top=0.9)
fig.savefig('acc.png')
# plt.show()

import matplotlib.patches as patches

qlist = [0,0.5]
for i in range(len(qlist)):
    # guessing seeds for nice looking datasets
    q = qlist[i]
    torch.manual_seed(int(2 * q) + 10)
    random.seed(int(2 * q) + 16)

    cm = plt.cm.coolwarm

    params = [
        (0.05, q),
        (0.3, q),
        (0.5, q),
    ]
    n = 0
    plt.figure(figsize=(4, 6), dpi=200)
    for coord, noise in params:
        dataset = ToyTask(10, coord, noise)

        (a, b), c = next(iter(dataset))

        ax_groundtrue = plt.subplot(len(params), 2, n + 1, aspect='equal')
        ax_data = plt.subplot(len(params), 2, n + 2, aspect='equal')
        for i, (weight, box) in enumerate(zip(a, b)):
            x = box[0]
            y = box[1]
            w = box[2] - box[0]
            h = box[3] - box[1]
            config = {
                'alpha': 0.3,
                'linewidth': 0,
            }
            ax_groundtrue.add_patch(patches.Rectangle(
                (x, y), w, h,
                **config,
                color=cm(1 - float(i < c))
            ))
            ax_data.add_patch(patches.Rectangle(
                (x, y), w, h,
                **config,
                color=cm(1 - weight)
            ))
            ax_groundtrue.axes.get_xaxis().set_visible(False)
            ax_data.axes.get_xaxis().set_visible(False)
            ax_groundtrue.axes.get_yaxis().set_major_locator(plt.NullLocator())
            ax_data.axes.get_yaxis().set_visible(False)
            ax_groundtrue.set_title('Ground truth: {}'.format(c))
            ax_data.set_title('Data')
            ax_groundtrue.set_ylabel('$l = {}$'.format(coord))
        n += 2
        plt.suptitle('$q = {}$'.format(noise))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.96, bottom=0.0, hspace=0)
    plt.savefig('dataset-{}{}.png'.format(0,int(round(10 * q))))
#     plt.show()
raise SystemExit("Done.")
