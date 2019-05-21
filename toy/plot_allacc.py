import pickle
import os.path
if os.path.isfile("resultacc.txt") == 0:
    print('resultacc.txt file does not exist')
    print('Please run vqa-counter.py with weights dimensions = 16')
    quit()
if os.path.isfile("resultacc8.txt") == 0:
    print('resultacc8.txt file does not exist')
    print('Please run vqa-counter.py with weights dimensions = 8')
    print('and rename the file resultacc.txt to resultacc8.txt')
    quit()
if os.path.isfile("resultacc32.txt") == 0:
    print('resultacc32.txt file does not exist')
    print('Please run vqa-counter.py with weights dimensions = 32')
    print('and rename the file resultacc.txt to resultacc32.txt')
    quit()


with open("resultacc.txt", "rb") as fp:   # Unpickling
    acc_plot = pickle.load(fp)
print(len(acc_plot))
with open("resultacc8.txt", "rb") as fp:   # Unpickling
    acc_plot8 = pickle.load(fp)
with open("resultacc32.txt", "rb") as fp:   # Unpickling
    acc_plot32 = pickle.load(fp)


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import colorcet as cc

import sys
import numpy as np
import random
import torch

resolution = int(len(acc_plot)/4) #100
print("Detected plot resolution: "+ str(resolution))
x = np.linspace(0, 1, 17, endpoint=True)
params_list = np.linspace(0, 1, resolution)

strlist = ['q=0','l=0','q=0.5','l=0.5']
namelist = ['easy', 'hard']
color_label = ['$l$','$q$','$l$','$q$']

fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(18,4))
x = np.linspace(0, 1, resolution)
for i in range(4):
    ax[i].plot(x,acc_plot8[i*resolution:(i+1)*resolution],label = '8',color = 'b')
    ax[i].plot(x,acc_plot[i*resolution:(i+1)*resolution],label = '16',color = 'r')
    ax[i].plot(x,acc_plot32[i*resolution:(i+1)*resolution],label = '32',color = 'y')
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
fig.legend(['8', '16','32'],
           ncol=3,
           loc='lower center',
           frameon=False,
          fontsize = 12)
plt.subplots_adjust(bottom=0.20, top=0.9)
fig.savefig('acc_8_16_32.png')
#plt.show()
