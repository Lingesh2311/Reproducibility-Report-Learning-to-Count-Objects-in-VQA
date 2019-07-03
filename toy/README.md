# Deep Learning Coursework: The COMP6248 Reproducibility Challenge  

## Introduction

This is the re-implementation of [Learning to Count Objects in Natural Images for Visual Question Answering][0].

Codes are re-implemented base on [Counting component for VQA][1].

## How to train

To train the two models, run both of the following two commands:
```
python vqa-counter.py 
python vqa-baseline.py 
```
Both models were trained with both easy and hard task.

## Train logs

Logs of model weights and testing accuracy will store in `resultacc.txt`, `resultdata.txt` and `resultacc-baseline.txt` respectively.

## Plot accuracy

Run the following commands to plot the result.
```
python plot.py  
```
Alternatively, pretrained model weights and evaluation accuracy are stored in `.txt` files, you can just run `plot.py` directly without training.

## Other things you can do

To check out the difference of accuracy with different weights dimensions, run:
```
python plot_allacc.py
```
Remind if the file does not exist please run `vqa-counter.py`  with missing file parameter. For example, change line 39 to

`
self.f = nn.ModuleList([PiecewiseLin(8) for _ in range(8)])
`

Alternatively, pretrained model weights and evaluation accuracy are stored in `.txt` files, you can just run `plot_allacc.py` directly without training.

## Dependencies

This code was confirmed to run with the following environment:

- Python 3.6.3
  - torch 1.0.1
  - torchvision 0.2.1
  - torchbearer 0.3.0
  - numpy 1.14.5
  - tqdm 4.31.1
  - matplotlib 3.0.3
  
- Cuda 10.0

[0]: https://openreview.net/forum?id=B12Js_yRb
[1]: https://github.com/Cyanogenoid/vqa-counting
