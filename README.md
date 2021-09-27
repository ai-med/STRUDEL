# STRUDEL
This repository contains the PyTorch implementation of the paper **STRUDEL: Self-Training with Uncertainty Dependent Label Refinement across Domains**. 
If you are using this code please cite:

```
@InProceedings{groger2021,
author="Gr{\"o}ger, Fabian
and Rickmann, Anne-Marie
and Wachinger, Christian",
editor="Lian, Chunfeng
and Cao, Xiaohuan
and Rekik, Islem
and Xu, Xuanang
and Yan, Pingkun",
title="STRUDEL: Self-training with Uncertainty Dependent Label Refinement Across Domains",
booktitle="Machine Learning in Medical Imaging",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="306--316"}
```

```
usage: main.py [-h] [--mode MODE] [--data DATA] [--config CONFIG]

optional arguments:
  -h, --help       show this help message and exit
  --mode MODE      either 'train' or 'eval'
  --data DATA      either 'adni', 'challenge' or 'concat'
  --config CONFIG  path to config file
```

## Getting Started
```
pip install -r requirements.txt
```
