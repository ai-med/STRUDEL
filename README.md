# STRUDEL
This repository contains the PyTorch implementation of the paper **STRUDEL: Self-Training with Uncertainty Dependent Label Refinement across Domains**. 
If you are using this code please cite:

```
@article{groger2021strudel,
  title={STRUDEL: Self-Training with Uncertainty Dependent Label Refinement across Domains},
  author={Gr{\"o}ger, Fabian and Rickmann, Anne-Marie and Wachinger, Christian},
  journal={arXiv preprint arXiv:2104.11596},
  year={2021},
  url={https://arxiv.org/abs/2104.11596}
}
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
