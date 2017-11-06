# Deep CORAL

An PyTorch implementation of 'Deep CORAL: Correlation Alignment for Deep Domain Adaptation. B Sun, K Saenko, ECCV 2016'

[中文介紹](https://ssarcandy.tw/2017/10/31/deep-coral/)

My implementation result (Task Amazon -> Webcam):

![](demo/result.jpg)

## Requirement

- Python 3
- PyTorch 0.2

## Usage

1. Unzip dataset in `dataset/office31.tar.gz`
2. Run `python3 main.py`

## Unit Test

```bash
$ cd ..
$ python -m DeepCORAL.tests.test -v
```