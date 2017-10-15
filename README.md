# Self Adversarial Training for Human Pose Estimation

Chia-Jung Chou, Jui-Ting Chien, Hwann-Tzong Chen, Self Adversarial Training for Human Pose Estimation, [arXiv:1707.02439](http://arxiv.org/abs/1603.06937)

## Prerequisites

This code is tested with [Torch7](https://github.com/torch/torch7), CUDA 8.0 and Ubuntu 16.04.

## Getting Started

Download [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de), [Leeds Sports Pose Dataset](http://sam.johnson.io/research/lsp.html) and
[Leeds Sports Pose Extended Training Dataset](http://sam.johnson.io/research/lspet.html). Place the images in `data/mpii/images` and `data/LSP/images`.

The file `data/lsp_mpii.h5` contains the annotations of MPII, LSP training data and LSP test data.

A model trained on MPII and LSP dataset is available [here](https://drive.google.com/file/d/0BzQZSyWHuFiUeTVUOVpQQzBTLVE/view?usp=sharing).

## Testing

Run the model on LSP test set and see the PCK and PCP scores. Prediction will be saved in `src/evalPose/prediction`
```
$th main.lua -finalPredictions -nEpochs 0 -loadModel /path/to/model
```

## Training

Train your model with MPII and LSP training data
```
$th main.lua -expID exp1
```
then the models will be saved in `exp/LSP/exp1`.


## TODO
![](https://github.com/jessiechouuu/adversarial-pose/blob/master/figure/pipeline.png?raw=true)
- demo code for running on single image

## Acknowledgements

This code is heavily built on [pose-hg-train](https://github.com/anewell/pose-hg-train)