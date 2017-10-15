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
$ th main.lua -finalPredictions -nEpochs 0 -loadModel /path/to/model
```

## Training

### Pipeline
![](https://github.com/jessiechouuu/adversarial-pose/blob/master/figure/pipeline.png?raw=true)

Train your model with MPII and LSP training data
```
$ th main.lua -expID exp1
```
then the models will be saved in `exp/LSP/exp1`.

You can add the options such as `-initial_Kt 0.5`, `-lambda_G 0.01`, etc.
More options can be found in `src/opts.lua`

Check out the fantastic [repo](https://github.com/anewell/pose-hg-train) of "Stacked Hourglass Network" for some advanced usage of this code.
For example, continue previous experiment with the same or different setting.

## TODO
- demo code for running on single image
- LIP annotations file

## Acknowledgements

Thanks for the open source from Alejandro Newell,
this code is heavily built on his repo [pose-hg-train](https://github.com/anewell/pose-hg-train)