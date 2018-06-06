# Description

This is project Semantic Segmentation by a Deep Encoder-Decoder Fully Convolutional Neural Network, which is developed by team CRY FOR LEARNING composed of Chen Du, Rui Cao and Yuihan Long.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Hardware



### Packages

Packages that need installation: numpy, Pillow, matplotlib, torch, torchvision.

Use the following commands to install packages.

  ```Shell
  $ pip install numpy

  $ pip install Pillow

  $ pip install matplotlib

  $ pip install torch

  $ pip install torchvision
  ```
### Installation (sufficient for the demo)

1. Clone the SegNetForECE285 repository
  ```Shell
  # Make sure to clone with --recursive
  git clone https://github.com/FatFriedFish/SegNetForECE285.git
  ```
2. If you want to try pretrained network, please download the weight from XXXXXXXX. And put it in the same directory as demo.ipynb.

### Code organizaiton

  folder:  history_files     --Containing the files of previous versions.

  file:   Data_saving.ipynb --Modifying the Cityscapes dataset as needed. More details are written in the file.

  file:   data_exam.ipynb   --Check whether the dataset is correctly modified.

  file:   SegNet.py         --Containing the segnet network.

  file:   DataLoader.py     --Dataloader used for training.

  file:   training_v4.ipynb --File for training a segnet.




