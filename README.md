# Description

This is project Semantic Segmentation by a Deep Encoder-Decoder Fully Convolutional Neural Network, which is developed by team CRY FOR LEARNING composed of Chen Du, Rui Cao and Yuhan Long.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Hardware
The training part of SegNet needs to be run on 


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
  git clone https://github.com/FatFriedFish/SegNetForECE285.git
  ```
2. If you want to try pretrained network, please download the model from XXXXXXXX (the model is beyound the size limitation of Github).    Then put it into the folder named "Models".

### Code organizaiton

  folder: history_files     --Containing the files of previous versions.
  
  folder: test              --test images for Demo.ipynb is included.
  
  folder: Models            --Please put all the downloaded models under this polder.

  file:   Data_saving.ipynb --Modifying the Cityscapes dataset as needed. More details are written in the file.

  file:   data_exam.ipynb   --Check whether the dataset is correctly modified.

  file:   SegNet.py         --Containing the segnet network.

  file:   DataLoader.py     --Dataloader used for training.

  file:   training_v6_fixedweight.ipynb --File for training a segnet with weighted loss function.
  
  file:   Demo.ipynb        --A demo to test SegNet.

### Training

Go through the training_v6_fixedweight.ipynb file, make sure the directories are correctly set.

### Validation

Go through the Demo.ipynb file, make sure the directories are correctly set.


