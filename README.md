# Description

This is project Semantic Segmentation by a Deep Encoder-Decoder Fully Convolutional Neural Network, which is developed by team CRY FOR LEARNING composed of Chen Du, Rui Cao and Yuhan Long.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Hardware
The training part of SegNet needs to be run on GPU with 12G memory ( 1080 Ti).


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
  
2. If you want to try pretrained network, please download the model from XXXXXXXX (the model is beyound the size limitation of Github).    Then put it into the folder named ```Models```.

3. If you want to run the training_v6_fixedweight.ipynb, please download the CITYCSAPES dataset, and run ```Data_saving.ipynb``` to        modify the dataset as needed. Then put the modified version under the same direcotory as ```training_v6_fixedweight.ipynb```. The new          dataset should be arranged as follows:
   ```shell
   ./Cityscape_modified/
                        train/ --Store training images.
                        valid/ --Store validating images.
                        test/  --Store test images.
   ```

### Code organizaiton
```shell

  Folder: history_files                 --Containing the files of previous versions.
  
  Folder: test                          --test images for Demo.ipynb is included.
  
  Folder: Models                        --Please put all the downloaded models under this folder.

  File:   Data_saving.ipynb             --Modifying the Cityscapes dataset as needed. More details are written in the file.
  
  File:   data_exam.ipynb               --Check whether the dataset is correctly modified.
  
  File:   SegNet.py                     --Containing the segnet network.
  
  File:   DataLoader.py                 --Dataloader used for training.
  
  File:   training_v6_fixedweight.ipynb --File for training a segnet with weighted loss function.
  
  File:   Demo.ipynb                    --A demo to test SegNet.
```
### Training

Before going through the ```training_v6_fixedweight.ipynb``` file, make sure the directories are correctly set. And please remember to modify the parameter ```root``` as :
```shell
./Cityscape_modified/
```

### Try our model

Download models from the link we provided in the "Installation" session and make sure to put the models under folder ```Models```. Before running, follow the "Code organization" session to make sure the directories are correctly set.

Open ```Demo.ipynb```, make sure the parameter "load_file_name" is correctly set. For example, we are using model named "checkpoint_with_epoch_00007_fixedweight_Adam.pth.tar". Then it should be set as:

```shell
load_file_name = 'Models/checkpoint_with_epoch_00007_fixedweight_Adam.pth.tar'
```
Then you can go through ```Demo.ipynb``` to see the result of test images we provided.

To test other images in the folder test, please follow the guide. The segmentation results from SegNet models are calculated in batch size 2. So be sure to put at least 2 test images. They are set in the parameter ```index_test```, where the number is the index shown in the name of test images. So if you want to try "test_00000_ori.png" and "test_00017_ori,png", simply set

```shell
index_test = [0, 17]
```

To test your own image, be sure to also upload a label version of your image, and correctly name these images as:

```shell
test_XXXXX_ori.png as your original image.
test_XXXXX_lbl.png as your label image.
```

You can set XXXXX as any five-digit positive integer. Remember to write it into the parameter ```index_test```.


