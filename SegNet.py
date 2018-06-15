
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


'''
Segnet network, default VGG16 encoder - VGG16 decoder

Encoder given to front_layer
Decoder given to back_layer

upsampling default unmax-pooling
may also try deconv, upsampling (preserved)


'''

# size of the max pooling need to be optimize when can not divided by 2
'''
two methods:
    1. fix the size of images, given to net as args, saved size in init.
        Need image preprocessing (resize) to the target size.
    
    2. Unfixed size, calculate in forward prop and save in list, may take time.
'''

class SegNet(nn.Module):
    def __init__(self, front_layer = [(64,),(64,),'M',(128,),(128,),'M',(256,),(256,),(256,), \
                                      'M', (512,),(512,),(512,),'M',(512,),(512,),(512,),'M'],
                 back_layer = [('I', 512),'UM',(512,), (512,), (512,), 'UM', (512,), (512,), (256,), 'UM',\
                              (256,), (256,),(128,), 'UM', (128,),(64,),'UM',(64,), (64,) ],
         class_num = 20,
         use_BN = True,
         upsampling = 'UM', # 'Deconv', 'USample'
         img_size = (900,600)
         ):
        super(SegNet, self).__init__()
        
        self.class_num = class_num
        self.use_BN = use_BN
        
        self.front_process = self.make_cnn_layers(front_layer, batch_norm= self.use_BN)
        self.back_process = self.make_cnn_layers(back_layer, batch_norm = self.use_BN)
        self.last_process = self.make_cnn_layers([('I',back_layer[-1][0]), (class_num,)])
        self.img_size = img_size
        self.upsampling = upsampling
        
        
        m,n = img_size
        self.size_lst = [(m,n)]
        for i in front_layer:
            if i == 'M':
                m,n = m//2, n//2
                self.size_lst.append((m,n))
                if m <= 0 or n <= 0:
                    raise Exception('Wrong Dimention or too many maxpooling!')
        
        
    def forward(self, x):
        idx_lst = []
        for layers in self.front_process:
            if isinstance(layers, nn.MaxPool2d):
                x,idx = layers(x)
                idx_lst.append(idx)
                #print(x.size())
            else:
                x = layers(x)
            
            
        #print('\nmiddle \n')
        
        if self.upsampling == 'UM':
            UM_cnt = -2
        
        for layers in self.back_process:
            
            if isinstance(layers, nn.MaxUnpool2d):
                
                x = layers(x, idx_lst.pop(-1),
                           output_size=torch.Size([x.shape[0],
                                                   x.shape[1],
                                                   self.size_lst[UM_cnt][0],
                                                   self.size_lst[UM_cnt][1]]))
                #print(x.size())
                UM_cnt -= 1
                
            else:
                x = layers(x)
            
        x = self.last_process(x)
        #print(x.size())
        return x
        

    def make_cnn_layers(self, cfg, batch_norm=True):

        '''
        The input should be a list, the elements should be tuples. 
        based on pytorch docs:
        Parameters:
            in_channels (int) – Number of channels in the input image
            out_channels (int) – Number of channels produced by the convolution
            kernel_size (int or tuple) – Size of the convolving kernel
            stride (int or tuple, optional) – Stride of the convolution. Default: 1
            padding (int or tuple, optional) – Zero-padding added to both sides of the input. Default: 1
            dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
            groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

        For convolution layers, the order is:
        (out_channels (int), kernel_size (int or tuple, optional, default = 3), 
        stride (int or tuple, optional), padding (int or tuple, optional), dilation (int or tuple, optional))
        if input is less than 0 (ie. -1) then will use default value.

        For maxpooling layer:
        'M', if need more argument, modify as needed.

        5/3/2018 C.

        '''
        
        layers = []
        in_channels = 3
        for v in cfg:

            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)]
            
            elif v == 'UM':
                layers += [nn.MaxUnpool2d(kernel_size=2, stride=2)]
                
            elif v == 'Deconv':
                raise Exception('Deconvolution network not written yet')
                
            elif v == 'Usample':
                raise Exception('Upsampling network not written yet')
            
            elif v[0] == 'I':
                in_channels = v[1]

            else:
                v_len = len(v)
                ker_size = 3
                stride_val = 1
                padding_val = 1
                dialtion_val = 1

                out_channels = v[0]
                if v_len >= 2:
                    if v[1] > 0:
                        ker_size = v[1]
                    if v_len >= 3:
                        if v[2] > 0:
                            stride_val = v[2]
                        if v_len >= 4:
                            if v[3] > 0:
                                padding_val = v[3]
                            if v_len >= 5:
                                if v[4] > 0:
                                    dialtion_val = v[4]


                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size = ker_size, stride = stride_val,
                                   padding = padding_val, dilation = dialtion_val)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v[0]
        return nn.Sequential(*layers)


