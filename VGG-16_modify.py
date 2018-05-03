
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.init as nninit
import torch.nn.functional as F
import numpy
import matplotlib.pyplot as plt


# In[5]:


class VGG_16_FCN(nn.Module):
    def __init__(self, front_layer, 
                 middle_layer, 
                 back_layer,
                 num_classes = 10, 
                 use_deconv = True
                 init_weight=True):
        
        super(VGG_16_FCN, self).__init__()
        
        self.class_num = num_classes
        self.front_layer = front_layer
        self.middle_layer = middle_layer
        self.back_layer = back_layer
        self.deconv = use_deconv
        
        
        self.front_features = self.make_layers(self.front_layer)
        self.mid_features = self.make_layers(self.middle_layer)
        self.back_features = self.make_layers(self.back_layer)
        
        self.relu_fcn = nn.ReLU(inplace=True)
        
        
        
        if use_deconv:
            self.upsamp_3 = nn.ConvTranspose2d(self.front_layer[-2][0],
                                              )
            self.upsamp_4 = nn.ConvTranspose2d(self.middle_layer[-2][0],
                                              )
            self.upsamp_5 = nn.ConvTranspose2d(self.back_layer[-2][0],
                                              )
            
        else: 
            self.upsamp_3 = nn.UpsamplingBilinear2d(self.front_layer[-2][0],
                                              )
            self.upsamp_4 = nn.UpsamplingBilinear2d(self.middle_layer[-2][0],
                                              )
            self.upsamp_5 = nn.UpsamplingBilinear2d(self.back_layer[-2][0],
                                              )
        
        if init_weights:
            self._initialize_weights()
        
        
        
        
    def _initialize_weights(self):
        '''    
        # may try xavier, need modifying
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.normal_(0)
                
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.normal_(0)
                
                
                
                
                
                
                
    def forward(self,x):
        h = x
        h = self.front_features(x)
        pool4 = h # 1/8

        h = self.mid_features(x)
        pool5 = h # 1/16

        h = self.back_features(x)
        pool6 = h # 1/32

        
        
        
    def make_layers(self, cfg, batch_norm=True):

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

        5/3/2018

        '''

        layers = []
        in_channels = 3
        for v in cfg:

            if v[0] == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

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


cfg = {
    'VGG16': [(64,-1), (64,), ('M',), (128,), (128,), ('M',), (256,), (256,), (256,), (256,), ('M',), (512,), (512,), (512,), (512,),
              ('M',), (512,), (512,), (512,), (512,), ('M',)],
    
    'VGG16_Front': [(64,-1), (64,), ('M',), (128,), (128,), ('M',), (256,), (256,), (256,), (256,), ('M',)],
    # after each 'M', size: 1/2, 1/4, 1/8. output pool3
    'VGG16_Middle': [('I',256),(512,), (512,), (512,), (512,),('M',)],
    # size: 1/16 output pool4
    'VGG16_Back': [('I',512),(512,), (512,), (512,), (512,), ('M',)]
    # size: 1/32 output pool5
    }
'''
# comment this while using  
test_output = make_layers(cfg['VGG16_Middle'], batch_norm=True)
print(test_output)
'''





# In[15]:


a = (4,2)
a[0]


# In[18]:


input_i = torch.FloatTensor([1,4,3,5,11,7,3,4,5]).view(1, 1, 3, 3)
print(input_i)
m = nn.UpsamplingBilinear2d(scale_factor=2)
m(input_i)

