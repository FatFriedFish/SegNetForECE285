
# coding: utf-8

# In[2]:


import numpy as np
import torchvision.transforms as transforms



# Define a data loader

def data_loader(index, root, mode, Transform_data, Transform_lbl, ctg_index):
    import torch
    from PIL import Image
    import PIL
    
    def Categories(input_image, input_list):
        input_list.sort()
        C = len(input_list)
        input_image = np.array(input_image)
        target = np.zeros((input_image.shape[0], input_image.shape[1]))

        for i in range(C):
            
            # Calculate Category ID
            if input_list[i] >=0 and input_list[i]<=6:
                pixel_value = 0
            elif input_list[i] >=7 and input_list[i]<=10:
                pixel_value = 1
            elif input_list[i] >=11 and input_list[i]<=16:
                pixel_value = 2
            elif input_list[i] >=17 and input_list[i]<=20:
                pixel_value = 3
            elif input_list[i] >=21 and input_list[i]<=22:
                pixel_value = 4
            elif input_list[i] ==23:
                pixel_value = 5
            elif input_list[i] >=24 and input_list[i]<=25:
                pixel_value = 6
            elif input_list[i] >= 26 or input_list[i] == -1:
                pixel_value = 7
            
            target[np.where(input_image == input_list[i])] = pixel_value
        target = np.array(target)
        
        return Image.fromarray(np.uint8(target))


    image_path0 = root + mode + '/' + mode + '_'
    
    for idx in index:
        
        # Read data
        data_path = image_path0 + '{0:05d}'.format(idx) + '_ori.png'
        lbl_path  = image_path0 + '{0:05d}'.format(idx) + '_lbl.png'
        data_idx = Image.open(data_path)
        lbl_idx  = Image.open(lbl_path)

        # Calculate categories of label image
        lbl_idx = Categories(lbl_idx, ctg_index)

        # Apply Resize, ToTensor, Normalization
        data_out = Transform_data(data_idx)
        lbl_out = Transform_lbl(lbl_idx)
        lbl_out = np.array(lbl_out)
        lbl_out = torch.from_numpy(lbl_out)
        
        # Add dimension
        data_out.unsqueeze_(0)
        lbl_out.unsqueeze_(0)
        
        yield data_out, lbl_out.long()

        



