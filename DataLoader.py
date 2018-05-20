import numpy as np
import PIL
from PIL import Image
import torchvision.transforms as transforms



# Define a data loader

def data_loader(index, root, mode, Transform_data, Transform_lbl, ctg_index):

    def Categories(IMG_path, input_list):
    
    
        input_image = Image.open(IMG_path)
        input_list.sort()


        C = len(input_list)
        target = np.zeros((input_image.shape[0], input_image.shape[1]))

        for i in range(C):
            target[np.where(input_image == input_list[i])] = i
        target = np.array(target)
        
        return target


    image_path0 = root + mode + '/' + mode + '_'
    
    for idx in index:
        
        # Read data
        data_path = image_path0 + '{0:05d}'.format(idx) + '_ori.png'
        lbl_path  = image_path0 + '{0:05d}'.format(idx) + '_lbl.png'
        data_idx = Image.open(data_path)
        lbl_idx  = Image.open(lbl_path)


        # Calculate categories of label image
        lbl_idx  =  Categories(lbl_idx, ctg_index)
        
        # Apply Resize, ToTensor, Normalization
        data_out = Transform_data(data_idx)
        lbl_out  = Transform_lbl(lbl_idx)
        
        # Add dimension
        data_out.unsqueeze_(0)

        yield data_out, lbl_out
        

# parameters

num_image = 100  # Size of dataset
index = np.arange(num_image).tolist()
np.random.shuffle(index) # Shuffle data

root = './Cityscape_mod/' # Root path
mode = 'train' # train, test, valid
output_size = (450, 900)  # (height, width)


Transform_data = transforms.Compose([transforms.Resize(output_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
Transform_lbl  = transforms.Compose([transforms.Resize(output_size),
                                     transforms.ToTensor(),
                                    ])

# How to use it:
# data_loader(index, root, mode, Transform_data, Transform_lbl)
