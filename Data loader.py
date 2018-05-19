import numpy as np
import PIL
from PIL import Image
import torchvision.transforms as transforms



# Define a data loader

def data_loader(index, root, mode, Transform_data, Transform_lbl):
    
    image_path0 = root + mode + '/' + mode + '_'
    
    for idx in index:
        
        # Read data
        data_path = image_path0 + '{0:05d}'.format(idx) + '_ori.png'
        lbl_path  = image_path0 + '{0:05d}'.format(idx) + '_lbl.png'
        data_idx = Image.open(data_path)
        lbl_idx  = Image.open(lbl_path)
        
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
