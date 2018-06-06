import numpy as np
import torchvision.transforms as transforms



# Define a data loader

def data_loader(index, root, mode, Transform_data, Transform_lbl, ctg_index, batch_size):
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
    
    for i in range(int(len(index) / batch_size)):
        #for idx in index:
        data_ori = []
        for j in range(batch_size):

            # Read data
            idx = index[i * batch_size + j]
            data_path = image_path0 + '{0:05d}'.format(idx) + '_ori.png'
            lbl_path  = image_path0 + '{0:05d}'.format(idx) + '_lbl.png'
            data_idx = Image.open(data_path)
            lbl_idx  = Image.open(lbl_path)
            data_ori.append(data_idx)

            # Calculate categories of label image
            lbl_idx = Categories(lbl_idx, ctg_index)

            # Apply Resize, ToTensor, Normalization
            data_outi = Transform_data(data_idx)
            lbl_outi = Transform_lbl(lbl_idx)
            lbl_outi = np.array(lbl_outi)
            lbl_outi = torch.from_numpy(lbl_outi)

            # Add dimension
            data_outi.unsqueeze_(0)
            lbl_outi.unsqueeze_(0)
            if j == 0:
                data_out = data_outi
                lbl_out = lbl_outi
            else:
                data_out = torch.cat((data_out, data_outi), 0)
                lbl_out = torch.cat((lbl_out, lbl_outi), 0)

        yield data_out, lbl_out.long(), data_ori
