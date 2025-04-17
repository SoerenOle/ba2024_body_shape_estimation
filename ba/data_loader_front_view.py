import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms


class MakeHumanDatasetFrontView(Dataset):

    def __init__(self, combinations_table_csv, img_dir, transform=None):
        self.img_parameter = pd.read_csv(combinations_table_csv, header=None)
        self.img_dir = img_dir
        self.transform = transform        
        self.transform = transforms.Compose([
            transforms.Resize((108, 192))]) # Resize image to 108, 192 pixels
              
    def __len__(self):
        return len(self.img_parameter)

    def __getitem__(self, idx):
        
        # Extract image ID and parameters from the CSV-File
        img_id = self.img_parameter.iloc[idx,0]                                         # First column in the CSV-file refers to the image ID
        parameter_vector = self.img_parameter.iloc[idx, 1:].values.astype('float32')    # The remaining 6 columns in the CSV-File refer to the parameters
        img_path = os.path.join(self.img_dir, f"frame_100_id_{img_id}.png")             # Example for image notation with img_id = "0":  "frame_100_id_0"
        img = read_image(img_path).to(torch.get_default_dtype()) / 255.0
        
        if self.transform:
            img = self.transform(img)
         
        # Convert the parameter vector to a tensor
        parameter_tensor = torch.tensor(parameter_vector, dtype=torch.get_default_dtype()) 
        
        return img, parameter_tensor