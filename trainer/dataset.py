from torch.utils.data import  Dataset 
import pandas as pd
from PIL import Image

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['resized']
        image = Image.open(img_path).convert('RGB')
        label = 1 if self.dataframe.iloc[idx]['clas'] == 'PNEUMONIA' else 0
        
        if self.transform:
            image = self.transform(image)
            
        return image, label