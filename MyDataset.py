import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
class CSVDataset_old(Dataset):
    def __init__(self, filepath):
        print(f"reading {filepath}")

        df = pd.read_csv(
            filepath, header=0, index_col=None,
        )

        df1 = df.iloc[:, 0:320]

        df2 = (df.iloc[:, 320])

        feat = df1.iloc[:, 0:320].values
        label = df2.values

        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)

        self.x = torch.unsqueeze(self.x, 1)

        self.x = self.x.to(torch.float32)
        self.y = self.y.to(torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class CSVDataset(Dataset):
    def __init__(self,filepath):

        print(f"reading {filepath}")

        df = pd.read_csv(
           filepath, header=0, index_col=None,
        )

        df1_1 = df.iloc[:, 32:264]
        df1_2 = df.iloc[:, 300:320]
        df1 = pd.concat([df1_1, df1_2], axis=1)
        
        df2 = (df.iloc[:, 320])


        feat = df1.iloc[:, 0:252].values
        label = df2.values

        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)

        self.x = torch.unsqueeze(self.x, 1) 

        self.x = self.x.to(torch.float32)
        self.y = self.y.to(torch.int64)

    
    def __len__(self): 
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]