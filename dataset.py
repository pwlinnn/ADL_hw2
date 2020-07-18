import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from utils import DataSample

class QADataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __getitem__(self, index):
        sample = self.data_list[index]
        return sample.question_id, sample.context, sample.question, sample.answerable, sample.ans_start, sample.ans_end
    
    def __len__(self):
        return len(self.data_list)

def collate_fn(data):
    pass 


