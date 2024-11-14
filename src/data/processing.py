from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

def get_ruls(df: pd.DataFrame) -> np.ndarray:
    """Get RUL values for train datasets."""
    RUL = np.zeros(shape=(df.shape[0],),dtype=int)
    for unit in df['unit'].unique():
        df_unit = df[df['unit']==unit]
        RUL[df_unit.index] = df_unit['cycles'].max() - df_unit['cycles']
    return RUL


def create_torch_tensors(df: pd.DataFrame, window_size: int, data_indices: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    engineid = []
    RULs = []
    RULs_pw = []
    data = []
    for unit in df['unit'].unique():
        df_unit = df[df['unit']==unit]
        engineid.append(torch.tensor(df_unit['unit'].values).unfold(0, window_size, 1)[:,0])# all labels are the same, just done for correct dimensions
        RULs.append(torch.tensor(df_unit['RUL'].values, dtype=torch.float64).unfold(0, window_size, 1)[:,-1]) # care only about the last RUL, just done for correct dimensions
        RULs_pw.append(torch.tensor(df_unit['RUL_pw'].values, dtype=torch.float64).unfold(0, window_size, 1)[:,-1]) # care only about the last RUL, just done for correct dimensions
        data.append(torch.tensor(df_unit[data_indices].values).unfold(0, window_size, 1).swapaxes(1,2))
    return torch.cat(engineid, 0), torch.cat(RULs, 0), torch.cat(data, 0), torch.cat(RULs_pw, 0)



class CMAPSS_dataset(Dataset): # custom dataset
    def __init__(self, engineid: torch.Tensor, RULs: torch.Tensor, RULspw: torch.Tensor, data: torch.Tensor, subset: str = None):
        self.engineid = engineid
        self.RULs = RULs
        self.RULspw = RULspw
        self.data = data
        self.subset = subset # FD00#
    
    def __len__(self) -> int:
        return len(self.RULs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx], self.RULs[idx], self.RULspw[idx], self.engineid[idx]
    
    def get_unit_ids(self) -> torch.Tensor:
        return self.engineid.unique()
    
    def get_unit_by_id(self, id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.where(self.engineid == id)[0]
        return self.__getitem__(idx)