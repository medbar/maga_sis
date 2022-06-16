import pandas as pd
import torch
import numpy as np

def load_anno_tensor(anno_fname: str) ->  torch.Tensor:
    """load full annotation for file anno_fname
    :return: float tensor"""
    expert_labels_ds = pd.read_csv(anno_fname, sep=' ').fillna(0)
    return torch.from_numpy(expert_labels_ds.values).float()


def load_vad_df(vad_fname: str, thr=0.7) -> pd.DataFrame:
    """Load vad as pandas DataFrame. Conatins start_sec and end_sec"""
    df = pd.read_csv(vad_fname, names=['start_sec', 'end_sec', 'zero', 'label'], sep=';')
    return df[df['label']>thr].loc[:, ('start_sec', 'end_sec')]