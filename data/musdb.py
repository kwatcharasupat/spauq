import pytorch_lightning as pl
from torch.utils import data

class MusDB18Dataset(data.Dataset):
    def __init__(self, mode, variant) -> None:
        super().__init__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    
class MusDB18HQDataset(MusDB18Dataset):
    def __init__(self, mode) -> None:
        super().__init__(mode=mode, variant='HQ')