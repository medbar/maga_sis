import torch
import numpy as np
import os

from tqdm.auto import tqdm

from zipfile import ZipFile

from .dataloader import make_sorted_dataloader


class ZippedDataSet:
    def __init__(self, zip_fname, original_ds=None, rewrite=False, min_sample_len=2):
        self.zip_fname = zip_fname
        self.min_sample_len = min_sample_len
        if os.path.exists(zip_fname) and not rewrite:
            print(f"{zip_fname=} exists! Using that archive")
        else:
            print(f"generating {zip_fname}.")
            assert original_ds is not None, 'zip_fname is not exist and original_ds is None'
            self.process(original_ds)
        
        self.zip_obj = ZipFile(zip_fname)
        with self.zip_obj.open('sizes.txt', 'r') as f:
            self.sizes = np.frombuffer(f.read(), dtype=float)
        with self.zip_obj.open('anno_dim.txt', 'r') as f:
            self.anno_dim = int.from_bytes(f.read(), 'big')
        with self.zip_obj.open('feats_dim.txt', 'r') as f:
            self.feats_dim = int.from_bytes(f.read(), 'big')
            
    def process(self, ds):
        dataloader=make_sorted_dataloader(ds, min_sample_len=self.min_sample_len, num_workers=6, batch_size=1)
        sizes=[]
        featsdim=None
        annodim=None
        with ZipFile(self.zip_fname, 'w') as f:
            for i, b in enumerate(tqdm(dataloader)):
            #tqdm.tqdm(range(len(ds))):
                x, y = b['feats'], b['labels']
                featsdim = x.shape[1] if not featsdim else featsdim
                annodim = y.shape[-1] if not annodim else annodim
                #print(x.shape)
                index = b['indices'][0]
                sizes.append(ds.size(index))
                f.writestr(f'{i}.feats', x.reshape(-1).numpy().astype(float).tobytes())
                f.writestr(f'{i}.labels', y.reshape(-1).numpy().astype(float).tobytes())
                #print(featdim, annodim)
                
            sizes=np.array(sizes, dtype=float)
            f.writestr('sizes.txt', sizes.tobytes())
            f.writestr('anno_dim.txt', (annodim).to_bytes(4, byteorder='big'))
            f.writestr('feats_dim.txt', (featsdim).to_bytes(4, byteorder='big'))
            
    def close(self):
        self.zip_obj.close()
        
    def __enter__(self):
        pass
    
    def __exit__(self, *args, **kwargs):
        self.close()
        
    def __len__(self):
        return len(self.sizes)
    
    def size(self, index):
        return self.sizes[index]
    
    def __getitem__(self, index):
        with self.zip_obj.open(f'{index}.feats') as f:
            x = torch.tensor(np.frombuffer(f.read(), dtype=float)).float()
        with self.zip_obj.open(f'{index}.labels') as f:
            anno = torch.tensor(np.frombuffer(f.read(), dtype=float)).float()
        anno = anno.reshape(-1, self.anno_dim)
        x = x.reshape(self.feats_dim, -1)
        return {'feats': x, 
                'labels': anno, 
                'padding': torch.ones_like(anno),
                'index': index}
    
    def train_test_split(self, test_ratio=0.2, seed=42):
        #from torch import default_generator, randperm
        gen = torch.Generator()
        gen.manual_seed(seed)
        indices = [i for i in torch.randperm(len(self), generator=gen).tolist() if self.size(i) >=min_sample_len]
        test_len = int(len(self)*test_ratio)
        test_indices = indices[: test_len]
        train_indices = indices[test_len: ]
        
        return Subset(self, train_indices), Subset(self, test_indices)
    
class Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    
    def size(self, index):
        return self.dataset.size(self.indices[index])
    
    def __getattr__(self, key):
        return getattr(self.dataset, key)
    
class ConcatDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        self.indices = [(i, j) for i, dataset in enumerate(datasets) for j in range(len(dataset))]

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.datasets[i][j] for i,j in map(lambda x:self.indices[x], idx)]
        i, j = self.indices[idx]
        return self.datasets[i][j]

    def __len__(self):
        return len(self.indices)
    
    def size(self, index):
        i, j = self.indices[index]
        return self.datasets[i].size(j)
    
    # def __getattr__(self, key):
    #     return getattr(self.dataset, key)