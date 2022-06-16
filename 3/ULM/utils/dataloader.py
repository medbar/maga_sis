import torch

from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader

def make_sorted_dataloader(ds, min_sample_len=None, **kwargs):
    assert 'collate_fn' not in kwargs and 'sampler' not in kwargs and 'shuffle' not in kwargs , f"bad kwargs {kwargs}"
    if min_sample_len is None:
        min_sample_len = getattr(ds, 'min_sample_len', 2)
    print(f'Dataloader {min_sample_len=}')
    return DataLoader(ds, 
                      collate_fn=collate_with_paddings, 
                              sampler=SortedSampler(ds, min_sample_len=min_sample_len),
                              shuffle=False, 
                              **kwargs)

def collate_with_paddings(samples):
    x = [item['feats'].T for item in samples] # btz X time X feats
    y = [item['labels'] for item in samples]
    y_cls = None
    if 'cls_labels' in samples[0]:
        y_cls = [item['cls_labels'] for item in samples]
    pad = [item['padding'] for item in samples]
    indices = [item['index'] for item in samples]
    x_batch = pad_sequence(x, batch_first=True, padding_value=0.0).transpose(-2, -1) # btz X feats X time
    #y_batch = pad_sequence(y, batch_first=True, padding_value=0)
    y_batch = torch.cat(y)
    pad_batch = pad_sequence(pad, batch_first=True, padding_value=0)
    batch = {'feats': x_batch, 
            'labels': y_batch, 
            'padding': pad_batch,
            'indices': indices}
    if y_cls is not None:
        batch['cls_labels'] = torch.LongTensor(y_cls)
    return batch


class SortedSampler(torch.utils.data.Sampler):
    def __init__(self, ds, min_sample_len):
        self.ds = ds
        self.min_sample_len = min_sample_len
        self.sizes_and_index = [(self.ds.size(i), i) for i in range(len(self.ds)) if self.ds.size(i) >= min_sample_len]
        print(f'SortedSampler remove {len(ds) - len(self.sizes_and_index)} ({len(ds)} -> {len(self.sizes_and_index)})')

    def __len__(self):
        return len(self.sizes_and_index)

    def __iter__(self):
        return iter((i for s, i in sorted(self.sizes_and_index)))