import torch
import torchaudio
import numpy as np

from glob import glob
from collections import namedtuple
from .scan_data import scan_rootdir, CHANNELS
from .load_data import load_anno_tensor, load_vad_df
from .segment_data import SegmentEgs

def add_noise(clean, noise, min_amp, max_amp):
    """
    Функция, которая используется для зашумления данных
    clean и noise -- это считанные аудиозаписи с чистой речью и шумом соответственно
    min_amp -- минимальное отношение мощности шума к мощности речи
    max_amp -- максимальное отношение мощности шума к мощности речи
    """
    # степень шума случайна от min_amp до max_amp
    noise_amp = np.random.uniform(min_amp, max_amp)
    # если запись с шумом короче, чем чистая, то она дублируется нужное число раз
    noise = noise.repeat(1, clean.shape[1] // noise.shape[1] + 2)
    # так как теперь шумная запись длиннее, то выбираем случайный момент начала шумной записи
    start = np.random.randint(0, noise.shape[1] - clean.shape[1] + 1)
    noise_part = noise[:, start:start+clean.shape[1]]
    # накладываем шум
    noise_mult = clean.abs().max() / noise_part.abs().max() * noise_amp
    return (clean + noise_part * noise_mult) / (1 + noise_amp)


class ChunkAugDataSet:
    def __init__(self, rootdir, noises_dir, 
                 channels=CHANNELS, 
                 transform=torchaudio.transforms.MFCC(n_mfcc=80, melkwargs={'n_fft': 1280}),
                feats2anno_rate=1,
                chunk_size_s=2,
                chunk_hop_s=1):
        """ feats2anno_rate = feats_sr / anno_sr """
        self.rootdir = rootdir
        self.channels = channels
        self.transform = transform
        self.feats2anno_rate = feats2anno_rate
        self.finfos = scan_rootdir(rootdir, channels)
        self.noises = [file for file in glob(noises_dir, '*.wav')] 
        preloaded_annos = [load_anno_tensor(f.anno[0]) for f in self.finfos]
        self.segments = []
        Chunk = namedtuple('Chunk', ['start_sec', 'end_sec'])
        for f, p_a in zip(self.finfos, preloaded_annos):
            for _, row in load_vad_df(f.vad).iterrows():
                start = row.start_sec
                #row.end_sec
                keep_doing=True
                while keep_doing:
                    end = start + chunk_size_s 
                    if end > row.end_sec:
                        end = row.end_sec
                        start = max(0, end-chunk_size_s)
                        keep_doing=False
                    chunk = Chunk(start, end)
                    start += chunk_hop_s
                    self.segments.append(SegmentEgs(f, chunk, p_a))
        print(f"{len(self.segments)} chunks")
  
    def __len__(self):
        return len(self.segments)

    def total_sec(self):
        return sum(s.duration for s in self.segments)

    def size(self, index):
        return self.segments[index].duration
    
    def select_noise(self):
        pass
    
    def __getitem__(self, index):
        seq = self.segments[index]
        noise = 
        feats = self.transform(seq.wav).squeeze() # featsXtime
        anno = seq.anno
        corr_anno_len = round(feats.shape[-1] / self.feats2anno_rate)
        if abs(anno.shape[0] - corr_anno_len) > 2:
            print(f"WARNING: element {index}, {anno.shape[0]=} ({corr_anno_len=}), {feats.shape[-1]=}, {self.feats2anno_rate=}")
        anno = anno[:corr_anno_len]
        corr_feats_len = round(anno.shape[0] * self.feats2anno_rate)
        feats = feats[:, :corr_feats_len]
        return {'feats': feats, 
                'labels': anno, 
                'padding': torch.ones_like(anno),
                'index': index}

