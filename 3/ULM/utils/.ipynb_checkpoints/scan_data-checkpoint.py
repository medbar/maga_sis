import os
import dataclasses

from glob import glob
from dataclasses import dataclass
from typing import Tuple, List

ANNOTATORS = ['', '_jonas', "_sandra", "_silvan"]
ANNO_SR=25
CHANNELS = ['expert', 'novice']



@dataclass
class DataFolderMetaInfo:
    data_dir: str
    channel: str
    wav: Tuple[str, int]
    anno: Tuple[str, int]
    vad: str
    annotator: str

    @classmethod
    def build_from_dir(cls, data_dir, channel):
        wav = get_wav_fname(data_dir, channel)
        *anno, aor = get_anno_fname(data_dir, channel)
        vad = get_vad_fname(data_dir, channel)
        return cls(data_dir=data_dir,
                   channel=channel,
                   wav=wav,
                   anno=anno,
                   vad=vad,
                   annotator=aor)
    
    def is_correct(self):
        if self.anno[0] is not None and self.wav[0] is not None:
            return True
        else:
            print(f"Warning: {self.data_dir} is not correct")
            return False


def get_wav_fname(data_dir: str, channel: str) -> Tuple[str, int]:
    """ Getting wav file name for specific channel
    :channel: expert or novice
    :return: (wav file name, sample rate) """
    for sr in [16000, 48000]:
        wav = os.path.join(data_dir, f'{channel}.audio[{sr}]_clean.wav')
        if os.path.exists(wav):
            return wav, sr
        wav = os.path.join(data_dir, f'{channel}.audio[{sr}].wav')
        if os.path.exists(wav):
            return wav, sr
    raise RuntimeError(f"Didn't find wav for {data_dir} {channel}")

def get_anno_fname(data_dir: str, channel: str) -> Tuple[str, int, str]:
    """ Getting annotanion file name for specific channel
    :channel: expert or novice
    :return: (annotation file name, annotation sample rate, annotator name) """
    anno = os.path.join(data_dir, f'annotation_{channel}.txt')
    if os.path.exists(anno):
        return anno, ANNO_SR, 'gold_standard'
    print(f"Didn't find ANNO for {data_dir} {channel}.")
    return None, None, None
    
    
#     for a in ANNOTATORS:
#         anno = os.path.join(data_dir, f'engagement_{channel}{a}.annotation~')
#         if os.path.exists(anno):
#             return anno, ANNO_SR, a
    
    
    raise RuntimeError(f"Didn't find annotaitor for {data_dir} {channel}")

def get_vad_fname(data_dir: str, channel: str) -> str:
    """ Return vad file name for specific channel
    :channel: expert or novice
    :return: vad file name"""
    vad = os.path.join(data_dir, f"VAD_{channel}.annotation~")
    if os.path.exists(vad):
        return vad
    return None
    #raise RuntimeError(f"Didn't find VAD for {data_dir} {channel}")
    

def scan_rootdir(rootdir: str, channels=CHANNELS) -> List[DataFolderMetaInfo]:
    """Return List of DataFolderMetaInfo"""
    data = (DataFolderMetaInfo.build_from_dir(d, c) for d in glob(f'{rootdir}/*') 
                                                     for c in channels if os.path.isdir(d))
    return [i for i in data if i.is_correct()]