import torch
import torchaudio

from dataclasses import dataclass

from .scan_data import DataFolderMetaInfo

@dataclass
class WavSegment:
    wav_fname: str
    sr: int
    start_sec: float
    end_sec: float

    @property
    def wav(self):
        orig_freq = torchaudio.info(self.wav_fname).sample_rate
        frame_offset = int(self.start_sec * orig_freq)
        num_frames = int(self.end_sec * orig_freq - frame_offset)
        wav, _ = torchaudio.load(self.wav_fname, 
                              frame_offset=frame_offset, 
                              num_frames=num_frames)
        if orig_freq != self.sr:
            wav = torchaudio.functional.resample(wav, 
                                               orig_freq=orig_freq, 
                                               new_freq=self.sr)
        return wav
    
@dataclass
class AnnoSegment:
    full_anno: torch.Tensor
    sr: int
    start_sec: int
    end_sec: int

    @property
    def anno(self):
        frame_offset = int(self.start_sec * self.sr)
        frame_end = int(self.end_sec*self.sr)
        return self.full_anno[frame_offset:frame_end]



class SegmentEgs:
    def __init__(self, folder_info: DataFolderMetaInfo, seg_info, preloaded_anno):
        """ seg_info - usually is a row from load_data.load_vad_df"""
        self.folder_info = folder_info
        self.seg_info = seg_info
        self.wav_keeper = WavSegment(folder_info.wav[0], 
                                     folder_info.wav[1],
                                     seg_info.start_sec,
                                     seg_info.end_sec)

        self.anno_keeper = AnnoSegment(preloaded_anno,
                                       folder_info.anno[1],
                                       seg_info.start_sec,
                                       seg_info.end_sec)
    
    @property
    def wav(self):
        return self.wav_keeper.wav

    @property
    def anno(self):
        return self.anno_keeper.anno

    @property
    def duration(self):
        return self.seg_info.end_sec - self.seg_info.start_sec