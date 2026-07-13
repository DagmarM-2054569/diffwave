# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf

from glob import glob
from torch.utils.data.distributed import DistributedSampler

from diffwave.cqt import cqt_audio_samples_for_conditioning, is_complex_cqt_target


class ConditionalDataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    super().__init__()
    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    spec_filename = f'{audio_filename}.spec.npy'
    signal, sample_rate = load_audio(audio_filename)
    spectrogram = np.load(spec_filename)
    return {
        'audio': signal[0].numpy(),
        'spectrogram': spectrogram.T,
        'sample_rate': sample_rate,
    }

def load_audio(audio_filename):
    signal, sr = sf.read(audio_filename, dtype="float32")
    if signal.ndim == 2:
        signal = signal.mean(axis=1)  # stereo -> mono, als nodig
    signal = torch.from_numpy(signal).unsqueeze(0)  # [1, T]
    return signal, sr

class UnconditionalDataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    super().__init__()
    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    spec_filename = f'{audio_filename}.spec.npy'
    signal, sample_rate = load_audio(audio_filename)
    return {
        'audio': signal[0].numpy(),
        'spectrogram': None,
        'sample_rate': sample_rate,
    }


class Collator:
  def __init__(self, params):
    self.params = params

  def _drop_volume_row_if_requested(self, spectrogram):
    if getattr(self.params, 'ignore_global_volume_row', False) and spectrogram.shape[0] > 0:
      spectrogram = spectrogram.copy()
      spectrogram[0, :] = 0.0
    return spectrogram

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    for record in minibatch:
      if int(record.get('sample_rate', self.params.sample_rate)) != int(self.params.sample_rate):
        raise ValueError(f"Invalid sample rate {record['sample_rate']}; expected {self.params.sample_rate}.")

      if self.params.unconditional:
          # Filter out records that aren't long enough.
          if len(record['audio']) < self.params.audio_len:
            del record['spectrogram']
            del record['audio']
            continue

          start = random.randint(0, record['audio'].shape[-1] - self.params.audio_len)
          end = start + self.params.audio_len
          record['audio'] = record['audio'][start:end]
          record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])), mode='constant')
      else:
          if is_complex_cqt_target(self.params):
            condition_frames = int(getattr(self.params, 'cqt_condition_frames', 0) or (int(self.params.audio_len) // samples_per_frame))
            target_samples = cqt_audio_samples_for_conditioning(self.params, condition_frames)
          else:
            condition_frames = int(self.params.crop_mel_frames)
            target_samples = condition_frames * samples_per_frame

          # Filter out records that aren't long enough.
          if len(record['spectrogram']) < condition_frames:
            del record['spectrogram']
            del record['audio']
            continue

          start = random.randint(0, record['spectrogram'].shape[0] - condition_frames)
          end = start + condition_frames
          record['spectrogram'] = self._drop_volume_row_if_requested(record['spectrogram'][start:end].T)

          start *= samples_per_frame
          end = start + target_samples
          record['audio'] = record['audio'][start:end]
          record['audio'] = np.pad(record['audio'], (0, target_samples - len(record['audio'])), mode='constant')

    audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
    if self.params.unconditional:
        return {
            'audio': torch.from_numpy(audio),
            'spectrogram': None,
        }
    spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
    return {
        'audio': torch.from_numpy(audio),
        'spectrogram': torch.from_numpy(spectrogram),
    }

  # for gtzan
  def collate_gtzan(self, minibatch):
    ldata = []
    mean_audio_len = self.params.audio_len # change to fit in gpu memory
    # audio total generated time = audio_len * sample_rate
    # GTZAN statistics
    # max len audio 675808; min len audio sample 660000; mean len audio sample 662117
    # max audio sample 1; min audio sample -1; mean audio sample -0.0010 (normalized)
    # sample rate of all is 22050
    for data in minibatch:
      if data[0].shape[-1] < mean_audio_len:  # pad
        data_audio = F.pad(data[0], (0, mean_audio_len - data[0].shape[-1]), mode='constant', value=0)
      elif data[0].shape[-1] > mean_audio_len:  # crop
        start = random.randint(0, data[0].shape[-1] - mean_audio_len)
        end = start + mean_audio_len
        data_audio = data[0][:, start:end]
      else:
        data_audio = data[0]
      ldata.append(data_audio)
    audio = torch.cat(ldata, dim=0)
    return {
          'audio': audio,
          'spectrogram': None,
    }


def from_path(data_dirs, params, is_distributed=False):
  if params.unconditional:
    dataset = UnconditionalDataset(data_dirs)
  else:#with condition
    dataset = ConditionalDataset(data_dirs)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate,
      shuffle=not is_distributed,
      num_workers=4,
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)


def from_gtzan(params, is_distributed=False):
  dataset = torchaudio.datasets.GTZAN('./data', download=True)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate_gtzan,
      shuffle=not is_distributed,
      num_workers=4,
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)
