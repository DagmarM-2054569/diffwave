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

import math
import struct
import zlib
from pathlib import Path

import numpy as np
import torch


WAVEFORM_TARGET = 'waveform'
COMPLEX_CQT_TARGET = 'complex_cqt'


def midi_to_hz(note):
  return 440.0 * (2.0 ** ((float(note) - 69.0) / 12.0))


def target_representation(params):
  return getattr(params, 'target_representation', WAVEFORM_TARGET)


def is_complex_cqt_target(params):
  return target_representation(params) == COMPLEX_CQT_TARGET


def target_channel_count(params):
  if is_complex_cqt_target(params):
    return 2 * int(params.cqt_n_bins)
  return 1


def expected_cqt_frames(num_samples, hop_length):
  hop_length = max(int(hop_length), 1)
  return int(math.ceil(float(num_samples) / float(hop_length)))


def cqt_audio_samples_for_conditioning(params, condition_frames=None):
  if condition_frames is None:
    condition_frames = getattr(params, 'cqt_condition_frames', None)
  if condition_frames is None:
    condition_frames = int(params.audio_len) // int(params.hop_samples)
  return int(condition_frames) * int(params.hop_samples)


def flatten_complex_cqt_target(target):
  if target.ndim != 4:
    raise ValueError(f'Expected complex CQT target [N, 2, F, T], got {tuple(target.shape)}.')
  n, two, bins, frames = target.shape
  if two != 2:
    raise ValueError(f'Expected complex CQT target channel dimension to be 2, got {two}.')
  return target.reshape(n, two * bins, frames)


def unflatten_complex_cqt_target(target, n_bins):
  if target.ndim != 3:
    raise ValueError(f'Expected flattened complex CQT target [N, 2*F, T], got {tuple(target.shape)}.')
  n, channels, frames = target.shape
  expected_channels = 2 * int(n_bins)
  if channels != expected_channels:
    raise ValueError(f'Expected {expected_channels} channels for {n_bins} CQT bins, got {channels}.')
  return target.reshape(n, 2, int(n_bins), frames)


def _fix_time_axis_np(cqt, expected_frames):
  frames = cqt.shape[-1]
  if frames > expected_frames:
    return cqt[..., :expected_frames]
  if frames < expected_frames:
    pad_width = [(0, 0)] * cqt.ndim
    pad_width[-1] = (0, expected_frames - frames)
    return np.pad(cqt, pad_width, mode='constant')
  return cqt


def _fix_time_axis_torch(cqt, expected_frames):
  frames = cqt.shape[-1]
  if frames > expected_frames:
    return cqt[..., :expected_frames]
  if frames < expected_frames:
    pad = [0, expected_frames - frames]
    return torch.nn.functional.pad(cqt, pad)
  return cqt


class ComplexCQTCodec:
  """Creates normalized complex CQT targets and inverts them back to waveform."""

  def __init__(self, params):
    self.sample_rate = int(params.sample_rate)
    self.n_bins = int(params.cqt_n_bins)
    self.bins_per_octave = int(params.cqt_bins_per_octave)
    self.hop_length = int(params.cqt_hop_length)
    self.fmin = float(getattr(params, 'cqt_fmin', None) or midi_to_hz(21))
    self.filter_scale = float(getattr(params, 'cqt_filter_scale', 1.0))
    self.norm = getattr(params, 'cqt_norm', 1)
    self.sparsity = float(getattr(params, 'cqt_sparsity', 0.01))
    self.window = getattr(params, 'cqt_window', 'hann')
    self.scale = bool(getattr(params, 'cqt_scale', True))
    self.pad_mode = getattr(params, 'cqt_pad_mode', 'constant')
    self.res_type = getattr(params, 'cqt_res_type', 'soxr_hq')
    self.value_scale = float(getattr(params, 'cqt_value_scale', 8.0))
    self.compression = float(getattr(params, 'cqt_compression', 10.0))
    self.backend = getattr(params, 'cqt_backend', 'auto')
    self._nnaudio_cqt = None

  def to_target(self, audio, expected_frames=None):
    if audio.ndim == 3 and audio.shape[1] == 1:
      audio = audio[:, 0]
    if audio.ndim != 2:
      raise ValueError(f'Expected audio [N, T] or [N, 1, T], got {tuple(audio.shape)}.')
    if expected_frames is None:
      expected_frames = expected_cqt_frames(audio.shape[-1], self.hop_length)

    if self._can_use_nnaudio(audio.device):
      return self._to_target_nnaudio(audio, expected_frames)
    return self._to_target_librosa(audio, expected_frames)

  def target_to_audio(self, target, length=None, renderer='sum_real'):
    if renderer == 'sum_real':
      return self._sum_real_audio(target, length)

    if isinstance(target, torch.Tensor):
      target_np = target.detach().cpu().numpy()
    else:
      target_np = np.asarray(target)

    squeeze_batch = False
    if target_np.ndim == 3:
      target_np = target_np[None, ...]
      squeeze_batch = True
    if target_np.ndim != 4 or target_np.shape[1] != 2:
      raise ValueError(f'Expected target [N, 2, F, T] or [2, F, T], got {target_np.shape}.')

    audio = []
    for item in target_np:
      complex_cqt = self._channels_to_complex_np(item)
      if renderer == 'diagnostic':
        y = self._diagnostic_griffinlim(complex_cqt, length)
      elif renderer == 'icqt':
        y = self._icqt(complex_cqt, length)
      else:
        raise ValueError(f'Unknown CQT renderer {renderer!r}. Use "sum_real", "icqt", or "diagnostic".')
      audio.append(np.asarray(y, dtype=np.float32))

    audio = np.stack(audio)
    if squeeze_batch:
      audio = audio[0]
    return torch.from_numpy(audio)

  def save_real_component_png(self, target, output_path, seconds=0.1, batch_index=0):
    real = self.real_component_crop_for_png(target, seconds=seconds, batch_index=batch_index)
    image = self._real_values_to_red_blue_image(real)
    _write_rgb_png(output_path, image)

  def real_component_crop_for_png(self, target, seconds=0.1, batch_index=0):
    if isinstance(target, torch.Tensor):
      if target.ndim == 3:
        target = target.unsqueeze(0)
      if target.ndim != 4 or target.shape[1] != 2:
        raise ValueError(f'Expected target [N, 2, F, T] or [2, F, T], got {tuple(target.shape)}.')
      if not 0 <= int(batch_index) < target.shape[0]:
        raise IndexError(f'batch_index {batch_index} is out of range for batch size {target.shape[0]}.')
      start, end = self._middle_time_bounds(target.shape[-1], seconds)
      cropped = target[int(batch_index):int(batch_index) + 1, :, :, start:end]
      restored = self._denormalize_channels_torch(cropped.float())
      real = restored[0, 0]
      return real.detach().cpu().numpy().astype(np.float32, copy=False)

    target_np = np.asarray(target, dtype=np.float32)
    if target_np.ndim == 3:
      target_np = target_np[None, ...]
    if target_np.ndim != 4 or target_np.shape[1] != 2:
      raise ValueError(f'Expected target [N, 2, F, T] or [2, F, T], got {target_np.shape}.')
    if not 0 <= int(batch_index) < target_np.shape[0]:
      raise IndexError(f'batch_index {batch_index} is out of range for batch size {target_np.shape[0]}.')
    start, end = self._middle_time_bounds(target_np.shape[-1], seconds)
    cropped = target_np[int(batch_index):int(batch_index) + 1, :, :, start:end]
    restored = self._denormalize_channels_np(cropped)
    real = restored[0, 0]
    return real

  def _middle_time_bounds(self, available_frames, seconds):
    frames = self._png_frame_count(seconds, available_frames)
    start = max((int(available_frames) - frames) // 2, 0)
    return start, start + frames

  def _png_frame_count(self, seconds, available_frames):
    requested = int(round(float(seconds) * float(self.sample_rate) / float(self.hop_length)))
    requested = max(requested, 1)
    return min(requested, int(available_frames))

  def _real_values_to_red_blue_image(self, real):
    real = np.asarray(real, dtype=np.float32)
    if real.ndim != 2:
      raise ValueError(f'Expected real CQT crop [F, T], got {real.shape}.')
    scale = float(np.max(np.abs(real))) if real.size else 0.0
    if scale <= 0.0:
      return np.zeros((*real.shape, 3), dtype=np.uint8)
    normalized = np.clip(real / scale, -1.0, 1.0)
    image = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    image[..., 0] = np.where(normalized < 0.0, np.rint(-normalized * 255.0), 0.0).astype(np.uint8)
    image[..., 2] = np.where(normalized > 0.0, np.rint(normalized * 255.0), 0.0).astype(np.uint8)
    return image

  def _sum_real_audio(self, target, length):
    if isinstance(target, torch.Tensor):
      squeeze_batch = False
      if target.ndim == 3:
        target = target.unsqueeze(0)
        squeeze_batch = True
      if target.ndim != 4 or target.shape[1] != 2:
        raise ValueError(f'Expected target [N, 2, F, T] or [2, F, T], got {tuple(target.shape)}.')

      restored = self._denormalize_channels_torch(target.float())
      audio = restored[:, 0].sum(dim=1)
      audio = _fix_time_axis_torch(audio, length) if length is not None else audio
      audio = self._peak_normalize_torch(audio)
      if squeeze_batch:
        audio = audio[0]
      return audio.detach().cpu()

    target_np = np.asarray(target, dtype=np.float32)
    squeeze_batch = False
    if target_np.ndim == 3:
      target_np = target_np[None, ...]
      squeeze_batch = True
    if target_np.ndim != 4 or target_np.shape[1] != 2:
      raise ValueError(f'Expected target [N, 2, F, T] or [2, F, T], got {target_np.shape}.')

    restored = self._denormalize_channels_np(target_np)
    audio = restored[:, 0].sum(axis=1)
    audio = _fix_time_axis_np(audio, length) if length is not None else audio
    peak = np.max(np.abs(audio), axis=-1, keepdims=True)
    audio = np.where(peak > 1e-8, audio / peak * 0.95, audio)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
    if squeeze_batch:
      audio = audio[0]
    return torch.from_numpy(audio)

  def _peak_normalize_torch(self, audio):
    peak = audio.abs().amax(dim=-1, keepdim=True)
    audio = torch.where(peak > 1e-8, audio / peak * 0.95, audio)
    return torch.clamp(audio, -1.0, 1.0)

  def _can_use_nnaudio(self, device):
    if self.backend == 'librosa':
      return False
    if device.type != 'cuda':
      raise RuntimeError(
          'Complex CQT target generation requires CUDA + nnAudio. Refusing to '
          'fall back to CPU librosa. Install nnAudio and run training on a CUDA '
          'device, or set cqt_backend="librosa" only for an explicit CPU diagnostic.'
      )
    try:
      import nnAudio.Spectrogram  # noqa: F401
    except ImportError as exc:
      raise RuntimeError(
          'nnAudio is not installed, and automatic CPU librosa fallback is disabled. '
          'Install the optional GPU dependency with: pip install ".[cqt-gpu]".'
      ) from exc
    return True

  def _get_nnaudio_cqt(self, device):
    if self._nnaudio_cqt is None:
      from nnAudio.Spectrogram import CQT1992v2
      kwargs = dict(
          sr=self.sample_rate,
          hop_length=self.hop_length,
          fmin=self.fmin,
          n_bins=self.n_bins,
          bins_per_octave=self.bins_per_octave,
          trainable=False,
          output_format='Complex',
          pad_mode=self.pad_mode,
      )
      try:
        self._nnaudio_cqt = CQT1992v2(**kwargs, verbose=False)
      except TypeError:
        self._nnaudio_cqt = CQT1992v2(**kwargs)
      self._nnaudio_cqt.eval()
    return self._nnaudio_cqt.to(device)

  def _to_target_nnaudio(self, audio, expected_frames):
    transform = self._get_nnaudio_cqt(audio.device)
    with torch.no_grad():
      cqt = transform(audio.float())
      if torch.is_complex(cqt):
        channels = torch.stack([cqt.real, cqt.imag], dim=1)
      elif cqt.ndim == 4 and cqt.shape[-1] == 2:
        channels = cqt.permute(0, 3, 1, 2)
      elif cqt.ndim == 4 and cqt.shape[1] == 2:
        channels = cqt
      else:
        raise RuntimeError(f'Unexpected nnAudio CQT output shape {tuple(cqt.shape)}.')
      channels = _fix_time_axis_torch(channels, expected_frames)
      return self._normalize_channels_torch(channels)

  def _to_target_librosa(self, audio, expected_frames):
    import librosa
    audio_np = audio.detach().cpu().float().numpy()
    batch = []
    for item in audio_np:
      cqt = librosa.cqt(
          item,
          sr=self.sample_rate,
          hop_length=self.hop_length,
          fmin=self.fmin,
          n_bins=self.n_bins,
          bins_per_octave=self.bins_per_octave,
          filter_scale=self.filter_scale,
          norm=self.norm,
          sparsity=self.sparsity,
          window=self.window,
          scale=self.scale,
          pad_mode=self.pad_mode,
          res_type=self.res_type,
          dtype=np.complex64,
      )
      cqt = _fix_time_axis_np(cqt, expected_frames)
      channels = self._complex_to_channels_np(cqt)
      batch.append(channels)
    target = torch.from_numpy(np.stack(batch))
    target = target.to(device=audio.device, dtype=torch.float32)
    return target

  def _normalize_channels_torch(self, channels):
    channels = channels.float() / self.value_scale
    if self.compression <= 0.0:
      return channels
    real = channels[:, 0]
    imag = channels[:, 1]
    magnitude = torch.sqrt(real * real + imag * imag)
    compressed = torch.log1p(self.compression * magnitude) / math.log1p(self.compression)
    factor = compressed / torch.clamp(magnitude, min=1e-12)
    normalized = torch.stack([real * factor, imag * factor], dim=1)
    return torch.where(magnitude.unsqueeze(1) > 0.0, normalized, torch.zeros_like(normalized))

  def _denormalize_channels_np(self, channels):
    channels = np.asarray(channels, dtype=np.float32)
    if self.compression <= 0.0:
      return channels * self.value_scale
    if channels.ndim == 3:
      channel_axis = 0
      real = channels[0]
      imag = channels[1]
    elif channels.ndim == 4:
      channel_axis = 1
      real = channels[:, 0]
      imag = channels[:, 1]
    else:
      raise ValueError(f'Expected normalized CQT channels [2, F, T] or [N, 2, F, T], got {channels.shape}.')
    magnitude = np.sqrt(real * real + imag * imag)
    restored_mag = np.expm1(magnitude * math.log1p(self.compression)) / self.compression
    factor = restored_mag / np.maximum(magnitude, 1e-12)
    restored = np.stack([real * factor, imag * factor], axis=channel_axis)
    restored = np.where(np.expand_dims(magnitude > 0.0, axis=channel_axis), restored, 0.0)
    return restored * self.value_scale

  def _denormalize_channels_torch(self, channels):
    channels = channels.float()
    if self.compression <= 0.0:
      return channels * self.value_scale
    if channels.ndim == 3:
      channel_axis = 0
      real = channels[0]
      imag = channels[1]
    elif channels.ndim == 4:
      channel_axis = 1
      real = channels[:, 0]
      imag = channels[:, 1]
    else:
      raise ValueError(f'Expected normalized CQT channels [2, F, T] or [N, 2, F, T], got {tuple(channels.shape)}.')
    magnitude = torch.sqrt(real * real + imag * imag)
    restored_mag = torch.expm1(magnitude * math.log1p(self.compression)) / self.compression
    factor = restored_mag / torch.clamp(magnitude, min=1e-12)
    restored = torch.stack([real * factor, imag * factor], dim=channel_axis)
    restored = torch.where(torch.unsqueeze(magnitude > 0.0, dim=channel_axis), restored, torch.zeros_like(restored))
    return restored * self.value_scale

  def _complex_to_channels_np(self, cqt):
    channels = np.stack([np.real(cqt), np.imag(cqt)], axis=0).astype(np.float32, copy=False)
    channels = channels / self.value_scale
    if self.compression <= 0.0:
      return channels
    real = channels[0]
    imag = channels[1]
    magnitude = np.sqrt(real * real + imag * imag)
    compressed = np.log1p(self.compression * magnitude) / math.log1p(self.compression)
    factor = compressed / np.maximum(magnitude, 1e-12)
    normalized = np.stack([real * factor, imag * factor], axis=0)
    return np.where(magnitude[None, ...] > 0.0, normalized, 0.0).astype(np.float32, copy=False)

  def _channels_to_complex_np(self, channels):
    restored = self._denormalize_channels_np(channels)
    return restored[0].astype(np.float32) + 1j * restored[1].astype(np.float32)

  def _icqt(self, complex_cqt, length):
    import librosa
    return librosa.icqt(
        complex_cqt,
        sr=self.sample_rate,
        hop_length=self.hop_length,
        fmin=self.fmin,
        bins_per_octave=self.bins_per_octave,
        filter_scale=self.filter_scale,
        norm=self.norm,
        sparsity=self.sparsity,
        window=self.window,
        scale=self.scale,
        res_type=self.res_type,
        length=length,
      )

  def _diagnostic_griffinlim(self, complex_cqt, length):
    import librosa
    magnitude = np.abs(complex_cqt)
    if length is not None:
      native_frames = 1 + int(length) // max(int(self.hop_length), 1)
      magnitude = _fix_time_axis_np(magnitude, native_frames)
    return librosa.griffinlim_cqt(
        magnitude,
        sr=self.sample_rate,
        hop_length=self.hop_length,
        fmin=self.fmin,
        bins_per_octave=self.bins_per_octave,
        filter_scale=self.filter_scale,
        norm=self.norm,
        sparsity=self.sparsity,
        window=self.window,
        scale=self.scale,
        length=length,
        n_iter=32,
      )


def _write_rgb_png(output_path, image):
  image = np.asarray(image, dtype=np.uint8)
  if image.ndim != 3 or image.shape[2] != 3:
    raise ValueError(f'Expected RGB image [H, W, 3], got {image.shape}.')
  height, width, _ = image.shape
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  def chunk(chunk_type, data):
    return (
        struct.pack('>I', len(data)) +
        chunk_type +
        data +
        struct.pack('>I', zlib.crc32(chunk_type + data) & 0xffffffff)
    )

  raw_rows = [b'\x00' + image[row].tobytes() for row in range(height)]
  png = (
      b'\x89PNG\r\n\x1a\n' +
      chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) +
      chunk(b'IDAT', zlib.compress(b''.join(raw_rows), level=9)) +
      chunk(b'IEND', b'')
  )
  output_path.write_bytes(png)
