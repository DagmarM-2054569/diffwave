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
import torch
import torchaudio

from argparse import ArgumentParser

from diffwave.cqt import (
    ComplexCQTCodec,
    COMPLEX_CQT_TARGET,
    WAVEFORM_TARGET,
    is_complex_cqt_target,
    target_channel_count,
    unflatten_complex_cqt_target,
)
from diffwave.params import AttrDict, params as base_params
from diffwave.model import DiffWave


models = {}

def _params_cache_key(params):
  if params is None:
    return None
  return repr(sorted(dict(params).items()))


def _load_model(model_dir, params, device):
  # Lazy load model.
  cache_key = (model_dir, _params_cache_key(params))
  if cache_key not in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    model_params = AttrDict(base_params)
    if isinstance(checkpoint, dict) and 'params' in checkpoint:
      model_params.override(checkpoint['params'])
    if params is not None:
      model_params.override(params)
    model = DiffWave(model_params).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[cache_key] = model
  return models[cache_key]


def _prepare_spectrogram(spectrogram, params, device):
  if spectrogram is None:
    return None
  if len(spectrogram.shape) == 2:
    spectrogram = spectrogram.unsqueeze(0)
  if getattr(params, 'ignore_global_volume_row', False) and spectrogram.shape[1] > 0:
    spectrogram = spectrogram.clone()
    spectrogram[:, 0, :] = 0.0
  return spectrogram.to(device)


def predict(spectrogram=None, model_dir=None, params=None, device=torch.device('cuda'), fast_sampling=False, cqt_renderer='icqt'):
  model = _load_model(model_dir, params, device)
  with torch.no_grad():
    # Change in notation from the DiffWave paper for fast sampling.
    # DiffWave paper -> Implementation below
    # --------------------------------------
    # alpha -> talpha
    # beta -> training_noise_schedule
    # gamma -> alpha
    # eta -> beta
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)


    complex_cqt_target = is_complex_cqt_target(model.params)
    diffusion_channels = target_channel_count(model.params)

    if not model.params.unconditional:
      spectrogram = _prepare_spectrogram(spectrogram, model.params, device)
      target_length = model.params.hop_samples * spectrogram.shape[-1]
      sample = torch.randn(spectrogram.shape[0], diffusion_channels, target_length, device=device)
    else:
      target_length = model.params.audio_len
      sample = torch.randn(1, diffusion_channels, target_length, device=device)

    for n in range(len(alpha) - 1, -1, -1):
      print(f'noise level: {n}')
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      sample = c1 * (sample - c2 * model(sample, torch.tensor([T[n]], device=sample.device), spectrogram))
      if n > 0:
        noise = torch.randn_like(sample)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        sample += sigma * noise
      if not complex_cqt_target:
        sample = torch.clamp(sample, -1.0, 1.0)

    if complex_cqt_target:
      cqt_target = unflatten_complex_cqt_target(sample, model.params.cqt_n_bins)
      codec = ComplexCQTCodec(model.params)
      audio = codec.target_to_audio(cqt_target, length=target_length, renderer=cqt_renderer)
      audio = torch.clamp(audio, -1.0, 1.0)
    else:
      audio = torch.clamp(sample[:, 0], -1.0, 1.0)
  return audio, model.params.sample_rate


def main(args):
  if args.spectrogram_path:
    spectrogram = torch.from_numpy(np.load(args.spectrogram_path))
  else:
    spectrogram = None
  overrides = {key: value for key, value in {
      'target_representation': args.target_representation,
      'cqt_backend': args.cqt_backend,
      'ignore_global_volume_row': args.ignore_global_volume_row if args.ignore_global_volume_row else None,
  }.items() if value is not None}
  runtime_params = AttrDict(base_params).override(overrides) if overrides else None
  audio, sr = predict(
      spectrogram,
      model_dir=args.model_dir,
      fast_sampling=args.fast,
      params=runtime_params,
      cqt_renderer=args.cqt_renderer,
  )
  torchaudio.save(args.output, audio.cpu(), sample_rate=sr)


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('--spectrogram_path', '-s',
      help='path to a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('--output', '-o', default='output.wav',
      help='output file name')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')
  parser.add_argument('--target_representation', choices=[WAVEFORM_TARGET, COMPLEX_CQT_TARGET],
      help='override checkpoint target representation')
  parser.add_argument('--cqt_backend', choices=['auto', 'nnaudio', 'librosa'],
      help='CQT backend override for complex CQT models')
  parser.add_argument('--cqt_renderer', choices=['icqt', 'diagnostic'], default='icqt',
      help='renderer for complex CQT output; diagnostic uses magnitude-only Griffin-Lim and is not an exact inverse')
  parser.add_argument('--ignore_global_volume_row', action='store_true',
      help='zero conditioning row 0 before inference')
  main(parser.parse_args())
