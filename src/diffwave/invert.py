
import csv
import math
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as TT
from tqdm import tqdm

from diffwave.model import DiffWave
from diffwave.params import AttrDict, params as base_params


def _checkpoint_path(model_dir):
  weights_path = os.path.join(model_dir, 'weights.pt')
  return weights_path if os.path.exists(weights_path) else model_dir


def load_model(model_dir, device):
  checkpoint = torch.load(_checkpoint_path(model_dir), map_location=device)
  params = AttrDict(base_params)
  if isinstance(checkpoint, dict) and 'params' in checkpoint:
    params.override(checkpoint['params'])

  model = DiffWave(params).to(device)
  state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
  model.load_state_dict(state_dict)
  model.eval()
  for parameter in model.parameters():
    parameter.requires_grad_(False)
  return model, params


def load_audio(filename, params, device, strict_sample_rate=False):
  audio, sample_rate = torchaudio.load(filename)
  audio = audio[0]
  audio = torch.clamp(audio, -1.0, 1.0)

  if sample_rate != params.sample_rate:
    if strict_sample_rate:
      raise ValueError(f'Invalid sample rate {sample_rate} in file {filename}; expected {params.sample_rate}.')
    audio = torchaudio.functional.resample(audio, sample_rate, params.sample_rate)
  return audio.to(device)


def preprocess_like_spectrogram(audio, params):
  mel_args = {
      'sample_rate': params.sample_rate,
      'win_length': params.hop_samples * 4,
      'hop_length': params.hop_samples,
      'n_fft': params.n_fft,
      'f_min': 20.0,
      'f_max': params.sample_rate / 2.0,
      'n_mels': params.n_mels,
      'power': 1.0,
      'normalized': True,
  }
  transform = TT.MelSpectrogram(**mel_args).to(audio.device)
  spectrogram = transform(audio)
  spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
  return torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)


def fit_frame_count(spectrogram, frames):
  if spectrogram.shape[-1] > frames:
    return spectrogram[..., :frames]
  if spectrogram.shape[-1] < frames:
    pad_frames = frames - spectrogram.shape[-1]
    return torch.nn.functional.pad(spectrogram, (0, pad_frames))
  return spectrogram


def logits_from_spec(spectrogram, eps=1e-4):
  spectrogram = torch.clamp(spectrogram, eps, 1.0 - eps)
  return torch.logit(spectrogram)


def initialize_spectrogram(args, target_audio, frames, params, device):
  if args.init_spec:
    spectrogram = torch.from_numpy(np.load(args.init_spec)).float().to(device)
    if spectrogram.ndim != 2:
      raise ValueError(f'Expected init spec to have shape [n_mels, frames], got {tuple(spectrogram.shape)}.')
    if spectrogram.shape[0] != params.n_mels:
      raise ValueError(f'Expected init spec to have {params.n_mels} channels, got {spectrogram.shape[0]}.')
    return fit_frame_count(spectrogram, frames)

  if args.init == 'target-mel':
    return fit_frame_count(preprocess_like_spectrogram(target_audio, params), frames)
  if args.init == 'zeros':
    return torch.zeros(params.n_mels, frames, device=device)
  if args.init == 'random':
    return torch.rand(params.n_mels, frames, device=device)
  raise ValueError(f'Unsupported init mode: {args.init}')


def make_batch(target_audio, spectrogram, params, crop_frames, batch_size):
  frames = spectrogram.shape[-1]
  crop_frames = min(crop_frames, frames)
  max_start = frames - crop_frames

  audio_batch = []
  spectrogram_batch = []
  for _ in range(batch_size):
    start_frame = random.randint(0, max_start) if max_start > 0 else 0
    end_frame = start_frame + crop_frames
    start_sample = start_frame * params.hop_samples
    end_sample = end_frame * params.hop_samples
    audio_batch.append(target_audio[start_sample:end_sample])
    spectrogram_batch.append(spectrogram[:, start_frame:end_frame])

  return torch.stack(audio_batch, dim=0), torch.stack(spectrogram_batch, dim=0)


def optimize_spectrogram(args):
  if args.steps <= 0:
    raise ValueError('--steps must be greater than zero.')
  if args.batch_size <= 0:
    raise ValueError('--batch_size must be greater than zero.')
  if args.log_every <= 0:
    raise ValueError('--log_every must be greater than zero.')

  if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

  device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
  model, params = load_model(args.model_dir, device)
  if params.unconditional:
    raise ValueError('This inversion needs a conditional DiffWave checkpoint, but params.unconditional is True.')

  target_audio = load_audio(args.wav_path, params, device, strict_sample_rate=args.strict_sample_rate)
  frames = args.frames or int(math.ceil(target_audio.shape[-1] / params.hop_samples))
  if frames <= 0:
    raise ValueError('--frames must be greater than zero.')
  target_samples = frames * params.hop_samples
  if target_audio.shape[-1] < target_samples:
    target_audio = torch.nn.functional.pad(target_audio, (0, target_samples - target_audio.shape[-1]))
  elif target_audio.shape[-1] > target_samples:
    target_audio = target_audio[:target_samples]

  initial_spectrogram = initialize_spectrogram(args, target_audio, frames, params, device)
  spectrogram_logits = nn.Parameter(logits_from_spec(initial_spectrogram).unsqueeze(0))
  optimizer = torch.optim.Adam([spectrogram_logits], lr=args.learning_rate)
  loss_fn = nn.L1Loss() if args.loss == 'l1' else nn.MSELoss()

  noise_level = torch.from_numpy(np.cumprod(1 - np.array(params.noise_schedule)).astype(np.float32)).to(device)
  crop_frames = args.crop_frames or min(params.crop_mel_frames, frames)
  if crop_frames <= 0:
    raise ValueError('--crop_frames must be greater than zero.')
  history = []

  iterator = tqdm(range(args.steps), desc='Optimizing spec')
  for step in iterator:
    optimizer.zero_grad(set_to_none=True)
    spectrogram = torch.sigmoid(spectrogram_logits).squeeze(0)
    audio_batch, spectrogram_batch = make_batch(target_audio, spectrogram, params, crop_frames, args.batch_size)

    t = torch.randint(0, len(params.noise_schedule), [audio_batch.shape[0]], device=device)
    noise_scale = noise_level[t].unsqueeze(1)
    noise = torch.randn_like(audio_batch)
    noisy_audio = (noise_scale ** 0.5) * audio_batch + ((1.0 - noise_scale) ** 0.5) * noise

    predicted = model(noisy_audio, t, spectrogram_batch).squeeze(1)
    denoising_loss = loss_fn(noise, predicted)
    loss = denoising_loss

    smoothness_loss = torch.tensor(0.0, device=device)
    if args.smoothness > 0:
      smoothness_loss = torch.mean(torch.abs(spectrogram[:, 1:] - spectrogram[:, :-1]))
      loss = loss + args.smoothness * smoothness_loss

    sparsity_loss = torch.tensor(0.0, device=device)
    if args.sparsity > 0:
      sparsity_loss = torch.mean(torch.abs(spectrogram))
      loss = loss + args.sparsity * sparsity_loss

    binary_loss = torch.tensor(0.0, device=device)
    if args.binary_regularization > 0:
      binary_loss = torch.mean(spectrogram * (1.0 - spectrogram))
      loss = loss + args.binary_regularization * binary_loss

    loss.backward()
    if args.max_grad_norm is not None:
      nn.utils.clip_grad_norm_([spectrogram_logits], args.max_grad_norm)
    optimizer.step()

    metrics = {
        'step': step + 1,
        'loss': float(loss.detach().cpu()),
        'denoising_loss': float(denoising_loss.detach().cpu()),
        'smoothness_loss': float(smoothness_loss.detach().cpu()),
        'sparsity_loss': float(sparsity_loss.detach().cpu()),
        'binary_loss': float(binary_loss.detach().cpu()),
    }
    history.append(metrics)
    if (step + 1) % args.log_every == 0 or step == 0 or step + 1 == args.steps:
      iterator.set_postfix(loss=f"{metrics['loss']:.5f}", denoise=f"{metrics['denoising_loss']:.5f}")

  spectrogram = torch.sigmoid(spectrogram_logits).squeeze(0).detach().cpu().numpy().astype(np.float32)
  if args.threshold is not None:
    spectrogram = (spectrogram >= args.threshold).astype(np.float32)

  output_dir = os.path.dirname(os.path.abspath(args.output))
  os.makedirs(output_dir, exist_ok=True)
  np.save(args.output, spectrogram)
  if args.loss_csv:
    loss_dir = os.path.dirname(os.path.abspath(args.loss_csv))
    os.makedirs(loss_dir, exist_ok=True)
    with open(args.loss_csv, 'w', newline='') as handle:
      writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
      writer.writeheader()
      writer.writerows(history)

  return spectrogram, history[-1], params.sample_rate


def main(args):
  spectrogram, metrics, sample_rate = optimize_spectrogram(args)
  print(f'Saved optimized spec to {args.output}')
  print(f'Shape: {spectrogram.shape}; sample_rate: {sample_rate}; final loss: {metrics["loss"]:.6f}')


if __name__ == '__main__':
  parser = ArgumentParser(description='optimize a DiffWave conditioning .spec.npy file for a target wav')
  parser.add_argument('model_dir',
      help='directory containing weights.pt, or a direct checkpoint path')
  parser.add_argument('wav_path',
      help='target wav file')
  parser.add_argument('--output', '-o', default='inverted.spec.npy',
      help='output .spec.npy path')
  parser.add_argument('--steps', type=int, default=1000,
      help='number of optimization steps')
  parser.add_argument('--learning_rate', '--learning-rate', '--lr', type=float, default=0.05,
      help='Adam learning rate for the spec logits')
  parser.add_argument('--batch_size', '--batch-size', type=int, default=1,
      help='number of random crops/noise samples per optimization step')
  parser.add_argument('--crop_frames', '--crop-frames', type=int, default=None,
      help='conditioning frames per step; defaults to params.crop_mel_frames')
  parser.add_argument('--frames', type=int, default=None,
      help='force the output spec to this many frames')
  parser.add_argument('--init', choices=['target-mel', 'random', 'zeros'], default='target-mel',
      help='initial spec before optimization')
  parser.add_argument('--init_spec', '--init-spec',
      help='optional .spec.npy file to use as initialization')
  parser.add_argument('--loss', choices=['l1', 'mse'], default='l1',
      help='denoising loss type')
  parser.add_argument('--smoothness', type=float, default=0.0,
      help='weight for temporal smoothness regularization')
  parser.add_argument('--sparsity', type=float, default=0.0,
      help='weight for L1 sparsity regularization on the spec values')
  parser.add_argument('--binary_regularization', '--binary-regularization', type=float, default=0.0,
      help='weight that nudges values toward 0 or 1')
  parser.add_argument('--threshold', type=float, default=None,
      help='optionally threshold the saved spec to binary values')
  parser.add_argument('--max_grad_norm', '--max-grad-norm', type=float, default=None,
      help='optional gradient clipping for the optimized spec')
  parser.add_argument('--strict_sample_rate', '--strict-sample-rate', action='store_true',
      help='raise instead of resampling if the wav sample rate differs from the model')
  parser.add_argument('--loss_csv', '--loss-csv',
      help='optional path for per-step optimization metrics')
  parser.add_argument('--device',
      help='override device, for example cuda, cuda:0, or cpu')
  parser.add_argument('--seed', type=int, default=None,
      help='random seed for reproducible crop/noise sampling')
  parser.add_argument('--log_every', '--log-every', type=int, default=10,
      help='progress-bar update interval')
  main(parser.parse_args())
