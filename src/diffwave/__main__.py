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

from argparse import ArgumentParser
from torch.cuda import device_count
from torch.multiprocessing import spawn

from diffwave.cqt import COMPLEX_CQT_TARGET, WAVEFORM_TARGET
from diffwave.learner import train, train_distributed
from diffwave.params import params


def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


def main(args):
  overrides = {
      key: value for key, value in {
          'target_representation': args.target_representation,
          'charbonnier_eps': args.charbonnier_eps,
          'cqt_backend': args.cqt_backend,
          'cqt_condition_frames': args.cqt_condition_frames,
      }.items() if value is not None
  }
  if args.ignore_global_volume_row:
    overrides['ignore_global_volume_row'] = True
  params.override(overrides)

  replica_count = device_count()
  if replica_count > 1:
    if params.batch_size % replica_count != 0:
      raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    params.batch_size = params.batch_size // replica_count
    port = _get_free_port()
    spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
  else:
    train(args, params)


if __name__ == '__main__':
  parser = ArgumentParser(description='train (or resume training) a DiffWave model')
  parser.add_argument('model_dir',
      help='directory in which to store model checkpoints and training logs')
  parser.add_argument('data_dirs', nargs='+',
      help='space separated list of directories from which to read .wav files for training')
  parser.add_argument('--max_steps', default=None, type=int,
      help='maximum number of training steps')
  parser.add_argument('--fp16', action='store_true', default=False,
      help='use 16-bit floating point operations for training')
  parser.add_argument('--target_representation', choices=[WAVEFORM_TARGET, COMPLEX_CQT_TARGET],
      help='diffusion target representation; defaults to params.py')
  parser.add_argument('--charbonnier_eps', default=None, type=float,
      help='epsilon for Charbonnier noise-prediction loss')
  parser.add_argument('--cqt_backend', choices=['auto', 'nnaudio', 'librosa'],
      help='CQT backend for complex_cqt target generation; auto/nnaudio require CUDA + nnAudio, librosa is explicit CPU diagnostic mode')
  parser.add_argument('--cqt_condition_frames', default=None, type=int,
      help='number of symbolic conditioning frames per CQT training crop')
  parser.add_argument('--ignore_global_volume_row', action='store_true',
      help='zero symbolic conditioning row 0 while keeping MIDI velocity pitch rows')
  main(parser.parse_args())
