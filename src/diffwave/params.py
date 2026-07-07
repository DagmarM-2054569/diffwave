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


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self


params = AttrDict(
    # Training params
    batch_size=1,   #eigenverandering (2)   29 causes memory crash
    learning_rate=3e-5, #eigenverandering 2e-4 -> 2e-5
    max_grad_norm=None,
    charbonnier_eps=1e-3,

    # Data params
    sample_rate=22050,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    crop_mel_frames=256,  # Probably an error in paper. eigenverandering 62->160
    ignore_global_volume_row=False,

    # Diffusion target params. The symbolic .spec.npy files remain conditioning;
    # the complex CQT target is computed from the cropped WAV during training.
    target_representation='complex_cqt',
    cqt_backend='auto',
    cqt_n_bins=89,
    cqt_bins_per_octave=12,
    cqt_hop_length=1,
    cqt_fmin=27.5,
    cqt_filter_scale=1.0,
    cqt_norm=1,
    cqt_sparsity=0.01,
    cqt_window='hann',
    cqt_scale=True,
    cqt_pad_mode='constant',
    cqt_res_type='soxr_hq',
    cqt_value_scale=8.0,
    cqt_compression=10.0,
    cqt_condition_frames=int(22050 * 5) // 256,

    # Model params
    residual_layers=39,       #eigenverandering 30->39
    residual_channels=64,
    dilation_cycle_length=13, #eigenverandering 10->13
    unconditional = False,  #eigenverandering
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

    # unconditional sample len
    audio_len = int(22050*5), # unconditional_synthesis_samples #eigenverandering

    # piano roll encoder (note that n_mels is still in use for note dimension)
    conditioner_hidden_channels=128,
    conditioner_out_channels=128,
    conditioner_dilations=[1, 2, 4, 8]
)
