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
    batch_size=15,   #eigenverandering (2)   29 causes memory crash
    learning_rate=4e-5, #eigenverandering 2e-4 -> 2e-5
    max_grad_norm=None,

    # Data params
    sample_rate=22050,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    crop_mel_frames=256,  # Probably an error in paper. eigenverandering 62->160

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
