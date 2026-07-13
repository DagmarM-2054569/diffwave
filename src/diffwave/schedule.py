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


def rescale_zero_terminal_snr(beta):
  """Rescale beta values so cumulative alpha reaches zero at the last step."""
  beta = np.asarray(beta, dtype=np.float64)
  if beta.ndim != 1:
    raise ValueError(f'Expected a 1-D beta schedule, got shape {beta.shape}.')
  if len(beta) < 2:
    raise ValueError('Need at least two beta values for zero-terminal-SNR rescaling.')
  if np.any(beta <= 0.0) or np.any(beta >= 1.0):
    raise ValueError('All beta values must be in the open interval (0, 1).')

  alpha_bar_sqrt = np.sqrt(np.cumprod(1.0 - beta))
  alpha_bar_sqrt_0 = alpha_bar_sqrt[0].copy()
  alpha_bar_sqrt_T = alpha_bar_sqrt[-1].copy()
  if alpha_bar_sqrt_T == 0.0:
    return beta.astype(np.float64)

  alpha_bar_sqrt = alpha_bar_sqrt - alpha_bar_sqrt_T
  alpha_bar_sqrt *= alpha_bar_sqrt_0 / (alpha_bar_sqrt_0 - alpha_bar_sqrt_T)

  alpha_bar = alpha_bar_sqrt ** 2
  alpha = np.empty_like(beta)
  alpha[0] = alpha_bar[0]
  alpha[1:] = alpha_bar[1:] / alpha_bar[:-1]
  return np.clip(1.0 - alpha, 0.0, 1.0).astype(np.float64)
