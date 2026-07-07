# DiffWave
![PyPI Release](https://img.shields.io/pypi/v/diffwave?label=release) [![License](https://img.shields.io/github/license/lmnt-com/diffwave)](https://github.com/lmnt-com/diffwave/blob/master/LICENSE)

**We're hiring!**
If you like what we're building here, [come join us at LMNT](https://explore.lmnt.com).

DiffWave is a fast, high-quality neural vocoder and waveform synthesizer. It starts with Gaussian noise and converts it into speech via iterative refinement. The speech can be controlled by providing a conditioning signal (e.g. log-scaled Mel spectrogram). The model and architecture details are described in [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf).

## What's new (2021-11-09)
- unconditional waveform synthesis (thanks to [Andrechang](https://github.com/Andrechang)!)

## What's new (2021-04-01)
- fast sampling algorithm based on v3 of the DiffWave paper

## What's new (2020-10-14)
- new pretrained model trained for 1M steps
- updated audio samples with output from new model

## Status (2021-11-09)
- [x] fast inference procedure
- [x] stable training
- [x] high-quality synthesis
- [x] mixed-precision training
- [x] multi-GPU training
- [x] command-line inference
- [x] programmatic inference API
- [x] PyPI package
- [x] audio samples
- [x] pretrained models
- [x] unconditional waveform synthesis

Big thanks to [Zhifeng Kong](https://github.com/FengNiMa) (lead author of DiffWave) for pointers and bug fixes.

## Audio samples
[22.05 kHz audio samples](https://lmnt.com/assets/diffwave)

## Pretrained models
[22.05 kHz pretrained model](https://lmnt.com/assets/diffwave/diffwave-ljspeech-22kHz-1000578.pt) (31 MB, SHA256: `d415d2117bb0bba3999afabdd67ed11d9e43400af26193a451d112e2560821a8`)

This pre-trained model is able to synthesize speech with a real-time factor of 0.87 (smaller is faster).

### Pre-trained model details
- trained on 4x 1080Ti
- default parameters
- single precision floating point (FP32)
- trained on LJSpeech dataset excluding LJ001&ast; and LJ002&ast;
- trained for 1000578 steps (1273 epochs)

## Install

Install using pip:
```
pip install diffwave
```

or from GitHub:
```
git clone https://github.com/lmnt-com/diffwave.git
cd diffwave
pip install .
```

### Training
Before you start training, you'll need to prepare a training dataset. The dataset can have any directory structure as long as the contained .wav files are 16-bit mono (e.g. [LJSpeech](https://keithito.com/LJ-Speech-Dataset/), [VCTK](https://pytorch.org/audio/_modules/torchaudio/datasets/vctk.html)). By default, this implementation assumes a sample rate of 22.05 kHz. If you need to change this value, edit [params.py](https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/params.py).

```
python -m diffwave.preprocess /path/to/dir/containing/wavs
python -m diffwave /path/to/model/dir /path/to/dir/containing/wavs

# in another shell to monitor training progress:
tensorboard --logdir /path/to/model/dir --bind_all
```

You should expect to hear intelligible (but noisy) speech by ~8k steps (~1.5h on a 2080 Ti).

### Complex CQT target training
This checkout can train a symbolic-conditioning model whose diffusion target is a
complex CQT instead of a waveform. In this mode, the existing `.wav.spec.npy`
files are still the symbolic piano-roll-like conditioning files. They are not
replaced by CQT files. For each cropped WAV segment, training computes a
normalized complex CQT target shaped `[2, 89, T]`, where channel `0` is real and
channel `1` is imaginary. The default CQT settings are 22.05 kHz, 12 bins per
octave, 89 bins, `fmin=27.5`, and `hop_length=1`.

The model flattens `[2, 89, T]` to `[178, T]` internally. The symbolic
conditioner still upsamples by `hop_samples`, so CQT crops use
`cqt_condition_frames * hop_samples` audio samples. With the defaults, that is
`430 * 256 = 110080` samples, which is the existing 5-second crop rounded down
to the conditioning grid.

Use Charbonnier noise-prediction loss:

```
loss = mean(sqrt((predicted_noise - true_noise)^2 + charbonnier_eps^2) - charbonnier_eps)
```

`charbonnier_eps` defaults to `1e-3`. The CQT values are divided by
`cqt_value_scale=8.0` and then compressed with an invertible `log1p` magnitude
compression (`cqt_compression=10.0`), so the inverse renderer can undo the
normalization before waveform reconstruction.

For early runs where the dense WAV-derived conditioning row is a shortcut, pass:

```
python -m diffwave /path/to/model/dir /path/to/data --ignore_global_volume_row
```

This zeros conditioning row `0` only. MIDI pitch rows and their velocity-derived
note intensities are kept intact. Later refinement runs can omit the flag.

`cqt_backend=auto` requires `nnAudio` on a CUDA tensor. It intentionally refuses
to fall back to CPU `librosa`, so a missing GPU CQT path fails fast instead of
quietly starting a slow CPU training run. Install the optional GPU CQT
dependency with:

```
pip install ".[cqt-gpu]"
```

Use `--cqt_backend librosa` only for explicit short CPU diagnostics.

Inference samples a complex CQT target and renders it to WAV with `librosa.icqt`
by default:

```
python -m diffwave.inference /path/to/model -s /path/to/file.wav.spec.npy -o output.wav
```

The alternate renderer is explicitly diagnostic:

```
python -m diffwave.inference /path/to/model -s /path/to/file.wav.spec.npy -o diagnostic.wav --cqt_renderer diagnostic
```

`--cqt_renderer diagnostic` uses magnitude-only CQT Griffin-Lim and is not a
mathematically exact inverse of the predicted complex CQT.

#### Multi-GPU training
By default, this implementation uses as many GPUs in parallel as returned by [`torch.cuda.device_count()`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.device_count). You can specify which GPUs to use by setting the [`CUDA_DEVICES_AVAILABLE`](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) environment variable before running the training module.

### Inference API
Basic usage:

```python
from diffwave.inference import predict as diffwave_predict

model_dir = '/path/to/model/dir'
spectrogram = # get your hands on a spectrogram in [N,C,W] format
audio, sample_rate = diffwave_predict(spectrogram, model_dir, fast_sampling=True)

# audio is a GPU tensor in [N,T] format.
```

### Inference CLI
```
python -m diffwave.inference --fast /path/to/model /path/to/spectrogram -o output.wav
```

## References
- [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Code for Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion)
