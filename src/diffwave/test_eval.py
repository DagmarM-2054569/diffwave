# test_eval.py
import os
import glob
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffwave.params import params
from diffwave.model import DiffWave


def load_test_wavs(test_dir):
    """Load all test waveforms and normalize them"""
    filenames = glob.glob(f'{test_dir}/*.wav')
    test_audio = []
    for fname in filenames:
        audio, sr = torchaudio.load(fname)
        assert sr == params.sample_rate, f"Sample rate mismatch in {fname}"
        audio = audio[0]  # take first channel
        audio = torch.clamp(audio, -1.0, 1.0)

        # Handle cases where audio is shorter than required
        if len(audio) < params.audio_len:
            # Pad with zeros if too short
            audio = torch.nn.functional.pad(audio, (0, params.audio_len - len(audio)))
        elif len(audio) > params.audio_len:
            # Only crop if there's actually extra length
            max_start = len(audio) - params.audio_len
            if max_start > 0:
                start = torch.randint(0, max_start, (1,)).item()
                audio = audio[start:start + params.audio_len]
            else:
                # If exactly equal, just take the whole thing
                audio = audio[:params.audio_len]

        test_audio.append(audio)
    return torch.stack(test_audio)

def evaluate_checkpoint(model, checkpoint_path, test_audio, device):
    """Evaluate a single checkpoint on test data"""
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.eval()

    # Precompute noise levels
    noise_levels = torch.from_numpy(np.cumprod(1 - np.array(params.noise_schedule))).float().to(device)

    losses = []
    with torch.no_grad():
        for audio in test_audio:
            audio = audio.to(device)
            N = 1  # Batch size of 1

            # Correct shape preparation: [batch_size, channels, length]
            audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, length]

            # Same noise sampling as during training
            t = torch.randint(0, len(params.noise_schedule), [N], device=device)
            noise_scale = noise_levels[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale ** 0.5
            noise = torch.randn_like(audio)
            noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

            predicted = model(noisy_audio, t, None)  # None for spectrogram in unconditional
            loss = torch.nn.functional.l1_loss(noise, predicted)
            losses.append(loss.item())

    return np.mean(losses)


def plot_test_losses(checkpoint_dir, test_dir, output_plot="test_losses.png"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    print("Loading test data...")
    test_audio = load_test_wavs(test_dir)

    # Initialize model
    model = DiffWave(params).to(device)

    # Find all checkpoints
    checkpoint_files = sorted(glob.glob(f'{checkpoint_dir}/weights-*.pt'),
                              key=lambda x: int(x.split('-')[-1].split('.')[0]))

    if not checkpoint_files:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    # Evaluate each checkpoint
    steps = []
    losses = []
    print("Evaluating checkpoints...")
    for checkpoint_path in tqdm(checkpoint_files):
        try:
            step = int(os.path.basename(checkpoint_path).split('-')[-1].split('.')[0])
            loss = evaluate_checkpoint(model, checkpoint_path, test_audio, device)
            steps.append(step)
            losses.append(loss)
            print(f"Step {step}: Loss = {loss:.4f}")
        except Exception as e:
            print(f"Skipping {checkpoint_path} due to error: {str(e)}")
            continue

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, 'b-o')
    plt.title('Test Loss vs Training Step')
    plt.xlabel('Training Step')
    plt.ylabel('L1 Loss')
    plt.grid(True)
    plt.savefig(output_plot)
    print(f"Saved plot to {output_plot}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir', help='Directory containing weight checkpoints')
    parser.add_argument('test_dir', help='Directory containing test .wav files')
    parser.add_argument('--output', default='test_losses.png', help='Output plot filename')
    args = parser.parse_args()

    plot_test_losses(args.checkpoint_dir, args.test_dir, args.output)