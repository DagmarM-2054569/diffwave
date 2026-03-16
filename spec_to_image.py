import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_and_plot_spectrogram(spec_file, output_image_file=None):
    """
    Load a .spec.npy file and plot it as an image where:
    - Width = number of time frames (1 pixel per frame).
    - Height = n_mels (number of frequency bins).
    """
    # Load the spectrogram
    spectrogram = np.load(spec_file)

    # The spectrogram is already magnitude-only (Mel spectrograms discard phase).
    # Shape: (n_mels, time_frames)
    print(f"Spectrogram shape: {spectrogram.shape}")

    # Transpose to (time_frames, n_mels) for plotting (time on x-axis, frequency on y-axis)

    # Create the plot
    plt.figure(figsize=(10, 4))
    #plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Magnitude (scaled)')
    plt.xlabel('Time (frames)')
    plt.ylabel('Mel Frequency Bin')
    plt.title('Spectrogram')

    # Save or show the image
    if output_image_file:
        plt.savefig(output_image_file, bbox_inches='tight', dpi=300)
        print(f"Saved image to: {output_image_file}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a .spec.npy file to an image.')
    parser.add_argument('spec_file', help='Path to the .spec.npy file')
    parser.add_argument('--output', help='Output image file (e.g., spectrogram.png)', default=None)
    args = parser.parse_args()

    load_and_plot_spectrogram(args.spec_file, args.output)

# example usage: python spec_to_image.py path/to/audio_file.wav.spec.npy --output spectrogram.png