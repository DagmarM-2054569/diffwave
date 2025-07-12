import numpy as np
import os

audio_dir = r"C:\Users\meers\OneDrive\Bureaublad\unief\diffwave parent folder\rachchunks_16bitmono22050Hz"
for file in os.listdir(audio_dir):
    if file.endswith(".npy"):
        try:
            data = np.load(os.path.join(audio_dir, file))
            if data.size == 0:
                print(f"Empty file: {file}")
        except Exception as e:
            print(f"Corrupted file: {file} | Error: {e}")