import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import scipy.io as sio

# ------------------- Data loading-------------------
def sliding_window_cut(signal, window_size=128, overlap_ratio=0.5):
    n_channels, n_timesteps = signal.shape
    step = int(window_size * (1 - overlap_ratio))
    n_windows = (n_timesteps - window_size) // step + 1

    # Generate window index
    windows = np.zeros((n_windows, n_channels, 1, window_size))
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        windows[i] = signal[:, start:end].reshape(n_channels, 1, window_size)

    return windows


class EEGDataset(Dataset):
    def __init__(self, data_path: str, snr: float, window_size=128, overlap_ratio=0.5):
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.data, self.noisy_data, self.labels = self._preprocess_data(data_path, snr)

    def _preprocess_data(self, data_path: str, snr_db: float):
        all_clean, all_noisy, all_labels = [], [], []

        for file in os.listdir(data_path):
            mat = sio.loadmat(os.path.join(data_path, file))
            eeg = mat['data'][:, :32, :]
            emg = mat['data'][:, 32:34, :]
            eog = mat['data'][:, 34:36, :]
            labels = mat['labels'][:, :2]

            for trial in range(eeg.shape[0]):
                eeg_rms = np.sqrt(np.mean(eeg[trial] ** 2))

                emg_noise = emg[trial][np.random.choice(2, 32), :]
                eog_noise = eog[trial][np.random.choice(2, 32), :]

                mixed_noise = 0.5 * emg_noise + 0.5 * eog_noise

                noise_rms = np.sqrt(np.mean(mixed_noise ** 2))
                snr_linear = 10 ** (snr_db / 20)
                lambda_noise = eeg_rms / (snr_linear * noise_rms)

                noisy = eeg[trial] + lambda_noise * mixed_noise
                noisy += np.random.randn(*eeg[trial].shape) * 0.01

                clean_windows = sliding_window_cut(eeg[trial], self.window_size, self.overlap_ratio)
                noisy_windows = sliding_window_cut(noisy, self.window_size, self.overlap_ratio)

                binary_labels = (labels[trial] > 5).astype(np.float32)
                all_labels.extend([binary_labels] * len(clean_windows))
                all_clean.extend(clean_windows)
                all_noisy.extend(noisy_windows)

        return np.array(all_clean), np.array(all_noisy), np.array(all_labels)

    def __getitem__(self, idx):
        noisy = torch.FloatTensor(self.noisy_data[idx])
        clean = torch.FloatTensor(self.data[idx])
        label = torch.FloatTensor(self.labels[idx])
        return noisy, clean, label

    def __len__(self):
        return len(self.data)
