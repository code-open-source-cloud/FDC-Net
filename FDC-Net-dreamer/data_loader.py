import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import h5py
import scipy.io

# ------------------- Data loading-------------------
class DREAMERDataset(Dataset):
    def __init__(self, data_path: str, snr: float, window_size=128, overlap_ratio=0.5):
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.data, self.noisy_data, self.labels = self._preprocess_data(data_path, snr)

    def _load_mat_data(self, data_path):
        try:
            mat_data = scipy.io.loadmat(data_path)
        except:
            mat_data = h5py.File(data_path, 'r')

        dreamer_data = mat_data['DREAMER']
        data = dreamer_data['Data'][0]
        data = data[0]

        eeg_data = []
        ecg_data = []
        labels = []

        for subj_idx in range(data.shape[1]):
            subj_struct = data[0, subj_idx]
            eeg_struct = subj_struct['EEG'][0, 0]
            eeg_stimuli = eeg_struct['stimuli'][0, 0]
            ecg_struct = subj_struct['ECG'][0, 0]
            ecg_stimuli = ecg_struct['stimuli'][0, 0]
            valence = subj_struct['ScoreValence'][0, 0]
            arousal = subj_struct['ScoreArousal'][0, 0]

            for trial in range(18):
                eeg_trial = eeg_stimuli[trial][0]
                if eeg_trial.shape[1] != 14:
                    eeg_trial = eeg_trial.T

                ecg_trial = ecg_stimuli[trial][0]
                if ecg_trial.shape[1] != 2:
                    ecg_trial = ecg_trial.T

                label = np.array([
                    valence[trial] if valence.ndim == 1 else valence[trial, 0],
                    arousal[trial] if arousal.ndim == 1 else arousal[trial, 0]
                ])

                eeg_data.append(eeg_trial)
                ecg_data.append(ecg_trial)
                labels.append(label)

        eeg_min_length = min([eeg.shape[0] for eeg in eeg_data])
        eeg_data = [eeg[:eeg_min_length, :] for eeg in eeg_data]
        ecg_data = [ecg[:eeg_min_length, :] for ecg in ecg_data]

        eeg_data = np.stack(eeg_data, axis=0)
        eeg_mean = eeg_data.mean(axis=(0, 1), keepdims=True)
        eeg_std = eeg_data.std(axis=(0, 1), keepdims=True)
        eeg_data = (eeg_data - eeg_mean) / (eeg_std + 1e-8)

        ecg_data = np.stack(ecg_data, axis=0)
        labels = np.stack(labels, axis=0)

        return eeg_data, ecg_data, labels

    def _preprocess_data(self, data_path: str, snr_db: float):
        eeg_data, ecg_data, labels = self._load_mat_data(data_path)
        eeg_array = np.stack(eeg_data, axis=0)
        ecg_array = np.stack(ecg_data, axis=0)
        labels_array = np.array(labels)

        all_clean, all_noisy, all_labels = [], [], []

        for trial_idx in range(eeg_array.shape[0]):
            clean_eeg = eeg_array[trial_idx].T
            ecg = ecg_array[trial_idx].T
            label = labels_array[trial_idx]

            eeg_rms = np.sqrt(np.mean(clean_eeg ** 2))
            ecg_noise = ecg[np.random.choice(2, size=14, replace=True), :]
            mixed_noise = ecg_noise

            noise_rms = np.sqrt(np.mean(mixed_noise ** 2))
            snr_linear = 10 ** (snr_db / 20)
            lambda_noise = eeg_rms / (snr_linear * noise_rms)
            noisy_eeg = clean_eeg + lambda_noise * mixed_noise
            noisy_eeg = (noisy_eeg - noisy_eeg.mean()) / (noisy_eeg.std() + 1e-8)

            clean_windows = self._sliding_window_cut(clean_eeg)
            noisy_windows = self._sliding_window_cut(noisy_eeg)

            binary_label = (label > 3).astype(np.float32)

            all_labels.extend([binary_label] * len(clean_windows))
            all_clean.extend(clean_windows)
            all_noisy.extend(noisy_windows)

        clean_data = np.stack(all_clean, axis=0)
        noisy_data = np.stack(all_noisy, axis=0)
        label_data = np.stack(all_labels, axis=0)

        return clean_data, noisy_data, label_data

    def _sliding_window_cut(self, signal, window_size=None, overlap_ratio=None):
        window_size = self.window_size if window_size is None else window_size
        overlap_ratio = self.overlap_ratio if overlap_ratio is None else overlap_ratio

        n_channels, n_timesteps = signal.shape
        step = int(window_size * (1 - overlap_ratio))
        n_windows = (n_timesteps - window_size) // step + 1

        windows = np.zeros((n_windows, n_channels, 1, window_size))
        for i in range(n_windows):
            start = i * step
            end = start + window_size
            windows[i] = signal[:, start:end].reshape(n_channels, 1, window_size)

        return windows

    def __getitem__(self, idx):
        noisy = torch.FloatTensor(self.noisy_data[idx])
        clean = torch.FloatTensor(self.data[idx])
        label = torch.FloatTensor(self.labels[idx])
        return noisy, clean, label

    def __len__(self):
        return len(self.data)
