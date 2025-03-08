from required_libraries import *

def add_noise(audio_path, noise_path, snr_dB):
    # Load audio file using torchaudio
    y, sr = torchaudio.load(audio_path)  # Shape: (channels, samples)

    # Convert to mono if needed
    if y.shape[0] > 1:
        y = torch.mean(y, dim=0, keepdim=True)

    # Load noise file at the same sample rate
    noise, noise_sr = torchaudio.load(noise_path)

    # Convert noise to mono if needed
    if noise.shape[0] > 1:
        noise = torch.mean(noise, dim=0, keepdim=True)

    # Resample noise if sample rates do not match
    if noise_sr != sr:
        noise = torchaudio.transforms.Resample(orig_freq=noise_sr, new_freq=sr)(noise)

    # Ensure noise is at least as long as the audio
    if noise.shape[1] < y.shape[1]:
        noise = noise.repeat(1, (y.shape[1] // noise.shape[1]) + 1)[:, :y.shape[1]]
    elif noise.shape[1] > y.shape[1]:
        start = torch.randint(0, noise.shape[1] - y.shape[1] + 1, (1,)).item()
        noise = noise[:, start:start + y.shape[1]]

    # Compute power of original signal and noise
    signal_power = torch.mean(y ** 2)
    noise_power = torch.mean(noise ** 2)

    # Prevent division by zero
    if noise_power == 0:
        raise ValueError("Noise file is silent; cannot add noise.")

    # Compute target noise power for given SNR
    target_noise_power = signal_power / (10 ** (snr_dB / 10))

    # Scale noise to match target power
    noise = noise * torch.sqrt(target_noise_power / noise_power)

    # Add noise to the original signal
    noisy_signal = y + noise

    # Clip values to maintain valid audio range
    noisy_signal = torch.clamp(noisy_signal, -1.0, 1.0)

    return noisy_signal, sr  # Return sample rate for saving the file