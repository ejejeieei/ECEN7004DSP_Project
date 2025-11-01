import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

def load_eeg_segment(
    edf_path: str,
    start_time_sec: float = 0.0,
    duration_sec: float = 30.0,
    channel_name: str = None,
    channel_index: int = None,
    verbose: bool = False
):
    """
    从 Sleep-EDF .edf 文件中加载一段单通道 EEG 信号。

    参数:
        edf_path (str): EDF 文件的完整路径（支持 Windows 路径）
        start_time_sec (float): 起始时间（秒），默认 0
        duration_sec (float): 片段长度（秒），默认 30（标准睡眠分期窗口）
        channel_name (str, optional): 通道名称，如 'EEG Fpz-Cz'
        channel_index (int, optional): 通道索引（从 0 开始），如 1
        verbose (bool): 是否显示 MNE 加载信息

    返回:
        eeg_signal (np.ndarray): 一维 EEG 信号，形状 (n_samples,)
        sfreq (float): 采样率（Hz）

    注意:
        - 必须指定 channel_name 或 channel_index 之一
        - Sleep-EDF 常见通道名包括: 'EEG Fpz-Cz', 'EEG Pz-Oz'
    """
    if not (channel_name or channel_index is not None):
        raise ValueError("必须指定 channel_name 或 channel_index")

    # 加载原始数据（不自动加载到内存以节省资源）
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=verbose)

    sfreq = raw.info['sfreq']
    total_duration = raw.n_times / sfreq

    # 边界检查
    if start_time_sec + duration_sec > total_duration:
        raise ValueError(f"请求的时间段超出信号总长度 ({total_duration:.1f}s)")

    # 确定通道
    if channel_name:
        picks = mne.pick_channels(raw.ch_names, include=[channel_name])
        if len(picks) == 0:
            raise ValueError(f"通道 '{channel_name}' 未找到。可用通道: {raw.ch_names}")
        ch_idx = picks[0]
    else:
        if channel_index >= len(raw.ch_names):
            raise IndexError(f"通道索引 {channel_index} 超出范围。总通道数: {len(raw.ch_names)}")
        ch_idx = channel_index

    # 计算样本索引
    start_samp = int(start_time_sec * sfreq)
    n_samp = int(duration_sec * sfreq)
    end_samp = start_samp + n_samp

    # 提取单通道数据
    eeg_data, _ = raw[ch_idx, start_samp:end_samp]  # shape: (1, n_samp)
    eeg_signal = np.squeeze(eeg_data)  # 转为一维

    return eeg_signal, sfreq


def plot_spectrum(
    x: np.ndarray,
    fs: float,
    window_type: str = 'hamming',
    nperseg: int = None,
    plot: bool = True,
    title_suffix: str = ""
):
    """
    Perform spectral analysis using scipy tools.
    Compute dominant frequency and bandwidth (95% energy bandwidth).
    
    Parameters:
        x (np.ndarray): Input 1D signal
        fs (float): Sampling frequency (Hz)
        window_type (str): 'rectangular', 'hamming', or 'hanning'
        nperseg (int): Window length (default: full signal length)
        plot (bool): Whether to plot spectrum
        title_suffix (str): Suffix for plot title
    
    Returns:
        dominant_freq (float): Frequency with max amplitude
        bandwidth_95 (float): Bandwidth containing 95% of total energy (Hz)
        freqs (np.ndarray): Frequency vector
        magnitudes (np.ndarray): Magnitude spectrum
    """
    N = len(x)
    if nperseg is None:
        nperseg = N
    if nperseg > N:
        nperseg = N

    # Truncate signal
    x_seg = x[:nperseg]

    # Apply window
    if window_type == 'rectangular':
        win = np.ones(nperseg)
    elif window_type == 'hamming':
        win = signal.windows.hamming(nperseg)
    elif window_type == 'hanning':
        win = signal.windows.hann(nperseg)
    else:
        raise ValueError("window_type must be 'rectangular', 'hamming', or 'hanning'")
    
    x_windowed = x_seg * win

    # Compute FFT
    X = fft(x_windowed)
    freqs = fftfreq(nperseg, 1 / fs)
    
    # Keep positive frequencies
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    X = X[pos_mask]
    magnitudes = (2.0 / nperseg) * np.abs(X)

    # --- Dominant frequency ---
    dominant_idx = np.argmax(magnitudes)
    dominant_freq = freqs[dominant_idx]

    # --- Bandwidth: 95% energy bandwidth ---
    total_energy = np.sum(magnitudes**2)
    cum_energy = np.cumsum(magnitudes**2)
    idx_95 = np.searchsorted(cum_energy, 0.95 * total_energy)
    bandwidth_95 = freqs[idx_95]  # from 0 to this freq contains 95% energy

    # Optional: Plot
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, magnitudes, label=f'{window_type.capitalize()} window')
        plt.axvline(dominant_freq, color='r', linestyle='--', 
                    label=f'Dominant freq: {dominant_freq:.2f} Hz')
        plt.axvline(bandwidth_95, color='g', linestyle='--', 
                    label=f'95% BW: {bandwidth_95:.2f} Hz')
        plt.title(f'Spectrum ({window_type.capitalize()} Window, N={nperseg}) {title_suffix}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, 30)  # Focus on 0–50 Hz for sleep EEG
        plt.grid(True, ls='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return dominant_freq, bandwidth_95, freqs, magnitudes


def plot_fixed_window_vary_length(
    eeg_signal: np.ndarray,
    fs: float,
    window_type: str = 'hamming',
    window_lengths: list = None,
    xlim: float = 50.0
):
    """
    Plot spectra for a FIXED window type with VARYING window lengths.
    - Each length uses a distinct, highly visible color.
    - Dominant frequency: solid line (same color)
    - 95% bandwidth: dashed line (same color)
    
    Parameters:
        eeg_signal: Input 1D EEG signal
        fs: Sampling frequency (Hz)
        window_type: One of 'hamming', 'hanning', 'rectangular'
        window_lengths: List of window lengths (e.g., [3000, 512, 256, 128])
        xlim: Max frequency to display (Hz)
    """
    if window_lengths is None:
        full_len = len(eeg_signal)
        window_lengths = [full_len, 512, 256, 128]
    
    # Filter invalid lengths
    window_lengths = [n for n in window_lengths if n <= len(eeg_signal)]
    
    # Use highly distinguishable colors (e.g., Tableau 10 or similar)
    distinct_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
    ]
    
    plt.figure(figsize=(12, 7))
    results = []

    for i, nperseg in enumerate(window_lengths):
        color = distinct_colors[i % len(distinct_colors)]

        # Get window
        if window_type == 'rectangular':
            win = np.ones(nperseg)
            win_name = 'Rectangular'
        elif window_type == 'hamming':
            win = signal.windows.hamming(nperseg)
            win_name = 'Hamming'
        elif window_type == 'hanning':
            win = signal.windows.hann(nperseg)
            win_name = 'Hanning'
        else:
            raise ValueError("Unsupported window_type. Use 'hamming', 'hanning', or 'rectangular'.")

        # Apply window
        x_seg = eeg_signal[:nperseg]
        x_win = x_seg * win

        # FFT
        X = fft(x_win)
        freqs = fftfreq(nperseg, 1 / fs)
        pos = freqs >= 0
        freqs = freqs[pos]
        magnitudes = (2.0 / nperseg) * np.abs(X[pos])

        # Dominant frequency
        f_dom = freqs[np.argmax(magnitudes)]

        # 95% energy bandwidth
        energy = magnitudes ** 2
        cum_energy = np.cumsum(energy)
        idx_95 = np.searchsorted(cum_energy, 0.95 * cum_energy[-1])
        bw_95 = freqs[idx_95]

        results.append((nperseg, f_dom, bw_95))

        # Plot spectrum
        plt.plot(freqs, magnitudes, color=color, linewidth=2.0, label=f'N = {nperseg}')

        # Mark dominant frequency (solid line)
        plt.axvline(f_dom, color=color, linestyle='-', linewidth=1.5, alpha=0.85)
        # Mark bandwidth (dashed line)
        plt.axvline(bw_95, color=color, linestyle='--', linewidth=1.5, alpha=0.85)

    plt.title(f'Spectral Analysis ({win_name} Window, Varying Length)', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim(0, min(xlim, fs / 2))
    plt.grid(True, ls='--', alpha=0.6)
    plt.legend(title='Window Length', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Print table for report
    print(f"{'N (samples)':>10} | {'Dominant Freq (Hz)':>18} | {'95% BW (Hz)':>14}")
    print("-" * 50)
    for n, f_dom, bw in results:
        print(f"{n:>10} | {f_dom:>18.2f} | {bw:>14.2f}")
import numpy as np
import numpy as np

def add_interference_to_eeg(
    eeg_signal: np.ndarray,
    fs: float,
    noise_types: list = None,
    snr_db: float = 15.0,
    fixed_freqs: list = None,
    relative_amplitudes: list = None,   # e.g., [0.2, 0.1] means 20%, 10% of EEG std
    amplitude_ref: str = 'std'          # 'std' or 'peak'
):
    """
    Add interference with amplitudes defined RELATIVE to the original EEG signal.
    
    Parameters:
        eeg_signal (np.ndarray): Clean EEG signal
        fs (float): Sampling frequency (Hz)
        noise_types (list): ['fixed_freq', 'white']
        snr_db (float): SNR for white noise (dB)
        fixed_freqs (list): Frequencies to add, e.g., [50, 100]
        relative_amplitudes (list): Relative amplitudes (unitless), e.g., [0.2, 0.1]
            - If None, defaults to 0.1 (10%) for all frequencies
        amplitude_ref (str): Reference for relative amplitude:
            - 'std': amplitude = rel_amp * np.std(eeg_signal)
            - 'peak': amplitude = rel_amp * np.max(np.abs(eeg_signal))
    
    Returns:
        noisy_signal, noise
    """
    if noise_types is None:
        noise_types = ['fixed_freq', 'white']
    if fixed_freqs is None:
        fixed_freqs = [50.0]
    if relative_amplitudes is None:
        relative_amplitudes = [0.1] * len(fixed_freqs)
    if len(relative_amplitudes) != len(fixed_freqs):
        raise ValueError("Length of relative_amplitudes must match fixed_freqs")

    # Choose reference amplitude
    if amplitude_ref == 'std':
        ref_amp = np.std(eeg_signal)
    elif amplitude_ref == 'peak':
        ref_amp = np.max(np.abs(eeg_signal))
    else:
        raise ValueError("amplitude_ref must be 'std' or 'peak'")

    N = len(eeg_signal)
    t = np.arange(N) / fs
    noise = np.zeros(N)

    # --- Fixed-frequency sinusoidal interference ---
    if 'fixed_freq' in noise_types:
        for f, rel_amp in zip(fixed_freqs, relative_amplitudes):
            if f > fs / 2:
                print(f"Warning: {f} Hz > Nyquist ({fs/2} Hz); may alias.")
            amp = rel_amp * ref_amp
            noise += amp * np.sin(2 * np.pi * f * t)

    # --- White Gaussian noise (controlled by SNR, also relative) ---
    if 'white' in noise_types:
        signal_power = np.mean(eeg_signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise += np.random.normal(0, np.sqrt(noise_power), N)

    noisy_signal = eeg_signal + noise
    return noisy_signal, noise

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def design_and_apply_fir_bandstop(
    noisy_signal: np.ndarray,
    fs: float,
    stopband: tuple = (8, 27),
    passband_ripple: float = 0.1,
    stopband_attenuation: float = 60,
    plot_response: bool = True
):
    """
    Design and apply an FIR band-stop filter to remove interference in [10, 25] Hz.
    
    Parameters:
        noisy_signal: Input noisy EEG
        fs: Sampling frequency
        stopband: (f_low, f_high) in Hz, e.g., (8, 27)
        passband_ripple: Max ripple in passband (dB)
        stopband_attenuation: Min attenuation in stopband (dB)
        plot_response: Whether to plot filter characteristics
    
    Returns:
        filtered_signal, b (filter coefficients)
    """
    nyq = fs / 2
    wp1 = 6 / nyq      # Lower passband edge
    ws1 = stopband[0] / nyq   # Stopband start
    ws2 = stopband[1] / nyq   # Stopband end
    wp2 = 30 / nyq     # Upper passband edge (adjust if needed)

    # Normalize frequencies
    wp = [wp1, wp2]
    ws = [ws1, ws2]

    # Estimate filter order using kaiserord
    delta_pass = (10**(passband_ripple/20) - 1) / (10**(passband_ripple/20) + 1)
    delta_stop = 10**(-stopband_attenuation/20)
    deltas = [delta_pass, delta_stop, delta_pass]
    N, beta = signal.kaiserord(ripple=stopband_attenuation, width=(ws1 - wp1))
    # Ensure even length for type I FIR (linear phase)
    if N % 2 == 0:
        N += 1

    # Design band-stop FIR using firwin2 or remez (here use firwin with custom bands)
    # Simpler: use signal.firwin with bandstop
    b = signal.firwin(
        numtaps=N,
        cutoff=[stopband[0], stopband[1]],
        window=('kaiser', beta),
        pass_zero='bandstop',
        fs=fs
    )

    # Apply zero-phase filtering
    filtered_signal = signal.filtfilt(b, [1.0], noisy_signal)

    if plot_response:
        # Frequency response
        w, h = signal.freqz(b, worN=4096, fs=fs)
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        # Magnitude (dB)
        ax[0, 0].plot(w, 20 * np.log10(np.abs(h)))
        ax[0, 0].set_title('Magnitude Response')
        ax[0, 0].set_xlabel('Frequency (Hz)')
        ax[0, 0].set_ylabel('Magnitude (dB)')
        ax[0, 0].grid()
        ax[0, 0].axvspan(stopband[0], stopband[1], color='red', alpha=0.1, label='Stopband')
        ax[0, 0].legend()

        # Phase response
        angles = np.unwrap(np.angle(h))
        ax[0, 1].plot(w, angles)
        ax[0, 1].set_title('Phase Response')
        ax[0, 1].set_xlabel('Frequency (Hz)')
        ax[0, 1].grid()

        # Group delay
        from scipy.signal import group_delay
        w_gd, gd = group_delay((b, [1.0]), w=4096, fs=fs)
        ax[1, 0].plot(w_gd, gd)
        ax[1, 0].set_title('Group Delay')
        ax[1, 0].set_xlabel('Frequency (Hz)')
        ax[1, 0].set_ylabel('Samples')
        ax[1, 0].grid()

        # Pole-zero plot (FIR: all poles at origin)
        zeros = np.roots(b)
        ax[1, 1].scatter(np.real(zeros), np.imag(zeros), marker='o', label='Zeros')
        ax[1, 1].scatter([0], [0], marker='x', s=100, label='Poles (origin)')
        ax[1, 1].set_title('Pole-Zero Plot')
        ax[1, 1].set_xlabel('Real')
        ax[1, 1].set_ylabel('Imaginary')
        ax[1, 1].grid()
        ax[1, 1].axis('equal')
        ax[1, 1].legend()

        plt.tight_layout()
        plt.show()

    return filtered_signal, b