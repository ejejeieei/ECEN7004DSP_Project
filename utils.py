import mne
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks

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


def fir_notch(
    freq: float,
    fs: float,
    Q: float = 30.0,
    numtaps: int = None
) -> tuple:
    """
    设计FIR陷波滤波器（带阻）
    
    参数:
    freq : 要陷波的中心频率 (Hz)
    fs   : 采样频率 (Hz)
    Q    : 品质因数 (Q = freq / bandwidth)，值越大，陷波越窄
    numtaps : 滤波器阶数（长度）。若为None，自动估算
    
    返回:
    b, a : 滤波器系数（FIR，a=[1]）
    """
    # 计算带宽
    bandwidth = freq / Q
    
    # 避免带宽过窄导致滤波器阶数爆炸
    min_bw = fs / 1000  # 最小带宽 0.1 Hz（可调）
    bandwidth = max(bandwidth, min_bw)
    
    # 陷波频带边界
    low = freq - bandwidth / 2
    high = freq + bandwidth / 2
    
    # 频率不能超过奈奎斯特频率
    nyq = fs / 2
    if high >= nyq:
        high = nyq - 1e-3
    
    if low <= 0:
        low = 1e-3
    
    # 自动估算阶数（基于经验公式）
    if numtaps is None:
        # 过渡带 ≈ bandwidth / 10
        trans_width = bandwidth / 10
        numtaps = int(4 / trans_width * fs)  # 粗略估计
        numtaps = numtaps if numtaps % 2 == 1 else numtaps + 1  # 必须为奇数
    
    # 设计带阻FIR滤波器
    b = signal.firwin(
        numtaps,
        [low, high],
        pass_zero='bandstop',
        fs=fs,
        window='hamming'
    )
    a = np.array([1.0])
    return b, a


def visu_threshold(coeffs, N):
    sigma = np.median(np.abs(coeffs)) / 0.6745
    return sigma * np.sqrt(2 * np.log(N))

# 使用


def optimal_eeg_denoise(
    data:np.ndarray,
    fs: float = 100.0,
    notch_freqs:list=[10, 15, 20, 25]) -> np.ndarray:
    # Step 1: FIR Notch Filter (无振铃)
    for f in notch_freqs:
        # b, a = fir_notch(f, fs, Q=40,numtaps=1001)  # FIR陷波，N=200保证陡峭过渡带
        # data = signal.filtfilt(b, a, data)

        b, a = fir_notch(f, fs, Q=20, numtaps=1001)
        data = signal.filtfilt(b, a, data)
    
    # Step 2: Wavelet Denoising with BayesShrink
    coeffs = pywt.wavedec(data, wavelet='sym5', level=2)
    coeffs_thresh = []
    for i, c in enumerate(coeffs):
        if i == 0:  # Keep approximation
            coeffs_thresh.append(c)
        else:
            # Use BayesShrink for colored noise robustness
            T = visu_threshold(c, len(data))
            c_thresh = pywt.threshold(c, value=T, mode='soft')
            coeffs_thresh.append(c_thresh)

    
    denoised = pywt.waverec(coeffs_thresh, wavelet='sym5')
    return denoised[:len(data)]
    # return data

    import numpy as np

# 如果没有 vmdpy，可使用以下简易实现或安装：
# pip install vmdpy
from vmdpy import VMD

def vmd_denoise(
    data: np.ndarray,
    fs: float = 100.0,
    K: int = 8,      # 分解模态数
    alpha: float = 2000,  # 正则化参数（越大越平滑）
    tau: float = 0   # 噪声容忍度
) -> np.ndarray:
    """
    使用VMD进行EEG去噪
    """
    # Step 1: FIR Notch Filter (先去除周期性干扰)
    notch_freqs = [10, 15, 20, 25]
    for f in notch_freqs:
        # b, a = fir_notch(f, fs, Q=20, numtaps=1001)
        # data = signal.filtfilt(b, a, data)

        b, a = fir_notch(f, fs, Q=20, numtaps=1001)
        data = signal.filtfilt(b, a, data)

    # Step 2: VMD Decomposition
    u, u_hat, omega = VMD(
        data, 
        alpha=alpha,     # 控制模态平滑度
        tau=tau,         # 噪声容忍度
        K=K,             # 模态数（覆盖0–35 Hz）
        DC=False,        # 不包含直流分量
        init=1,           # 初始化方式
        tol=1e-7
    )

    # Step 3: 重构 —— 保留前5个模态（低频+σ波），丢弃高频噪声模态
    denoised = np.sum(u[:5], axis=0)
    
    return denoised[:len(data)]


def check_eeg_time_domain(
    data: np.ndarray,
    fs: float = 100.0,
    segment_length: int = 10000,  # 100秒 @ 100Hz
    plot: bool = True
) -> dict:
    """
    检查EEG时域特征波形（δ, θ, α, σ, K-complex）
    
    返回:
    results : dict 包含各波形的检测结果
    """
    results = {}
    
    # Step 1: 分段处理（避免长信号计算慢）
    segments = []
    for start in range(0, len(data), segment_length):
        end = min(start + segment_length, len(data))
        segments.append(data[start:end])
    
    # Step 2: 设计带通滤波器
    def bandpass_filter(x, low, high, fs):
        b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
        return filtfilt(b, a, x)
    
    # Step 3: 检测各波形
    for i, seg in enumerate(segments):
        seg_results = {}
        
        # δ 波 (0.5–4 Hz)
        delta = bandpass_filter(seg, 0.5, 4, fs)
        seg_results['delta'] = {
            'amplitude': np.max(np.abs(delta)),
            'duration': len(delta) / fs,
            'has_slow_wave': np.max(np.abs(delta)) > 0.5e-5  # 阈值可调
        }
        
        # θ 波 (4–8 Hz)
        theta = bandpass_filter(seg, 4, 8, fs)
        seg_results['theta'] = {
            'amplitude': np.max(np.abs(theta)),
            'has_oscillation': np.std(theta) > 0.1e-5
        }
        
        # α 波 (8–13 Hz)
        alpha = bandpass_filter(seg, 8, 13, fs)
        seg_results['alpha'] = {
            'amplitude': np.max(np.abs(alpha)),
            'has_rhythm': np.std(alpha) > 0.1e-5
        }
        
        # σ 波 (12–16 Hz) - 纺锤波
        sigma = bandpass_filter(seg, 12, 16, fs)
        seg_results['sigma'] = {
            'amplitude': np.max(np.abs(sigma)),
            'spindle_count': 0,
            'spindle_duration': 0
        }
        
        # 检测纺锤波（基于包络和阈值）
        envelope = np.abs(sigma)
        threshold = np.percentile(envelope, 90)  # 90% 百分位作为阈值
        peaks, _ = find_peaks(envelope, height=threshold, distance=int(fs*0.5))  # 最小间隔0.5s
        
        if len(peaks) > 0:
            seg_results['sigma']['spindle_count'] = len(peaks)
            durations = []
            for p in peaks:
                # 找到纺锤波起止点
                left = p
                while left > 0 and envelope[left] > threshold * 0.5:
                    left -= 1
                right = p
                while right < len(envelope) and envelope[right] > threshold * 0.5:
                    right += 1
                durations.append((right - left) / fs)
            seg_results['sigma']['spindle_duration'] = np.mean(durations) if durations else 0
        
        # K-复合波（基于 δ 波 + 瞬态检测）
        k_complex = []
        for j in range(len(delta)-100):
            window = delta[j:j+100]
            if np.max(np.abs(window)) > 0.8e-5 and np.std(window) > 0.3e-5:
                k_complex.append(j)
        
        seg_results['k_complex'] = {
            'count': len(k_complex),
            'has_k_complex': len(k_complex) > 0
        }
        
        results[f'segment_{i}'] = seg_results
    
    # Step 4: 可视化（可选）
    if plot:
        fig, axes = plt.subplots(3, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, seg in enumerate(segments[:6]):  # 只画前6段
            ax = axes[i]
            t = np.arange(len(seg)) / fs
            ax.plot(t, seg, linewidth=0.5, label='Raw')
            
            # 标注纺锤波
            sigma_seg = bandpass_filter(seg, 12, 16, fs)
            envelope = np.abs(sigma_seg)
            threshold = np.percentile(envelope, 90)
            peaks, _ = find_peaks(envelope, height=threshold, distance=int(fs*0.5))
            for p in peaks:
                ax.axvspan((p-20)/fs, (p+20)/fs, color='yellow', alpha=0.3, label='Spindle' if p==peaks[0] else "")
            
            ax.set_title(f'Segment {i}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    return results


from scipy.signal import welch

def compare_denoising(
    original: np.ndarray,
    denoised: np.ndarray,
    fs: float = 100.0,
    start_sec: int = 600,  # 从第10分钟开始（典型N2期）
    duration_sec: int = 10,  # 显示10秒
    save_path: str = None
) -> dict:
    """
    对比原始信号与去噪信号的时域和频域特征
    
    参数:
    original : 原始含噪信号
    denoised : 去噪后信号
    fs : 采样率 (Hz)
    start_sec : 显示起始时间（秒）
    duration_sec : 显示时长（秒）
    save_path : 保存图像路径（如 'comparison.png'）
    
    返回:
    metrics : dict 包含关键指标
    """
    # 1. 截取时域片段
    start_idx = int(start_sec * fs)
    end_idx = start_idx + int(duration_sec * fs)
    t = np.arange(start_idx, end_idx) / fs - start_sec

    orig_seg = original[start_idx:end_idx]
    denoised_seg = denoised[start_idx:end_idx]

    # 2. 计算PSD
    nperseg = min(4096, len(original) // 4)  # 自适应分段长度
    f_orig, psd_orig = welch(original, fs=fs, nperseg=nperseg)
    f_denoised, psd_denoised = welch(denoised, fs=fs, nperseg=nperseg)

    # 3. 计算95%带宽
    def compute_95_bw(f, psd):
        cumsum_psd = np.cumsum(psd)
        threshold = 0.95 * cumsum_psd[-1]
        idx_95 = np.where(cumsum_psd >= threshold)[0][0]
        return f[idx_95]

    bw_orig = compute_95_bw(f_orig, psd_orig)
    bw_denoised = compute_95_bw(f_denoised, psd_denoised)

    # 4. 计算σ波能量（12–16 Hz）
    def compute_sigma_energy(f, psd):
        mask = (f >= 12) & (f <= 16)
        return np.sum(psd[mask]) if np.any(mask) else 0

    sigma_orig = compute_sigma_energy(f_orig, psd_orig)
    sigma_denoised = compute_sigma_energy(f_denoised, psd_denoised)

    # 5. 估算信噪比（SNR）
    # 假设原始信号 = 真实信号 + 噪声，去噪信号 ≈ 真实信号
    noise_est = original - denoised
    snr_db = 10 * np.log10(
        np.var(denoised) / (np.var(noise_est) + 1e-12)
    )

    # 6. 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('EEG Denoising Comparison', fontsize=16)

    # 时域波形（原始）
    axes[0, 0].plot(t, orig_seg, color='gray', alpha=0.8, linewidth=0.8)
    axes[0, 0].set_title('Original Signal (Time Domain)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # 时域波形（去噪）
    axes[0, 1].plot(t, denoised_seg, color='tab:blue', linewidth=1.0)
    axes[0, 1].set_title('Denoised Signal (Time Domain)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # 频域对比
    axes[1, 0].semilogy(f_orig, psd_orig, color='gray', alpha=0.7, label='Original')
    axes[1, 0].semilogy(f_denoised, psd_denoised, color='tab:blue', label='Denoised')
    axes[1, 0].axvline(bw_orig, color='red', linestyle='--', alpha=0.7, label=f'95% BW: {bw_orig:.2f} Hz')
    axes[1, 0].axvline(bw_denoised, color='green', linestyle='--', alpha=0.7, label=f'95% BW: {bw_denoised:.2f} Hz')
    axes[1, 0].set_xlim(0, 30)
    axes[1, 0].set_title('Power Spectral Density (PSD)')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('PSD [V²/Hz]')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # 重点区域：σ波（12–16 Hz）
    mask_focus = (f_denoised >= 10) & (f_denoised <= 20)
    axes[1, 1].semilogy(f_denoised[mask_focus], psd_denoised[mask_focus], color='tab:blue', linewidth=1.2)
    axes[1, 1].axvspan(12, 16, color='yellow', alpha=0.3, label='σ波 (12–16 Hz)')
    axes[1, 1].set_title('Sigma Band (12–16 Hz)')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('PSD [V²/Hz]')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

    # 7. 返回指标
    metrics = {
        'snr_db': snr_db,
        'bandwidth_95_orig': bw_orig,
        'bandwidth_95_denoised': bw_denoised,
        'sigma_energy_orig': sigma_orig,
        'sigma_energy_denoised': sigma_denoised,
        'sigma_energy_ratio': sigma_denoised / (sigma_orig + 1e-12)
    }

    return metrics