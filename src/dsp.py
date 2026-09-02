import numpy as np
from scipy import signal
from typing import Dict, Any, Tuple

def compute_psd(data: np.ndarray, fs: int, nperseg: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Computes Welch's Power Spectral Density."""
    is_complex = np.iscomplexobj(data)
    seg_len = min(nperseg, len(data))
    freqs, psd = signal.welch(
        data, 
        fs=fs, 
        nperseg=seg_len, 
        return_onesided=not is_complex, 
        scaling='density'
    )
    if is_complex:
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)
    return freqs, psd

def extract_signal_metrics(data: np.ndarray, fs: int) -> Dict[str, Any]:
    """Calculates all key signal parameters deterministically."""
    freqs, psd = compute_psd(data, fs)
    psd_db = 10 * np.log10(np.maximum(psd, 1e-15))
    
    peak_idx = np.argmax(psd)
    carrier_freq = float(freqs[peak_idx])
    peak_pwr_db = float(psd_db[peak_idx])
    
    # Noise floor & SNR via baseline median estimation
    noise_floor_db = float(np.median(psd_db))
    snr_db = float(peak_pwr_db - noise_floor_db)

    # -3 dB (Half-Power) Bandwidth
    half_power = psd[peak_idx] / 2.0
    above_half = np.where(psd >= half_power)[0]
    bw_3db = float(freqs[above_half[-1]] - freqs[above_half[0]]) if len(above_half) > 1 else 0.0

    # 99% Occupied Bandwidth (OBW)
    total_power = np.sum(psd)
    if total_power > 0:
        cum_power = np.cumsum(psd) / total_power
        low_idx = np.where(cum_power >= 0.005)[0]
        high_idx = np.where(cum_power >= 0.995)[0]
        obw_99 = float(freqs[high_idx[0]] - freqs[low_idx[0]]) if len(low_idx) > 0 and len(high_idx) > 0 else 0.0
    else:
        obw_99 = 0.0

    # Peak-to-Average Power Ratio (PAPR)
    pwr = np.abs(data) ** 2
    mean_pwr = np.mean(pwr)
    papr_db = float(round(10 * np.log10(np.max(pwr) / (mean_pwr + 1e-12)), 2))

    return {
        "carrier_freq_hz": round(carrier_freq, 2),
        "occupied_bw_99_hz": round(abs(obw_99), 2),
        "bandwidth_3db_hz": round(abs(bw_3db), 2),
        "peak_power_db": round(peak_pwr_db, 2),
        "noise_floor_db": round(noise_floor_db, 2),
        "snr_db": round(snr_db, 2),
        "papr_db": papr_db
    }