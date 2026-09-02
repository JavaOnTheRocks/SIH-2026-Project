import numpy as np
import soundfile as sf
from typing import Tuple, Dict, Any

def parse_wav(file_obj_or_path) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Parses audio waveforms from file paths or uploaded file buffers.
    """
    data, sample_rate = sf.read(file_obj_or_path, dtype='float32')
    if data.ndim > 1:
        data = np.mean(data, axis=1) # Convert multi-channel to mono
        
    duration = len(data) / sample_rate
    meta = {
        "format": "WAV (Audio)",
        "sample_rate_hz": int(sample_rate),
        "total_samples": len(data),
        "duration_sec": float(round(duration, 4)),
        "is_complex": False
    }
    return data, int(sample_rate), meta

def parse_iq(file_obj_or_path, sample_rate: int = 1_000_000, dtype_str: str = "float32") -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Parses interleaved IQ binary buffers: [I0, Q0, I1, Q1, ...] -> I + jQ.
    """
    if hasattr(file_obj_or_path, "read"):
        raw_bytes = file_obj_or_path.read()
        np_dtype = np.float32 if dtype_str == "float32" else np.int16
        raw_data = np.frombuffer(raw_bytes, dtype=np_dtype)
    else:
        np_dtype = np.float32 if dtype_str == "float32" else np.int16
        raw_data = np.fromfile(file_obj_or_path, dtype=np_dtype)

    if dtype_str == "int16":
        raw_data = raw_data.astype(np.float32) / 32768.0

    if len(raw_data) % 2 != 0:
        raw_data = raw_data[:-1]

    i_samples = raw_data[0::2]
    q_samples = raw_data[1::2]
    complex_iq = i_samples + 1j * q_samples

    duration = len(complex_iq) / sample_rate
    meta = {
        "format": f"IQ ({dtype_str})",
        "sample_rate_hz": int(sample_rate),
        "total_samples": len(complex_iq),
        "duration_sec": float(round(duration, 4)),
        "is_complex": True
    }
    return complex_iq, int(sample_rate), meta