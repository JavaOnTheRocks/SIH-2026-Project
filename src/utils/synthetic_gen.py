import os
import time
import numpy as np
import soundfile as sf
from typing import Tuple, Dict, Any

class SyntheticSignalGenerator:
    """
    Synthesizes randomized .iq (RF Digital/Analog) and .wav (Acoustic) signals.
    """

    @staticmethod
    def _apply_awgn(signal: np.ndarray, target_snr_db: float) -> np.ndarray:
        is_complex = np.iscomplexobj(signal)
        sig_power = np.mean(np.abs(signal) ** 2)
        if sig_power == 0:
            return signal

        noise_power = sig_power / (10 ** (target_snr_db / 10.0))
        if is_complex:
            noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * np.sqrt(noise_power / 2.0)
        else:
            noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
        return signal + noise

    @classmethod
    def generate_iq_signal(
        cls, 
        mod_type: str = "QPSK", 
        num_samples: int = 32768, 
        fs: int = 1_000_000, 
        snr_db: float = None, 
        cfo_hz: float = None
    ) -> Tuple[np.ndarray, float, float]:
        """Generates randomized IQ signals with unique frequency offsets and SNRs."""
        mod_type = mod_type.upper()
        snr_db = float(np.random.uniform(4.0, 24.0)) if snr_db is None else snr_db
        cfo_hz = float(np.random.uniform(-fs / 4.0, fs / 4.0)) if cfo_hz is None else cfo_hz

        if mod_type == "BPSK":
            bits = np.random.randint(0, 2, num_samples)
            symbols = (2 * bits - 1).astype(np.complex64)
        elif mod_type == "QPSK":
            bits = np.random.randint(0, 4, num_samples)
            symbols = np.exp(1j * (bits * np.pi / 2.0 + np.pi / 4.0)).astype(np.complex64)
        elif mod_type == "8PSK":
            bits = np.random.randint(0, 8, num_samples)
            symbols = np.exp(1j * (bits * 2.0 * np.pi / 8.0)).astype(np.complex64)
        elif mod_type == "16QAM":
            i_b = 2 * np.random.randint(0, 4, num_samples) - 3
            q_b = 2 * np.random.randint(0, 4, num_samples) - 3
            symbols = ((i_b + 1j * q_b) / np.sqrt(10.0)).astype(np.complex64)
        elif mod_type == "64QAM":
            i_b = 2 * np.random.randint(0, 8, num_samples) - 7
            q_b = 2 * np.random.randint(0, 8, num_samples) - 7
            symbols = ((i_b + 1j * q_b) / np.sqrt(42.0)).astype(np.complex64)
        elif mod_type == "BFSK":
            bits = np.random.randint(0, 2, num_samples)
            freq_dev = np.random.uniform(20000, 60000)
            freqs = np.where(bits == 1, freq_dev, -freq_dev)
            phase = 2.0 * np.pi * np.cumsum(freqs) / fs
            symbols = np.exp(1j * phase).astype(np.complex64)
        elif mod_type == "WBFM":
            t = np.arange(num_samples) / fs
            fm_rate = np.random.uniform(800, 3000)
            modulating_msg = np.sin(2.0 * np.pi * fm_rate * t)
            phase = 2.0 * np.pi * 50000.0 * np.cumsum(modulating_msg) / fs
            symbols = np.exp(1j * phase).astype(np.complex64)
        elif mod_type == "AM-DSB":
            t = np.arange(num_samples) / fs
            mod_f = np.random.uniform(1000, 4000)
            modulating_msg = np.sin(2.0 * np.pi * mod_f * t)
            carrier_env = 1.0 + 0.7 * modulating_msg
            symbols = carrier_env.astype(np.complex64)
        else:
            raise ValueError(f"Unsupported modulation scheme: {mod_type}")

        # Apply Random CFO
        t = np.arange(num_samples) / fs
        symbols = symbols * np.exp(1j * 2.0 * np.pi * cfo_hz * t)

        # Apply Noise
        noisy_iq = cls._apply_awgn(symbols, snr_db).astype(np.complex64)
        return noisy_iq, round(cfo_hz, 1), round(snr_db, 1)

    @classmethod
    def save_iq_file(cls, filepath: str, iq_array: np.ndarray, dtype_str: str = "float32"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if dtype_str == "float32":
            interleaved = np.empty((iq_array.size * 2,), dtype=np.float32)
            interleaved[0::2] = np.real(iq_array)
            interleaved[1::2] = np.imag(iq_array)
            interleaved.tofile(filepath)
        elif dtype_str == "int16":
            scaled_i = np.clip(np.real(iq_array) * 32767.0, -32768, 32767).astype(np.int16)
            scaled_q = np.clip(np.imag(iq_array) * 32767.0, -32768, 32767).astype(np.int16)
            interleaved = np.empty((iq_array.size * 2,), dtype=np.int16)
            interleaved[0::2] = scaled_i
            interleaved[1::2] = scaled_q
            interleaved.tofile(filepath)

    @classmethod
    def generate_wav_signal(
        cls, 
        audio_type: str = "SPEECH", 
        duration_sec: float = 2.0, 
        fs: int = 44100, 
        snr_db: float = None
    ) -> Tuple[np.ndarray, float]:
        audio_type = audio_type.upper()
        snr_db = float(np.random.uniform(5.0, 25.0)) if snr_db is None else snr_db
        num_samples = int(fs * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)

        if audio_type == "SPEECH":
            f0 = np.random.uniform(120.0, 240.0)
            harmonics = (
                0.50 * np.sin(2.0 * np.pi * f0 * t) +
                0.25 * np.sin(2.0 * np.pi * 2.0 * f0 * t) +
                0.15 * np.sin(2.0 * np.pi * 3.0 * f0 * t)
            )
            envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * np.random.uniform(3, 6) * t))
            signal_out = harmonics * envelope
        elif audio_type == "SIREN":
            base_f = np.random.uniform(800.0, 1400.0)
            dev_f = np.random.uniform(300.0, 600.0)
            instant_freq = base_f + dev_f * np.sin(2.0 * np.pi * np.random.uniform(1.5, 4.0) * t)
            phase = 2.0 * np.pi * np.cumsum(instant_freq) / fs
            signal_out = 0.6 * np.sin(phase)
        elif audio_type == "BEACON":
            f_tone = np.random.uniform(700.0, 1800.0)
            pulse_gate = (np.sin(2.0 * np.pi * np.random.uniform(2.0, 5.0) * t) > 0.3).astype(float)
            signal_out = 0.5 * np.sin(2.0 * np.pi * f_tone * t) * pulse_gate
        elif audio_type == "STATIC":
            noise = np.random.randn(num_samples)
            signal_out = np.convolve(noise, np.ones(5) / 5.0, mode='same') * 0.4
        else:
            raise ValueError(f"Unsupported audio type: {audio_type}")

        noisy_audio = cls._apply_awgn(signal_out, snr_db)
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 0:
            noisy_audio = 0.9 * (noisy_audio / max_val)
        return noisy_audio.astype(np.float32), round(snr_db, 1)

    @classmethod
    def save_wav_file(cls, filepath: str, audio_array: np.ndarray, fs: int = 44100):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        sf.write(filepath, audio_array, fs)


def build_full_sample_library(output_dir: str = "data/samples") -> int:
    """
    Cleans previous sample files and generates freshly randomized signals.
    """
    gen = SyntheticSignalGenerator()
    os.makedirs(output_dir, exist_ok=True)

    # Clean old samples to force browser refresh
    for f in os.listdir(output_dir):
        if f.endswith(('.iq', '.raw', '.wav')):
            try:
                os.remove(os.path.join(output_dir, f))
            except Exception:
                pass

    rf_mods = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "BFSK", "WBFM", "AM-DSB"]
    for mod in rf_mods:
        iq_data, cfo, snr = gen.generate_iq_signal(mod_type=mod, num_samples=32768, fs=1_000_000)
        fname = f"rf_{mod.lower().replace('-', '_')}_{int(snr)}dB.iq"
        gen.save_iq_file(os.path.join(output_dir, fname), iq_data, dtype_str="float32")

    audio_types = ["SPEECH", "SIREN", "BEACON", "STATIC"]
    for atype in audio_types:
        wav_data, snr = gen.generate_wav_signal(audio_type=atype, duration_sec=2.0, fs=44100)
        fname = f"audio_{atype.lower()}_{int(snr)}dB.wav"
        gen.save_wav_file(os.path.join(output_dir, fname), wav_data, fs=44100)

    return len(rf_mods) + len(audio_types)


if __name__ == "__main__":
    count = build_full_sample_library()
    print(f"Generated {count} randomized benchmark files.")