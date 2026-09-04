import numpy as np


def _normalize(iq):
    """Remove DC offset and normalize average signal power."""
    x = np.asarray(iq, dtype=np.complex128)

    # Remove DC
    x = x - np.mean(x)

    # Normalize power
    power = np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12
    x = x / power

    return x


def _estimate_and_correct_cfo(iq, order=4):
    """
    Simple blind carrier-frequency-offset / phase correction.

    Uses M-th power method, useful for PSK and square QAM
    prototype signals.
    """

    x = np.asarray(iq, dtype=np.complex128)

    if len(x) < 4:
        return x, 0.0

    z = x ** order

    phase_diff = np.angle(
        np.mean(z[1:] * np.conj(z[:-1]))
    )

    estimated_phase_step = phase_diff / order

    n = np.arange(len(x))

    corrected = x * np.exp(
        -1j * estimated_phase_step * n
    )

    return corrected, float(estimated_phase_step)


def _bits_to_string(bit_array):
    return "".join(str(int(b)) for b in bit_array)


# ============================================================
# BPSK
# ============================================================

def demodulate_bpsk(iq):

    x = _normalize(iq)

    # BPSK squaring removes data modulation
    x_corrected, phase_step = _estimate_and_correct_cfo(
        x,
        order=2
    )

    # Hard decision
    bits = (np.real(x_corrected) > 0).astype(np.uint8)

    return {
        "modulation": "BPSK",
        "bits": bits,
        "bitstream": _bits_to_string(bits),
        "symbols": np.real(x_corrected),
        "estimated_phase_step": phase_step
    }


# ============================================================
# QPSK
# ============================================================

def demodulate_qpsk(iq):

    x = _normalize(iq)

    x_corrected, phase_step = _estimate_and_correct_cfo(
        x,
        order=4
    )

    # Hard decision on I and Q
    i_bits = (np.real(x_corrected) > 0).astype(np.uint8)
    q_bits = (np.imag(x_corrected) > 0).astype(np.uint8)

    bits = np.empty(len(x_corrected) * 2, dtype=np.uint8)

    bits[0::2] = i_bits
    bits[1::2] = q_bits

    return {
        "modulation": "QPSK",
        "bits": bits,
        "bitstream": _bits_to_string(bits),
        "symbols": x_corrected,
        "estimated_phase_step": phase_step
    }


# ============================================================
# 8PSK
# ============================================================

def demodulate_8psk(iq):

    x = _normalize(iq)

    x_corrected, phase_step = _estimate_and_correct_cfo(
        x,
        order=8
    )

    phase = np.angle(x_corrected)

    # Map phase into 8 sectors
    symbol_index = np.mod(
        np.round(
            (phase / (2 * np.pi)) * 8
        ).astype(int),
        8
    )

    bits = []

    for symbol in symbol_index:
        bits.extend([
            (symbol >> 2) & 1,
            (symbol >> 1) & 1,
            symbol & 1
        ])

    bits = np.array(bits, dtype=np.uint8)

    return {
        "modulation": "8PSK",
        "bits": bits,
        "bitstream": _bits_to_string(bits),
        "symbols": x_corrected,
        "estimated_phase_step": phase_step
    }


# ============================================================
# 16-QAM
# ============================================================

def demodulate_16qam(iq):

    x = _normalize(iq)

    # Square QAM supports useful 4th-power correction
    x_corrected, phase_step = _estimate_and_correct_cfo(
        x,
        order=4
    )

    # Normalize approximately to 16-QAM levels
    scale = np.sqrt(10)

    i_val = np.real(x_corrected) * scale
    q_val = np.imag(x_corrected) * scale

    # Nearest ideal levels
    levels = np.array([-3, -1, 1, 3])

    i_index = np.argmin(
        np.abs(i_val[:, None] - levels[None, :]),
        axis=1
    )

    q_index = np.argmin(
        np.abs(q_val[:, None] - levels[None, :]),
        axis=1
    )

    bits = []

    # Simple 2-bit representation per I and Q component
    for i, q in zip(i_index, q_index):

        bits.extend([
            (i >> 1) & 1,
            i & 1,
            (q >> 1) & 1,
            q & 1
        ])

    bits = np.array(bits, dtype=np.uint8)

    return {
        "modulation": "16QAM",
        "bits": bits,
        "bitstream": _bits_to_string(bits),
        "symbols": x_corrected,
        "estimated_phase_step": phase_step
    }


# ============================================================
# BFSK
# ============================================================

def demodulate_bfsk(iq):

    x = _normalize(iq)

    phase = np.unwrap(np.angle(x))

    # Instantaneous frequency proxy
    inst_freq = np.diff(phase)

    # Remove average frequency offset
    inst_freq_centered = inst_freq - np.median(inst_freq)

    # Positive deviation = 1
    # Negative deviation = 0
    bits = (
        inst_freq_centered > 0
    ).astype(np.uint8)

    return {
        "modulation": "BFSK",
        "bits": bits,
        "bitstream": _bits_to_string(bits),
        "symbols": inst_freq_centered,
        "estimated_phase_step": float(np.median(inst_freq))
    }


# ============================================================
# UNIVERSAL ENTRY POINT
# ============================================================

def demodulate_signal(
    data,
    modulation,
    is_complex=True
):

    modulation = str(modulation).upper().replace("-", "")

    if not is_complex:
        return {
            "success": False,
            "message": (
                "Bitstream demodulation requires a digital "
                "RF/IQ signal. This WAV input is treated as "
                "an acoustic waveform."
            )
        }

    if modulation == "BPSK":

        result = demodulate_bpsk(data)

    elif modulation == "QPSK":

        result = demodulate_qpsk(data)

    elif modulation == "8PSK":

        result = demodulate_8psk(data)

    elif modulation in ["16QAM", "16-QAM"]:

        result = demodulate_16qam(data)

    elif modulation in ["BFSK", "2FSK", "FSK"]:

        result = demodulate_bfsk(data)

    else:

        return {
            "success": False,
            "message": (
                f"Demodulation prototype currently supports "
                f"BPSK, QPSK, 8PSK, 16QAM and BFSK. "
                f"Detected: {modulation}"
            )
        }

    result["success"] = True

    return result
