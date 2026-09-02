import json
from typing import Dict, Any, Optional
from google import genai
from google.genai import types

def analyze_signal_with_gemini(
    api_key: str, 
    meta: Dict[str, Any], 
    metrics: Dict[str, Any], 
    classification: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Feeds deterministic DSP metrics and neural classifier probabilities into Gemini
    to determine the signal's operational nature, protocol family, and transmission intent.
    """
    if not api_key:
        return _get_fallback_ai_data(meta, metrics, classification)

    try:
        client = genai.Client(api_key=api_key)
        
        # Build prompt from telemetry, neural confidence, and statistical cumulants
        prompt = f"""
You are an expert Electronic Intelligence (ELINT), Signal Intelligence (SIGINT), and Audio Forensics Analyst.

Analyze the following incoming transmission telemetry and determine its technical domain, standard protocol family, and operational context:

--- INGESTION & TIME METRICS ---
- Signal Ingestion Domain: {'Complex Radio Frequency (I/Q Baseband)' if meta.get('is_complex') else 'Acoustic / Demodulated Audio'}
- Ingestion Format: {meta.get('format', 'N/A')}
- Sampling Rate (Fs): {meta.get('sample_rate_hz', 0):,} Hz
- Total Buffers: {meta.get('total_samples', 0):,} samples
- Transmission Window: {meta.get('duration_sec', 0.0):.4f} seconds

--- DETERMINISTIC DSP TELEMETRY ---
- Carrier Center Frequency (fc): {metrics.get('carrier_freq_hz', 0.0):,.2f} Hz
- 99% Occupied Bandwidth (OBW): {metrics.get('occupied_bw_99_hz', 0.0):,.2f} Hz
- Half-Power Bandwidth (-3dB): {metrics.get('bandwidth_3db_hz', 0.0):,.2f} Hz
- Signal-to-Noise Ratio (SNR): {metrics.get('snr_db', 0.0):.2f} dB
- Channel Noise Floor: {metrics.get('noise_floor_db', 0.0):.2f} dB
- Peak-to-Average Power Ratio (PAPR): {metrics.get('papr_db', 0.0):.2f} dB

--- NEURAL CLASSIFIER & STATISTICAL PRIORS ---
- Top Predicted Scheme: {classification.get('predicted_signature', 'Unknown')}
- Confidence Rating: {classification.get('confidence', 0.0)*100:.1f}%
- Candidate Distribution: {classification.get('top_3_predictions', [])}
- Physical DSP Priors: {classification.get('dsp_priors', {})}

Perform a forensic assessment and return strictly a valid JSON object matching this schema:
{{
  "signal_characterization": "Concise definition of the transmission (e.g., 'High-Order Quadrature Amplitude Modulated Digital Stream' or 'Wideband Acoustic Warning Siren').",
  "probable_protocol_standard": "Specific likely standards (e.g., DVB-S2, IEEE 802.11 Wi-Fi, MIL-STD Tactical Link, APRS, VHF Marine, Air-Raid Siren, LoRa/FSK Telemetry).",
  "operational_context": "Explanation of what this data is typically used for in practical deployments (e.g., Drone Telemetry C2, Satellite Uplink, Emergency Warning, Broadcast FM).",
  "threat_level": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
  "spectral_efficiency_notes": "Insight into bandwidth utilization vs noise floor and PAPR.",
  "recommended_action": "Operational next step (e.g., 'Demodulate constellation via QAM symbols', 'Deploy Notch Filter at fc', 'Log acoustic signature')."
}}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return _get_fallback_ai_data(meta, metrics, classification, error_msg=str(e))


def _get_fallback_ai_data(
    meta: Dict[str, Any], 
    metrics: Dict[str, Any], 
    classification: Dict[str, Any], 
    error_msg: Optional[str] = None
) -> Dict[str, Any]:
    """Offline heuristic engine mapping DSP metrics to operational signal profiles."""
    pred = classification.get('predicted_signature', 'Unknown')
    is_rf = meta.get('is_complex', False) or any(k in pred for k in ['PSK', 'QAM', 'FSK', 'FM', 'AM'])
    snr = metrics.get('snr_db', 0.0)

    # Heuristic mapping for standard RF and acoustic schemas
    if "16QAM" in pred or "64QAM" in pred:
        char = "High-Density Constellation Digital Carrier"
        proto = "DVB-C / IEEE 802.11 (Wi-Fi) / Microwave Point-to-Point Backhaul"
        context = "High-throughput digital data payload conveying multiplexed video or broadband packets."
        threat = "MEDIUM" if snr > 12 else "LOW"
        action = "Equalize multipath channel and execute carrier phase lock for symbol demodulation."
    elif "QPSK" in pred or "8PSK" in pred:
        char = "Phase Shift Keyed Coherent Digital Stream"
        proto = "DVB-S2 Satellite Telemetry / CDMA / MIL-STD-188 Satellite Link"
        context = "Power-efficient digital uplink/downlink robust against phase noise."
        threat = "MEDIUM"
        action = "Engage Costas Loop carrier recovery to extract bitstream."
    elif "BPSK" in pred:
        char = "Binary Phase Shift Keyed Spread-Spectrum Transmission"
        proto = "GPS L1 C/A / Deep Space Telemetry / Military Datalink Sync Preamble"
        context = "Low-SNR command-and-control signaling and synchronization beacons."
        threat = "LOW"
        action = "Correlate with PRN codes to detect frame headers."
    elif "FSK" in pred:
        char = "Frequency Shift Keyed Narrowband Stream"
        proto = "LoRa Baseband / Paging / APRS / SCADA Industrial Telemetry"
        context = "Long-range, low-power telemetry packets from remote sensor nodes or drones."
        threat = "LOW"
        action = "Run dual-filter discriminator for tone extraction."
    elif "SIREN" in pred:
        char = "Dynamic Frequency-Swept Acoustic Tone"
        proto = "Civil Emergency Alert / Tactical Siren / Warning Horn"
        context = "Audible emergency signaling or perimeter defense warning broadcast."
        threat = "HIGH"
        action = "Log alert timeline and alert regional monitoring centers."
    else:
        char = f"Analyzed Baseband Signature ({pred})"
        proto = "Standard Baseband Transmission"
        context = "General telemetry or monitored channel communication."
        threat = "LOW"
        action = "Continue spectral surveillance."

    return {
        "signal_characterization": char,
        "probable_protocol_standard": proto,
        "operational_context": context,
        "threat_level": threat,
        "spectral_efficiency_notes": f"Occupied BW: {metrics.get('occupied_bw_99_hz', 0):,.1f} Hz with {snr:.1f} dB SNR.",
        "recommended_action": action,
        "api_note": f"Offline Mode: {error_msg}" if error_msg else "Offline heuristic analysis."
    }