import io
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(
    meta: dict, 
    metrics: dict, 
    classification: dict, 
    ai_insights: dict, 
    sig_data: np.ndarray, 
    fs: int
) -> bytes:
    """
    Builds a multi-section forensic PDF intelligence report including 
    Gemini AI analysis, parameter tables, and embedded waveform plots.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A')
    )
    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        textColor=colors.HexColor('#64748B'),
        leading=11
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155')
    )
    cell_bold = ParagraphStyle('CellB', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor('#0F172A'))
    cell_norm = ParagraphStyle('CellN', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10, textColor=colors.HexColor('#334155'))

    elements = []

    # 1. Header & Title Banner
    elements.append(Paragraph("SIGNAL INTELLIGENCE & TELEMETRY FORENSIC REPORT", title_style))
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    elements.append(Paragraph(f"Generated: {time_str} | Intelligence Engine: Gemini 2.5 Flash + Deterministic DSP Core", sub_style))
    elements.append(Spacer(1, 6))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=10))

    # 2. AI Executive Assessment Box
    threat_color = '#10B981' if ai_insights.get('threat_level') == 'LOW' else ('#F59E0B' if ai_insights.get('threat_level') == 'MEDIUM' else '#EF4444')
    
    ai_box_data = [
        [
            Paragraph(f"<b>AI SIGNAL CHARACTERIZATION:</b> {ai_insights.get('signal_characterization', 'N/A')}", cell_bold),
            Paragraph(f"<b>THREAT LEVEL: <font color='{threat_color}'>{ai_insights.get('threat_level', 'UNKNOWN')}</font></b>", cell_bold)
        ],
        [
            Paragraph(f"<b>Operational Context:</b> {ai_insights.get('operational_context', 'N/A')}", body_style),
            Paragraph(f"<b>Protocols / Standard:</b> {ai_insights.get('probable_protocol_standard', 'N/A')}<br/><b>Action:</b> {ai_insights.get('recommended_action', 'N/A')}", body_style)
        ]
    ]
    t_ai = Table(ai_box_data, colWidths=[340, 200])
    t_ai.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(t_ai)
    elements.append(Spacer(1, 10))

    # 3. Comprehensive Structured Telemetry Tables
    elements.append(Paragraph("1. Deterministic DSP & Acquisition Telemetry", h2_style))
    
    telemetry_table_data = [
        [Paragraph("Metric / Attribute", cell_bold), Paragraph("Value", cell_bold), Paragraph("Standard Metric", cell_bold), Paragraph("Observed Value", cell_bold)],
        [Paragraph("Ingestion Protocol", cell_norm), Paragraph(str(meta.get('format', 'N/A')), cell_norm), Paragraph("Peak Carrier ($f_c$)", cell_bold), Paragraph(f"{metrics['carrier_freq_hz']:,.1f} Hz", cell_norm)],
        [Paragraph("Sampling Rate ($F_s$)", cell_norm), Paragraph(f"{meta.get('sample_rate_hz', fs):,} Hz", cell_norm), Paragraph("99% Occupied BW", cell_bold), Paragraph(f"{metrics['occupied_bw_99_hz']:,.1f} Hz", cell_norm)],
        [Paragraph("Total Buffer Count", cell_norm), Paragraph(f"{meta.get('total_samples', len(sig_data)):,} samples", cell_norm), Paragraph("-3dB Bandwidth", cell_bold), Paragraph(f"{metrics['bandwidth_3db_hz']:,.1f} Hz", cell_norm)],
        [Paragraph("Signal Domain", cell_norm), Paragraph("Complex (I/Q)" if meta.get('is_complex') else "Real Audio", cell_norm), Paragraph("Estimated SNR", cell_bold), Paragraph(f"{metrics['snr_db']:.2f} dB", cell_norm)],
        [Paragraph("Duration", cell_norm), Paragraph(f"{meta.get('duration_sec', 0.0):.4f} sec", cell_norm), Paragraph("Noise Floor", cell_bold), Paragraph(f"{metrics['noise_floor_db']:.2f} dB", cell_norm)],
        [Paragraph("Classification Scheme", cell_norm), Paragraph(classification.get('predicted_signature', 'N/A'), cell_norm), Paragraph("PAPR Value", cell_bold), Paragraph(f"{metrics['papr_db']:.2f} dB", cell_norm)],
    ]
    
    t_telemetry = Table(telemetry_table_data, colWidths=[135, 135, 135, 135])
    t_telemetry.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94A3B8')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_telemetry)
    elements.append(Spacer(1, 10))

    # 4. Embedded Multi-Waveform Graphics
    elements.append(Paragraph("2. Spectral & Time-Domain Waveform Diagnostics", h2_style))

    # Generate 4-panel matplotlib waveform chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor('#FFFFFF')

    # Panel 1: Welch PSD
    is_complex = np.iscomplexobj(sig_data)
    f_w, p_w = signal.welch(sig_data, fs=fs, nperseg=min(1024, len(sig_data)), return_onesided=not is_complex)
    if is_complex:
        f_w = np.fft.fftshift(f_w)
        p_w = np.fft.fftshift(p_w)
    p_w_db = 10 * np.log10(np.maximum(p_w, 1e-15))
    ax1.plot(f_w / 1e3, p_w_db, color='#0284C7', lw=0.9)
    ax1.set_title("Power Spectral Density (PSD)", fontsize=7.5, fontweight='bold', color='#0F172A')
    ax1.set_xlabel("Frequency (kHz)", fontsize=6.5)
    ax1.set_ylabel("Power (dB/Hz)", fontsize=6.5)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.tick_params(labelsize=6)

    # Panel 2: STFT Spectrogram
    n_samples_spec = min(len(sig_data), 16384)
    f_s, t_s, zxx = signal.stft(sig_data[:n_samples_spec], fs=fs, nperseg=256)
    if is_complex:
        f_s = np.fft.fftshift(f_s)
        zxx = np.fft.fftshift(zxx, axes=0)
    spec_db = 20 * np.log10(np.abs(zxx) + 1e-12)
    ax2.imshow(spec_db, aspect='auto', origin='lower', extent=[t_s[0], t_s[-1], f_s[0]/1e3, f_s[-1]/1e3], cmap='viridis')
    ax2.set_title("STFT Waterfall / Spectrogram", fontsize=7.5, fontweight='bold', color='#0F172A')
    ax2.set_xlabel("Time (s)", fontsize=6.5)
    ax2.set_ylabel("Freq (kHz)", fontsize=6.5)
    ax2.tick_params(labelsize=6)

    # Panel 3: Time Domain Waveform
    n_show = min(800, len(sig_data))
    t_v = np.arange(n_show) / fs * 1e3
    if is_complex:
        ax3.plot(t_v, np.real(sig_data[:n_show]), color='#0284C7', lw=0.8, label='I')
        ax3.plot(t_v, np.imag(sig_data[:n_show]), color='#E11D48', lw=0.8, label='Q')
        ax3.legend(loc='upper right', fontsize=5)
    else:
        ax3.plot(t_v, sig_data[:n_show], color='#059669', lw=0.8)
    ax3.set_title("Time-Domain Baseband Scope", fontsize=7.5, fontweight='bold', color='#0F172A')
    ax3.set_xlabel("Time (ms)", fontsize=6.5)
    ax3.set_ylabel("Amplitude", fontsize=6.5)
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.tick_params(labelsize=6)

    # Panel 4: Polar Constellation or Audio Energy
    if is_complex:
        pts = sig_data[:1000]
        ax4.scatter(np.real(pts), np.imag(pts), s=2, color='#D97706', alpha=0.6)
        ax4.set_title("IQ Constellation Phase Plot", fontsize=7.5, fontweight='bold', color='#0F172A')
        ax4.set_xlabel("In-Phase (I)", fontsize=6.5)
        ax4.set_ylabel("Quadrature (Q)", fontsize=6.5)
    else:
        ax4.hist(sig_data[:8000], bins=30, color='#059669', alpha=0.7)
        ax4.set_title("Amplitude Distribution Histogram", fontsize=7.5, fontweight='bold', color='#0F172A')
        ax4.set_xlabel("Amplitude Bins", fontsize=6.5)
        ax4.set_ylabel("Count", fontsize=6.5)
    ax4.grid(True, linestyle='--', alpha=0.4)
    ax4.tick_params(labelsize=6)

    plt.tight_layout()
    plot_buf = io.BytesIO()
    plt.savefig(plot_buf, format='png', bbox_inches='tight')
    plt.close(fig)
    plot_buf.seek(0)

    elements.append(RLImage(plot_buf, width=540, height=310))
    elements.append(Spacer(1, 10))

    # 5. Strategic Footer
    footer_text = "CONFIDENTIAL & PROPRIETARY — AUTONOMOUS SIGNAL INTELLIGENCE PLATFORM (SIH 2026)"
    elements.append(Paragraph(footer_text, ParagraphStyle('Foot', fontName='Helvetica-Bold', fontSize=6.5, textColor=colors.HexColor('#94A3B8'), alignment=1)))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()