from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from scipy.signal import butter, find_peaks, sosfiltfilt, spectrogram


@dataclass
class AudioData:
    raw_bytes: bytes
    samples: np.ndarray
    sample_rate: int
    channels: int
    duration: float
    source_format: str


def decode_audio(raw: bytes, filename: str) -> AudioData:
    """Decode using libsndfile, falling back to pydub/FFmpeg."""
    try:
        data, sample_rate = sf.read(
            io.BytesIO(raw), dtype="float32", always_2d=True
        )
        source_format = filename.rsplit(".", 1)[-1].upper() if "." in filename else "UNKNOWN"
    except Exception as first_error:
        try:
            from pydub import AudioSegment

            segment = AudioSegment.from_file(io.BytesIO(raw))
            sample_rate = int(segment.frame_rate)
            channels = int(segment.channels)
            pcm = np.asarray(segment.get_array_of_samples())
            scale = float(1 << (8 * segment.sample_width - 1))
            data = (pcm.astype(np.float32) / scale).reshape(-1, channels)
            source_format = filename.rsplit(".", 1)[-1].upper() if "." in filename else "UNKNOWN"
        except Exception as second_error:
            raise RuntimeError(
                "Não foi possível decodificar o áudio. Para MP3/WebM/M4A, "
                "instale o FFmpeg. "
                f"Leitura direta: {first_error}; conversão: {second_error}"
            ) from second_error

    if data.size == 0 or data.shape[0] == 0:
        raise ValueError("O arquivo não contém amostras.")
    if not np.isfinite(data).all():
        raise ValueError("O áudio contém valores inválidos.")

    channels = int(data.shape[1])
    mono = data.mean(axis=1) if channels > 1 else data[:, 0]
    mono = np.asarray(mono, dtype=np.float32)
    return AudioData(
        raw_bytes=raw,
        samples=mono,
        sample_rate=int(sample_rate),
        channels=channels,
        duration=len(mono) / int(sample_rate),
        source_format=source_format,
    )


def rms(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def to_db(value: float, floor: float = -120.0) -> float:
    return floor if value <= 0 else max(floor, 20.0 * math.log10(value))


def wav_bytes(signal: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    # PCM16 cannot represent values outside [-1, 1].
    sf.write(
        buffer,
        np.clip(signal, -1.0, 1.0).astype(np.float32),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


def plot_samples(signal: np.ndarray, sample_rate: int, limit: int = 15000):
    stride = max(1, int(np.ceil(len(signal) / limit)))
    indices = np.arange(0, len(signal), stride)
    return indices / sample_rate, signal[indices]


st.set_page_config(
    page_title="Audio Engineering Lab",
    page_icon="🎙️",
    layout="wide",
)
st.title("Audio Engineering Lab")
st.caption(
    "Da aquisição acústica à representação digital: observe, processe e compare."
)

with st.sidebar:
    st.header("Roteiro")
    st.markdown(
        "1. **Aquisição** — formato, canais e amostragem\n"
        "2. **Análise** — forma de onda, FFT e espectrograma\n"
        "3. **DSP** — filtros e ajustes experimentais\n"
        "4. **ASR** — reconhecimento de fala (próxima etapa)\n"
        "5. **Experimentos** — histórico e comparação (próxima etapa)"
    )
    st.divider()
    st.caption("O original é preservado. Cada ajuste parte de uma cópia.")

uploaded = st.file_uploader(
    "Carregue um áudio",
    type=["wav", "mp3", "ogg", "flac", "webm", "m4a", "aac"],
    help="MP3, WebM, M4A e AAC normalmente exigem FFmpeg.",
)
if uploaded is None:
    st.info("Envie um arquivo para iniciar a análise.")
    st.stop()

try:
    audio = decode_audio(uploaded.getvalue(), uploaded.name)
except Exception as exc:
    st.error(str(exc))
    st.stop()

original = audio.samples
sr = audio.sample_rate
nyquist = sr / 2
peak = float(np.max(np.abs(original)))
level_rms = rms(original)

st.subheader("1. Aquisição e caracterização")
metrics = st.columns(5)
metrics[0].metric("Duração", f"{audio.duration:.2f} s")
metrics[1].metric("Taxa de amostragem", f"{sr:,} Hz".replace(",", "."))
metrics[2].metric("Canais", str(audio.channels))
metrics[3].metric("Pico", f"{peak:.4f}", f"{to_db(peak):.1f} dBFS")
metrics[4].metric("RMS", f"{level_rms:.4f}", f"{to_db(level_rms):.1f} dBFS")

with st.expander("Entenda os parâmetros de aquisição", expanded=True):
    st.markdown(
        f"""
- **Taxa de amostragem:** {sr:,} amostras por segundo.
- **Período entre amostras:** {1000 / sr:.4f} ms.
- **Frequência de Nyquist:** {nyquist:.1f} Hz; é metade da taxa de amostragem.
- **Canais:** {audio.channels}. Para as análises iniciais, o áudio é convertido para mono pela média dos canais.
- **Pico:** maior amplitude absoluta. Em PCM normalizado, 0 dBFS representa o limite digital.
- **RMS:** nível quadrático médio. Não mede sozinho a inteligibilidade nem a qualidade da fala.
"""
    )

st.audio(audio.raw_bytes)
if peak > 1.0:
    st.warning("Há amostras acima de ±1,0 no sinal decodificado.")
elif peak >= 0.98:
    st.warning("O pico está próximo de 0 dBFS; ganho adicional pode causar clipping.")
else:
    st.success("Não há indicação de clipping digital pelo pico medido.")

st.divider()
st.subheader("2. Análise do sinal")
wave_tab, fft_tab, spec_tab = st.tabs(
    ["Forma de onda", "Espectro (FFT)", "Espectrograma"]
)

with wave_tab:
    st.write("Amplitude ao longo do tempo. Ajuda a localizar pausas, picos e trechos de maior energia.")
    time_axis, values = plot_samples(original, sr)
    fig = go.Figure(go.Scattergl(x=time_axis, y=values, mode="lines", name="Áudio"))
    fig.update_layout(
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude relativa",
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_yaxes(range=[-1.05, 1.05])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A exibição pode reduzir pontos para manter a interface rápida; o áudio não é alterado.")

with fft_tab:
    st.write("A FFT estima as frequências presentes no trecho selecionado, mas não mostra quando cada frequência ocorreu.")
    max_start = max(0.0, audio.duration - 0.05)
    start = st.slider("Início do trecho (s)", 0.0, max_start, 0.0, 0.05)
    max_length = min(10.0, audio.duration - start)
    if max_length < 0.05:
        st.warning("O áudio é curto demais para selecionar um trecho estável.")
    else:
        default_length = min(2.0, max_length)
        length = st.slider(
            "Duração do trecho (s)",
            0.05,
            float(max_length),
            float(default_length),
            0.05,
        )
        first = int(start * sr)
        last = min(len(original), max(first + 1, int((start + length) * sr)))
        segment = original[first:last]
        if len(segment) < 8:
            st.warning("Trecho muito curto para uma análise espectral útil.")
        else:
            window = np.hanning(len(segment))
            spectrum = np.fft.rfft(segment * window)
            frequencies = np.fft.rfftfreq(len(segment), d=1 / sr)
            magnitude = np.abs(spectrum) / max(np.sum(window) / 2, 1e-12)
            magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-8))
            fig_fft = go.Figure(
                go.Scattergl(
                    x=frequencies,
                    y=magnitude_db,
                    mode="lines",
                    name="Magnitude",
                )
            )
            fig_fft.update_layout(
                xaxis_title="Frequência (Hz)",
                yaxis_title="Magnitude aproximada (dB)",
                height=360,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            fig_fft.update_xaxes(range=[0, min(12000, nyquist)])
            st.plotly_chart(fig_fft, use_container_width=True)
            peak_indices, _ = find_peaks(
                magnitude_db,
                distance=max(1, int(40 / (sr / len(segment)))),
            )
            strongest = sorted(
                peak_indices, key=lambda index: magnitude_db[index], reverse=True
            )[:5]
            if strongest:
                st.caption(
                    "Picos destacados: "
                    + ", ".join(f"{frequencies[index]:.0f} Hz" for index in strongest)
                    + ". Eles não são automaticamente ruído ou formantes."
                )

with spec_tab:
    st.write("O espectrograma mostra como a energia por frequência varia ao longo do tempo.")
    maximum_window = min(2048, len(original))
    window_options = [size for size in (128, 256, 512, 1024, 2048) if size <= maximum_window]
    if not window_options or maximum_window < 32:
        st.warning("Áudio curto demais para um espectrograma útil.")
    else:
        selected_window = st.select_slider(
            "Tamanho da janela (amostras)",
            options=window_options,
            value=window_options[-1],
        )
        frequencies, times, power = spectrogram(
            original,
            fs=sr,
            window="hann",
            nperseg=selected_window,
            noverlap=int(selected_window * 0.75),
            scaling="density",
            mode="psd",
        )
        power_db = 10 * np.log10(np.maximum(power, 1e-12))
        frequency_mask = frequencies <= min(10000, nyquist)
        fig_spec = go.Figure(
            go.Heatmap(
                x=times,
                y=frequencies[frequency_mask],
                z=power_db[frequency_mask, :],
                colorbar=dict(title="dB/Hz"),
                colorscale="Viridis",
            )
        )
        fig_spec.update_layout(
            xaxis_title="Tempo (s)",
            yaxis_title="Frequência (Hz)",
            height=430,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_spec, use_container_width=True)
        st.caption(
            f"Janela de {selected_window} amostras "
            f"({1000 * selected_window / sr:.1f} ms), com 75% de sobreposição. "
            "Janelas maiores aumentam a resolução em frequência e reduzem a resolução temporal."
        )

st.divider()
st.subheader("3. Laboratório DSP")
st.write("Os controles são aplicados a uma cópia mono. O áudio original permanece intacto.")

left, right = st.columns(2)
with left:
    highpass_enabled = st.checkbox("Ativar filtro passa-alta", value=False)
    highpass_cutoff = st.slider(
        "Corte passa-alta (Hz)",
        min_value=20,
        max_value=max(21, min(500, int(nyquist) - 1)),
        value=min(80, max(20, min(500, int(nyquist) - 1))),
        step=5,
        disabled=not highpass_enabled,
    )
    st.caption("Atenua graves. Pode reduzir vibração e vento, mas cortes altos enfraquecem a voz.")
with right:
    lowpass_enabled = st.checkbox("Ativar filtro passa-baixa", value=False)
    lowpass_max = max(1001, min(10000, int(nyquist) - 1))
    lowpass_cutoff = st.slider(
        "Corte passa-baixa (Hz)",
        min_value=1000,
        max_value=lowpass_max,
        value=min(7500, lowpass_max),
        step=100,
        disabled=not lowpass_enabled,
    )
    st.caption("Atenua agudos. Cortes baixos podem prejudicar fricativas e inteligibilidade.")

col_a, col_b, col_c = st.columns(3)
with col_a:
    reduction = st.slider("Redução espectral de ruído (%)", 0, 90, 0, 5)
    st.caption("Redução genérica; valores altos podem criar artefatos.")
with col_b:
    gain_db = st.slider("Ganho (dB)", -12, 12, 0, 1)
    st.caption("Altera o nível; não recupera informação perdida.")
with col_c:
    normalize_peak = st.checkbox("Normalizar pico para −3 dBFS", value=False)
    st.caption("Ajusta o nível de pico, mas não melhora a relação sinal-ruído por si só.")

processed = original.copy()
try:
    if highpass_enabled:
        if highpass_cutoff >= nyquist:
            st.error("O corte passa-alta deve ser menor que Nyquist.")
            st.stop()
        sos = butter(4, highpass_cutoff, btype="highpass", fs=sr, output="sos")
        processed = sosfiltfilt(sos, processed)

    if lowpass_enabled:
        if lowpass_cutoff >= nyquist:
            st.error("O corte passa-baixa deve ser menor que Nyquist.")
            st.stop()
        sos = butter(4, lowpass_cutoff, btype="lowpass", fs=sr, output="sos")
        processed = sosfiltfilt(sos, processed)

    if reduction > 0:
        import noisereduce as nr

        processed = nr.reduce_noise(
            y=processed,
            sr=sr,
            prop_decrease=reduction / 100.0,
            stationary=True,
        )

    processed *= 10 ** (gain_db / 20)
    if normalize_peak:
        current_peak = float(np.max(np.abs(processed)))
        if current_peak > 0:
            processed *= (10 ** (-3 / 20)) / current_peak
except Exception as exc:
    st.error(f"Falha no processamento: {exc}")
    if reduction > 0:
        st.info("Confirme a instalação de noisereduce pelo requirements.txt.")
    st.stop()

processed_peak = float(np.max(np.abs(processed)))
if processed_peak > 1.0:
    st.warning(
        "O processamento ultrapassou 0 dBFS. O WAV será limitado na exportação; "
        "reduza o ganho para evitar distorção."
    )

processed_wav = wav_bytes(processed, sr)
original_col, processed_col = st.columns(2)
with original_col:
    st.markdown("**Original**")
    st.audio(audio.raw_bytes)
with processed_col:
    st.markdown("**Processado**")
    st.audio(processed_wav, format="audio/wav")

levels = st.columns(3)
levels[0].metric("RMS original", f"{rms(original):.5f}")
levels[1].metric("RMS processado", f"{rms(processed):.5f}")
levels[2].metric("Pico processado", f"{processed_peak:.5f}", f"{to_db(processed_peak):.1f} dBFS")
st.download_button(
    "Baixar áudio processado (WAV)",
    data=processed_wav,
    file_name="audio_processado.wav",
    mime="audio/wav",
)

with st.expander("Limites e interpretação correta"):
    st.markdown(
        """
- RMS e pico descrevem amplitude; não provam melhoria da inteligibilidade.
- SNR requer uma referência adequada de sinal e ruído ou um estimador explicitamente descrito.
- WER/CER exigem uma transcrição de referência revisada por uma pessoa.
- O filtro Butterworth de quarta ordem tem uma região de transição; não é um corte abrupto.
- A conversão multicanal para mono pela média pode causar cancelamento de fase.
"""
    )

st.caption(
    "Versão incremental: aquisição, análise visual e DSP básico. "
    "ASR e histórico persistente de experimentos serão integrados em etapas futuras."
)
