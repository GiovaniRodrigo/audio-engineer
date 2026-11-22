from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, spectrogram, find_peaks

st.set_page_config(page_title="Audio Engineering Lab", page_icon="🎙️", layout="wide")

# ---------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------
@dataclass
class AudioData:
    raw_bytes: bytes
    samples: np.ndarray          # mono float32, nominally in [-1, 1]
    sample_rate: int
    channels: int
    duration: float
    source_format: str


def decode_audio(raw: bytes, filename: str) -> AudioData:
    """Decode supported audio using libsndfile first, then FFmpeg via pydub."""
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        fmt = (filename.rsplit(".", 1)[-1] if "." in filename else "unknown").upper()
    except Exception as first_error:
        try:
            from pydub import AudioSegment
            segment = AudioSegment.from_file(io.BytesIO(raw), format=None)
            sr = int(segment.frame_rate)
            channels = int(segment.channels)
            samples = np.asarray(segment.get_array_of_samples())
            # pydub returns interleaved integer PCM; normalize by the actual
            # integer range represented by sample_width.
            scale = float(1 << (8 * segment.sample_width - 1))
            data = (samples.astype(np.float32) / scale).reshape(-1, channels)
            fmt = (filename.rsplit(".", 1)[-1] if "." in filename else "unknown").upper()
        except Exception as second_error:
            raise RuntimeError(
                "Não foi possível decodificar o áudio. Para MP3/WebM, confirme "
                "que o FFmpeg está instalado. "
                f"Leitura direta: {first_error}; conversão: {second_error}"
            ) from second_error

    if data.size == 0 or data.shape[0] == 0:
        raise ValueError("O arquivo não contém amostras de áudio.")
    if not np.all(np.isfinite(data)):
        raise ValueError("O áudio contém valores inválidos (NaN ou infinito).")

    channels = int(data.shape[1])
    # Arithmetic mean is a transparent educational downmix; it can cancel
    # content if channels have opposite polarity, so retain channel count.
    mono = data.mean(axis=1).astype(np.float32) if channels > 1 else data[:, 0].astype(np.float32)
    duration = len(mono) / sr
    return AudioData(raw, mono, int(sr), channels, duration, fmt)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def dbfs(value: float, floor: float = -120.0) -> float:
    if value <= 0:
        return floor
    return max(floor, 20.0 * math.log10(value))


def safe_audio_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    sf.write(buf, clipped, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def downsample_for_plot(x: np.ndarray, sr: int, max_points: int = 12000):
    if len(x) <= max_points:
        idx = np.arange(len(x))
    else:
        step = int(np.ceil(len(x) / max_points))
        idx = np.arange(0, len(x), step)
    return idx / sr, x[idx]


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("Audio Engineering Lab")
st.caption("Da captura acústica à representação digital: observe, processe e compare.")

with st.sidebar:
    st.header("Roteiro do laboratório")
    st.markdown(
        "1. **Aquisição** — arquivo, formato, canais e amostragem  \n"
        "2. **Análise** — forma de onda, FFT e espectrograma  \n"
        "3. **DSP** — filtros e ajustes experimentais  \n"
        "4. **Reconhecimento** — etapa ASR (próxima evolução)  \n"
        "5. **Experimentos** — salvar e comparar configurações (próxima evolução)"
    )
    st.divider()
    st.caption("O arquivo original não é alterado. Cada processamento parte de uma cópia.")

uploaded = st.file_uploader(
    "Carregue um áudio para começar",
    type=["wav", "mp3", "ogg", "flac", "webm", "m4a", "aac"],
    help="WAV/FLAC/OGG podem ser lidos diretamente quando suportados pelo libsndfile. "
         "MP3/WebM/M4A/AAC normalmente exigem FFmpeg.",
)

if uploaded is None:
    st.info("Envie um arquivo de áudio. A tela apresentará primeiro os dados de aquisição e, em seguida, as análises do sinal.")
    st.stop()

try:
    audio = decode_audio(uploaded.getvalue(), uploaded.name)
except Exception as exc:
    st.error(str(exc))
    st.stop()

x = audio.samples
sr = audio.sample_rate
peak = float(np.max(np.abs(x)))
rms_value = rms(x)
duration = audio.duration
nyquist = sr / 2

st.subheader("1. Aquisição e caracterização")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Duração", f"{duration:.2f} s")
m2.metric("Amostragem", f"{sr:,} Hz".replace(",", "."))
m3.metric("Canais", str(audio.channels))
m4.metric("Pico", f"{peak:.4f}", f"{dbfs(peak):.1f} dBFS")
m5.metric("RMS", f"{rms_value:.4f}", f"{dbfs(rms_value):.1f} dBFS")

with st.expander("O que esses dados significam?", expanded=True):
    st.markdown(
        f"""
- **Taxa de amostragem ({sr} Hz):** o conversor A/D representa o sinal com {sr:,} amostras por segundo.
- **Período de amostragem:** cada amostra está separada por aproximadamente **{1/sr*1000:.4f} ms**.
- **Frequência de Nyquist:** é **{nyquist:.1f} Hz**. Frequências acima desse limite não podem ser representadas corretamente sem filtragem antialiasing na aquisição.
- **Canais ({audio.channels}):** quantidade de canais armazenados. Para as análises iniciais, o áudio multicanal é convertido em mono pela média dos canais.
- **Pico:** maior amplitude absoluta observada. **0 dBFS** é o limite máximo representável em PCM normalizado.
- **RMS:** medida de amplitude média quadrática; relaciona-se ao nível do sinal, mas não mede, isoladamente, qualidade ou inteligibilidade.
"""
    )

st.audio(audio.raw_bytes)

if peak > 1.0:
    st.warning("Há amostras acima de ±1,0 no sinal decodificado. Verifique o formato/normalização antes de exportar.")
elif peak > 0.98:
    st.warning("O pico está próximo de 0 dBFS. Ganho adicional pode causar clipping.")
else:
    st.success("O pico não indica clipping digital evidente no sinal decodificado.")

st.divider()
st.subheader("2. Análise do sinal")
tab_wave, tab_fft, tab_spec = st.tabs(["Forma de onda", "Espectro (FFT)", "Espectrograma"])

with tab_wave:
    st.markdown("A forma de onda mostra **amplitude ao longo do tempo**. Ela ajuda a localizar pausas, picos, cortes e trechos de maior energia.")
    t, y = downsample_for_plot(x, sr)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=t, y=y, mode="lines", name="Amplitude"))
    fig.update_layout(
        xaxis_title="Tempo (s)", yaxis_title="Amplitude relativa",
        height=360, margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified"
    )
    fig.update_yaxes(range=[-1.05, 1.05])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A visualização pode reduzir o número de pontos para manter a interface responsiva; o áudio original permanece intacto.")

with tab_fft:
    st.markdown("A FFT estima **quais frequências compõem o trecho analisado**. O espectro não informa sozinho em que instante cada frequência ocorreu.")
    max_fft_seconds = min(duration, 10.0)
    start = st.slider("Início do trecho para FFT (s)", 0.0, max(0.0, duration - 0.05), 0.0, 0.05)
    default_len = min(2.0, max_fft_seconds)
    length = st.slider("Duração do trecho (s)", 0.05, max(0.05, max_fft_seconds), float(default_len), 0.05)
    a = int(start * sr)
    b = min(len(x), max(a + 1, int((start + length) * sr)))
    segment = x[a:b]
    if len(segment) < 8:
        st.warning("Trecho muito curto para uma análise espectral estável.")
    else:
        window = np.hanning(len(segment))
        spectrum = np.fft.rfft(segment * window)
        freqs = np.fft.rfftfreq(len(segment), d=1/sr)
        magnitude = np.abs(spectrum) / max(np.sum(window) / 2, 1e-12)
        magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-8))
        max_display = min(12000.0, nyquist)
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scattergl(x=freqs, y=magnitude_db, mode="lines", name="Magnitude"))
        fig_fft.update_layout(
            xaxis_title="Frequência (Hz)", yaxis_title="Magnitude aproximada (dB)",
            height=360, margin=dict(l=20, r=20, t=20, b=20)
        )
        fig_fft.update_xaxes(range=[0, max_display])
        st.plotly_chart(fig_fft, use_container_width=True)
        # Peak markers are descriptive, not a speech/noise classifier.
        peaks, _ = find_peaks(magnitude_db, distance=max(1, int(40 / (sr / len(segment)))))
        peaks = sorted(peaks, key=lambda p: magnitude_db[p], reverse=True)[:5]
        if peaks:
            st.write("Picos espectrais mais destacados (não são automaticamente ruído ou formantes):")
            st.write(", ".join(f"{freqs[p]:.0f} Hz" for p in peaks))

with tab_spec:
    st.markdown("O espectrograma mostra **como a energia por frequência muda ao longo do tempo**. É útil para observar harmônicos, ruído contínuo e eventos transitórios.")
    nperseg_max = min(2048, len(x))
    if nperseg_max < 32:
        st.warning("Áudio curto demais para exibir um espectrograma útil.")
    else:
        nperseg = st.select_slider(
            "Resolução da janela (amostras)",
            options=[v for v in [128, 256, 512, 1024, 2048] if v <= nperseg_max],
            value=max(v for v in [128, 256, 512, 1024, 2048] if v <= nperseg_max)
        )
        noverlap = int(nperseg * 0.75)
        f, tt, Sxx = spectrogram(x, fs=sr, window="hann", nperseg=nperseg,
                                 noverlap=noverlap, scaling="density", mode="psd")
        S_db = 10 * np.log10(np.maximum(Sxx, 1e-12))
        fmax = min(10000, nyquist)
        mask = f <= fmax
        fig_spec = go.Figure(data=go.Heatmap(
            x=tt, y=f[mask], z=S_db[mask, :],
            colorbar=dict(title="dB/Hz"), colorscale="Viridis"
        ))
        fig_spec.update_layout(
            xaxis_title="Tempo (s)", yaxis_title="Frequência (Hz)",
            height=430, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_spec, use_container_width=True)
        st.caption(
            f"Janela de {nperseg} amostras ({nperseg/sr*1000:.1f} ms) e sobreposição de 75%. "
            "Janelas maiores melhoram a resolução em frequência, mas reduzem a resolução temporal."
        )

st.divider()
st.subheader("3. Laboratório DSP — processamento experimental")
st.markdown("Os ajustes abaixo são aplicados em uma cópia mono. O original continua disponível para comparação.")

left, right = st.columns(2)
with left:
    hp_on = st.checkbox("Ativar filtro passa-alta", value=False)
    hp = st.slider("Frequência de corte do passa-alta (Hz)", 20, max(21, min(500, int(nyquist - 1))), 80, 5, disabled=not hp_on)
    st.caption("Atenua componentes abaixo do corte. Pode reduzir vibração e vento grave, mas cortes altos enfraquecem a voz.")
with right:
    lp_on = st.checkbox("Ativar filtro passa-baixa", value=False)
    lp_max = max(1001, min(10000, int(nyquist - 1)))
    lp_default = min(7500, lp_max)
    lp = st.slider("Frequência de corte do passa-baixa (Hz)", 1000, lp_max, lp_default, 100, disabled=not lp_on)
    st.caption("Atenua componentes acima do corte. Cortes baixos podem prejudicar fricativas e a inteligibilidade.")

c1, c2, c3 = st.columns(3)
with c1:
    denoise = st.slider("Redução espectral de ruído (%)", 0, 90, 0, 5)
    st.caption("Redução genérica; artefatos podem surgir quando a intensidade é excessiva.")
with c2:
    gain_db = st.slider("Ganho (dB)", -12, 12, 0, 1)
    st.caption("Multiplica a amplitude. Não recupera informação ausente nem separa fala e ruído.")
with c3:
    normalize = st.checkbox("Normalizar pico para −3 dBFS", value=False)
    st.caption("Ajusta o pico máximo. Não melhora, por si só, a relação sinal-ruído.")

processed = x.copy()
try:
    if hp_on:
        if hp >= nyquist:
            st.error("O corte do passa-alta precisa ser menor que a frequência de Nyquist.")
            st.stop()
        processed = sosfiltfilt(butter(4, hp, btype="highpass", fs=sr, output="sos"), processed)
    if lp_on:
        if lp >= nyquist:
            st.error("O corte do passa-baixa precisa ser menor que a frequência de Nyquist.")
            st.stop()
        processed = sosfiltfilt(butter(4, lp, btype="lowpass", fs=sr, output="sos"), processed)
    if denoise:
        import noisereduce as nr
        processed = nr.reduce_noise(y=processed, sr=sr, prop_decrease=denoise/100, stationary=True)
    processed *= 10 ** (gain_db / 20)
    if normalize:
        p = float(np.max(np.abs(processed)))
        if p > 0:
            processed *= (10 ** (-3/20)) / p
except Exception as exc:
    st.error(f"Falha no processamento: {exc}")
    st.stop()

peak_before_export = float(np.max(np.abs(processed)))
if peak_before_export > 1:
    st.warning("O processamento ultrapassou 0 dBFS. O WAV exportado será limitado; reduza o ganho para evitar distorção.")
processed_bytes = safe_audio_bytes(processed, sr)

original_col, processed_col = st.columns(2)
with original_col:
    st.write("**Original**")
    st.audio(audio.raw_bytes)
with processed_col:
    st.write("**Processado**")
    st.audio(processed_bytes, format="audio/wav")

r1, r2, r3 = st.columns(3)
r1.metric("RMS original", f"{rms(x):.5f}")
r2.metric("RMS processado", f"{rms(processed):.5f}")
r3.metric("Pico processado", f"{peak_before_export:.5f}", f"{dbfs(peak_before_export):.1f} dBFS")
st.download_button("Baixar áudio processado (WAV)", processed_bytes,
                   file_name="audio_processado.wav", mime="audio/wav")

with st.expander("Limites desta versão e interpretação correta das métricas"):
    st.markdown(
        """
- RMS e pico descrevem nível/amplitude; não provam que a fala ficou mais inteligível.
- SNR só deve ser calculada quando houver uma referência adequada de sinal e ruído ou um método explícito de estimação.
- WER/CER exigem uma transcrição de referência revisada por uma pessoa. Esta versão ainda não executa ASR.
- O filtro Butterworth de quarta ordem não tem corte abrupto; existe uma região de transição.
- A conversão multicanal para mono por média é simples e didática, mas pode causar cancelamento de fase em alguns arquivos.
"""
    )
st.caption("Versão incremental: aquisição, visualização e DSP básico. ASR e histórico de experimentos serão integrados depois.")
