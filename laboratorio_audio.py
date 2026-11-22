import io
import numpy as np
import streamlit as st
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

st.set_page_config(page_title="Laboratório de Áudio", layout="wide")
st.title("Laboratório de áudio — experimente os parâmetros")
st.write(
    "Carregue um WAV, MP3, OGG ou WebM. Ajuste um controle por vez, "
    "ouça o resultado e compare com o original. O processamento é local."
)

uploaded = st.file_uploader("Arquivo de áudio", type=["wav", "mp3", "ogg", "flac", "webm"])
if uploaded is None:
    st.info("Comece carregando um arquivo. Para WebM/MP3, instale também o FFmpeg.")
    st.stop()

try:
    raw = uploaded.getvalue()
    # soundfile handles WAV/FLAC/OGG; pydub+ffmpeg handles WebM/MP3.
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:
        from pydub import AudioSegment
        segment = AudioSegment.from_file(io.BytesIO(raw))
        sr = segment.frame_rate
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        samples /= float(1 << (8 * segment.sample_width - 1))
        audio = samples.reshape((-1, segment.channels))
    if audio.shape[1] > 1:
        mono = audio.mean(axis=1)
    else:
        mono = audio[:, 0]
except Exception as exc:
    st.error(f"Não consegui abrir o arquivo: {exc}")
    st.stop()

st.caption(f"Duração: {len(mono)/sr:.2f} s · Taxa: {sr} Hz · Canais: {audio.shape[1]}")
st.subheader("1. Áudio original")
st.audio(raw)

col1, col2 = st.columns(2)
with col1:
    hp_on = st.checkbox("Ativar passa-alta", value=True)
    hp = st.slider("Corte passa-alta (Hz)", 20, min(500, sr//3), 80, 5,
                   disabled=not hp_on)
    st.caption("Aumentar o corte remove mais graves, mas pode afinar a voz.")
with col2:
    lp_on = st.checkbox("Ativar passa-baixa", value=False)
    lp_max = max(1000, min(10000, sr//2 - 200))
    lp = st.slider("Corte passa-baixa (Hz)", 1000, lp_max, min(7500, lp_max), 100,
                   disabled=not lp_on)
    st.caption("Diminuir o corte remove mais agudos; cortes excessivos tiram consoantes.")

st.subheader("2. Outros ajustes")
c1, c2, c3 = st.columns(3)
with c1:
    denoise = st.slider("Redução de ruído (%)", 0, 90, 0, 5)
    st.caption("Comece em 0 e aumente aos poucos. Muito alto pode deformar a fala.")
with c2:
    gain_db = st.slider("Ganho (dB)", -12, 12, 0, 1)
    st.caption("Aumenta/diminui o volume; não recupera detalhes já perdidos.")
with c3:
    normalize = st.checkbox("Normalizar pico para -3 dBFS", value=False)
    st.caption("Ajusta o nível final. Não remove ruído por si só.")

processed = mono.copy()
try:
    if hp_on:
        sos = butter(4, hp, btype="highpass", fs=sr, output="sos")
        processed = sosfiltfilt(sos, processed)
    if lp_on:
        if lp >= sr/2:
            st.warning("O corte passa-baixa precisa ser menor que metade da taxa de amostragem.")
        else:
            sos = butter(4, lp, btype="lowpass", fs=sr, output="sos")
            processed = sosfiltfilt(sos, processed)
    if denoise > 0:
        import noisereduce as nr
        # This is a general-purpose spectral reduction; it is not a substitute
        # for a dedicated noise-only profile when one is available.
        processed = nr.reduce_noise(
            y=processed, sr=sr, prop_decrease=denoise/100.0,
            stationary=True
        )
    processed *= 10 ** (gain_db / 20)
    if normalize:
        peak = np.max(np.abs(processed)) if len(processed) else 0
        if peak > 0:
            processed *= (10 ** (-3/20)) / peak
    # Prevent clipping in the exported PCM file; show warning if this is needed.
    peak_before_limit = float(np.max(np.abs(processed))) if len(processed) else 0
    if peak_before_limit > 1:
        st.warning("O sinal ultrapassou 0 dBFS; o arquivo exportado será limitado para evitar clipping.")
        processed = np.clip(processed, -1, 1)
except Exception as exc:
    st.error(f"Falha no processamento: {exc}")
    if denoise > 0:
        st.info("Para redução de ruído, instale o pacote noisereduce conforme o requirements.txt.")
    st.stop()

st.subheader("3. Compare")
st.write("Original")
st.audio(raw)
buf = io.BytesIO()
sf.write(buf, processed, sr, format="WAV", subtype="PCM_16")
processed_bytes = buf.getvalue()
st.write("Processado")
st.audio(processed_bytes, format="audio/wav")
st.download_button("Baixar áudio processado (WAV)", processed_bytes,
                   file_name="audio_processado.wav", mime="audio/wav")

def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
m1, m2, m3 = st.columns(3)
m1.metric("RMS original", f"{rms(mono):.4f}")
m2.metric("RMS processado", f"{rms(processed):.4f}")
m3.metric("Pico processado", f"{np.max(np.abs(processed)):.4f}")

st.warning(
    "Este painel permite experimentar filtros e redução espectral, mas não mede sozinho "
    "a precisão da transcrição. Para avaliar ASR, compare cada resultado com uma transcrição "
    "de referência e calcule WER. Preserve sempre o arquivo original."
)
