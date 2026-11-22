from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt, spectrogram


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


@st.cache_resource(show_spinner="Carregando o modelo de reconhecimento de fala...")
def load_asr_model(model_size: str):
    """Load and cache a CPU/int8 Faster-Whisper model."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


def prepare_for_asr(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert mono audio to the 16 kHz float32 waveform expected by Whisper."""
    signal = np.asarray(signal, dtype=np.float32)
    if sample_rate != 16000:
        divisor = math.gcd(int(sample_rate), 16000)
        signal = resample_poly(
            signal, 16000 // divisor, int(sample_rate) // divisor
        ).astype(np.float32)
    return np.ascontiguousarray(signal)


def transcribe_with_model(model, signal: np.ndarray, sample_rate: int, language: str, beam_size: int):
    waveform = prepare_for_asr(signal, sample_rate)
    selected_language = None if language == "Auto detectar" else language
    segments, info = model.transcribe(
        waveform,
        language=selected_language,
        beam_size=beam_size,
        vad_filter=True,
    )
    segments = list(segments)
    transcript = " ".join(item.text.strip() for item in segments).strip()
    rows = [
        {
            "start": float(item.start),
            "end": float(item.end),
            "text": item.text.strip(),
        }
        for item in segments
    ]
    return transcript, rows, info


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
    with st.expander("Aula rápida: como ler a forma de onda", expanded=True):
        st.markdown(
            """
**O que estou vendo?** Cada ponto representa a amplitude do sinal em um instante.
O eixo horizontal mostra o tempo em segundos; o vertical mostra a amplitude digital,
normalizada aproximadamente entre −1 e +1.

**Como interpretar?**
- Trechos próximos de zero podem indicar silêncio ou pausas.
- Oscilações maiores indicam maior amplitude, mas não necessariamente melhor qualidade.
- Topos achatados ou valores encostados em ±1 podem indicar clipping (distorção por limite digital).

**Experimente:** reproduza o áudio e localize uma palavra. Observe onde começam e terminam
as oscilações. Depois aumente o ganho no painel DSP e veja como a forma de onda cresce.
"""
        )

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
            with st.expander("Aula rápida: como ler o espectro FFT", expanded=True):
                st.markdown(
                    """
**O que estou vendo?** A FFT decompõe o trecho selecionado em componentes de frequência.
O eixo horizontal mostra frequência em hertz (Hz); o vertical mostra a magnitude aproximada
em decibéis.

**Como interpretar?**
- Picos indicam componentes relativamente fortes naquele trecho.
- Energia em baixas frequências pode vir da voz, vibração, vento ou manuseio do microfone.
- Componentes agudos podem conter detalhes importantes de consoantes.
- Um pico isolado não permite afirmar, sozinho, que existe ruído: é preciso ouvir e comparar.

**Experimente:** selecione um trecho com fala e outro com uma pausa. Compare os espectros.
Observe também que a resolução em frequência depende do tamanho do trecho analisado.
"""
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
        with st.expander("Aula rápida: como ler o espectrograma", expanded=True):
            st.markdown(
                """
**O que estou vendo?** É um mapa de energia do áudio ao longo do tempo.
O eixo horizontal representa o tempo; o vertical, a frequência; as cores representam
a densidade de energia espectral (dB/Hz).

**Como interpretar?**
- Faixas horizontais podem representar componentes tonais persistentes.
- Estruturas que mudam ao longo do tempo acompanham a articulação da fala.
- Ruídos contínuos podem aparecer durante quase todo o trecho; ruídos impulsivos aparecem em momentos específicos.
- A janela controla o compromisso entre resolução temporal e resolução em frequência.

**Experimente:** compare uma vogal sustentada com uma consoante fricativa, como /s/.
Observe como os padrões visuais são diferentes. A cor depende da escala do gráfico.
"""
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

with st.expander("Entenda os controles de processamento", expanded=False):
    st.markdown(
        """
- **Passa-alta:** atenua frequências abaixo do corte. Pode reduzir vento e vibrações graves,
  mas um corte alto remove parte do corpo da voz. O filtro Butterworth de 4ª ordem tem
  transição gradual, não uma parede abrupta.
- **Passa-baixa:** atenua frequências acima do corte. Pode reduzir chiado agudo, mas cortes
  baixos prejudicam consoantes como /s/, /f/ e /ch/.
- **Redução espectral:** estima e atenua componentes considerados ruído. Pode produzir
  artefatos metálicos ou voz “aquosa”; aumente gradualmente e sempre faça escuta crítica.
- **Ganho:** multiplica a amplitude. Um ganho de +6 dB aproximadamente dobra a amplitude;
  não melhora a relação sinal-ruído.
- **Normalização:** ajusta o pico para −3 dBFS. Evita aproximar o pico do limite digital,
  mas não remove ruído nem recupera informação perdida.

**Experimento sugerido:** altere apenas um controle por vez. Ouça o resultado e compare
o gráfico antes de combinar filtros.
"""
    )

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

st.subheader("4. Comparação visual: original × processado")
st.write(
    "As duas curvas usam a mesma escala de tempo e amplitude. "
    "A sobreposição ajuda a perceber mudanças no nível e na forma do sinal."
)
original_time, original_values = plot_samples(original, sr)
processed_time, processed_values = plot_samples(processed, sr)
comparison_fig = go.Figure()
comparison_fig.add_trace(
    go.Scattergl(
        x=original_time,
        y=original_values,
        mode="lines",
        name="Original",
        opacity=0.75,
    )
)
comparison_fig.add_trace(
    go.Scattergl(
        x=processed_time,
        y=processed_values,
        mode="lines",
        name="Processado",
        opacity=0.75,
    )
)
comparison_fig.update_layout(
    xaxis_title="Tempo (s)",
    yaxis_title="Amplitude relativa",
    height=380,
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h"),
)
st.plotly_chart(comparison_fig, use_container_width=True)
with st.expander("O que a comparação permite concluir?"):
    st.markdown(
        """
- Se a curva processada ficar maior, houve aumento de amplitude; isso não prova melhoria.
- Se os picos ficarem menores, pode ter ocorrido atenuação ou redução de ruído, mas a voz também pode ter sido afetada.
- Se a forma mudar muito, ouça com atenção: filtros e redução espectral podem introduzir artefatos.
- A avaliação final depende da audição e, para medir transcrição, de comparar o texto reconhecido
  com uma transcrição de referência usando WER/CER.
"""
    )

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

st.divider()
st.subheader("5. Reconhecimento automático de fala (ASR)")
st.markdown(
    "Agora podemos testar se o processamento altera o que o reconhecedor entende. "
    "O modelo recebe o áudio digital, não o gráfico: as visualizações ajudam a investigar "
    "o sinal, enquanto a transcrição permite avaliar o resultado linguístico."
)

asr_col1, asr_col2, asr_col3 = st.columns(3)
with asr_col1:
    asr_model_size = st.selectbox(
        "Modelo Faster-Whisper",
        ["tiny", "base", "small", "medium"],
        index=1,
        help="Modelos maiores tendem a ser mais precisos, mas consomem mais memória e tempo.",
    )
with asr_col2:
    asr_language_label = st.selectbox(
        "Idioma da fala",
        ["Auto detectar", "Português", "Inglês", "Espanhol"],
        index=1,
    )
    asr_language = {
        "Auto detectar": "Auto detectar",
        "Português": "pt",
        "Inglês": "en",
        "Espanhol": "es",
    }[asr_language_label]
with asr_col3:
    asr_beam_size = st.select_slider(
        "Beam size",
        options=[1, 3, 5, 8],
        value=5,
        help="Valores maiores exploram mais hipóteses de texto e podem aumentar o tempo.",
    )

st.caption(
    "Execução local em CPU com quantização int8. Na primeira execução, os pesos do modelo "
    "serão baixados automaticamente. O tempo depende do tamanho do modelo e do computador."
)
reference_text = st.text_area(
    "Transcrição de referência (opcional, revise manualmente)",
    placeholder="Digite aqui exatamente o que foi falado no áudio...",
    help="WER e CER só são calculados quando você fornece uma referência confiável.",
    height=100,
)
run_asr = st.button("Transcrever original e processado", type="primary")
if run_asr:
    try:
        with st.spinner("Carregando o modelo e transcrevendo os dois áudios..."):
            asr_model = load_asr_model(asr_model_size)
            original_text, original_segments, original_info = transcribe_with_model(
                asr_model, original, sr, asr_language, asr_beam_size
            )
            processed_text, processed_segments, processed_info = transcribe_with_model(
                asr_model, processed, sr, asr_language, asr_beam_size
            )
        st.session_state["asr_result"] = {
            "model": asr_model_size,
            "language": asr_language_label,
            "original_text": original_text,
            "processed_text": processed_text,
            "original_segments": original_segments,
            "processed_segments": processed_segments,
        }
        st.success(
            f"Transcrição concluída. Idioma detectado no original: "
            f"{getattr(original_info, 'language', 'não informado')}."
        )
    except ImportError:
        st.error(
            "Faster-Whisper não está instalado. Execute: "
            "python -m pip install -r requirements.txt"
        )
    except Exception as exc:
        st.error(f"Falha no reconhecimento: {exc}")

result = st.session_state.get("asr_result")
if result:
    st.caption(
        f"Último resultado: modelo {result['model']} · "
        f"idioma configurado: {result['language']}"
    )
    transcript_col1, transcript_col2 = st.columns(2)
    with transcript_col1:
        st.markdown("**Transcrição do áudio original**")
        st.text_area(
            "Texto original reconhecido",
            value=result["original_text"],
            height=160,
            key="asr_original_display",
        )
    with transcript_col2:
        st.markdown("**Transcrição do áudio processado**")
        st.text_area(
            "Texto processado reconhecido",
            value=result["processed_text"],
            height=160,
            key="asr_processed_display",
        )

    with st.expander("Segmentos e marcações de tempo"):
        st.markdown("**Original**")
        for item in result["original_segments"]:
            st.write(f"[{item['start']:.2f}–{item['end']:.2f} s] {item['text']}")
        st.markdown("**Processado**")
        for item in result["processed_segments"]:
            st.write(f"[{item['start']:.2f}–{item['end']:.2f} s] {item['text']}")

    if reference_text.strip():
        try:
            from jiwer import cer, wer

            reference = reference_text.strip()
            original_wer = wer(reference, result["original_text"]) * 100
            processed_wer = wer(reference, result["processed_text"]) * 100
            original_cer = cer(reference, result["original_text"]) * 100
            processed_cer = cer(reference, result["processed_text"]) * 100
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.markdown("**WER — taxa de erro de palavras**")
                st.metric("Original", f"{original_wer:.2f}%")
                st.metric(
                    "Processado",
                    f"{processed_wer:.2f}%",
                    delta=f"{processed_wer - original_wer:+.2f} p.p.",
                    delta_color="inverse",
                )
            with metric_col2:
                st.markdown("**CER — taxa de erro de caracteres**")
                st.metric("Original", f"{original_cer:.2f}%")
                st.metric(
                    "Processado",
                    f"{processed_cer:.2f}%",
                    delta=f"{processed_cer - original_cer:+.2f} p.p.",
                    delta_color="inverse",
                )
            st.caption(
                "Quanto menores WER e CER, melhor a correspondência com a referência. "
                "WER pode ultrapassar 100% quando há muitas inserções. Use a mesma referência "
                "e os mesmos parâmetros ao comparar experimentos."
            )
        except ImportError:
            st.warning("Instale jiwer com: python -m pip install -r requirements.txt")
    else:
        st.info(
            "Adicione uma transcrição de referência revisada para calcular WER e CER. "
            "Sem ela, temos apenas duas hipóteses de transcrição, não uma medida de precisão."
        )

    st.download_button(
        "Baixar transcrições (TXT)",
        data=(
            "TRANSCRIÇÃO — ORIGINAL\n"
            + result["original_text"]
            + "\n\nTRANSCRIÇÃO — PROCESSADO\n"
            + result["processed_text"]
            + "\n"
        ),
        file_name="transcricoes_audio.txt",
        mime="text/plain",
    )

with st.expander("Como interpretar WER e CER"):
    st.markdown(
        """
- **WER (Word Error Rate):** soma de substituições, exclusões e inserções de palavras,
  dividida pelo número de palavras da referência.
- **CER (Character Error Rate):** a mesma ideia, aplicada a caracteres.
- **0%:** transcrição idêntica à referência após a normalização feita pela biblioteca.
- **Valores maiores:** mais diferenças em relação ao texto de referência.
- **Limitação:** uma transcrição de referência incorreta ou incompleta compromete a avaliação.
  Revise pontuação, números, nomes próprios e palavras inaudíveis antes de comparar.
"""
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
