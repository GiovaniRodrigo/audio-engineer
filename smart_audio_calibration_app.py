from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components

from smart_audio_calibration import (
    CANDIDATE_PROFILES, apply_profile, prepare_for_asr,
    select_profile, signal_diagnostics, transcript_error,
)

st.set_page_config(page_title="Smart Audio Calibration", page_icon="🎙️", layout="wide")
st.title("🎙️ Smart Audio Calibration")
st.caption("Calibração automática do áudio orientada à redução de erros de transcrição — não à estética sonora.")

PHRASES = [
    "O rato roeu a roupa do rei.",
    "Três pratos de trigo para três tigres.",
    "A casa azul fica próxima à estação.",
    "Papai comprou pipoca para Pedro.",
    "Sessenta e seis sessões foram realizadas.",
    "Amanhã enviarei os documentos necessários.",
]

@st.cache_resource(show_spinner="Carregando Faster-Whisper em CPU…")
def load_whisper(model_size: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cpu", compute_type="int8")

def decode_recording(upload):
    data, sr = sf.read(io.BytesIO(upload.getvalue()), dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError("Gravação vazia.")
    # Avoid phase cancellation: select the channel with the highest RMS.
    channel_rms = np.sqrt(np.mean(data.astype(np.float64) ** 2, axis=0))
    mono = data[:, int(np.argmax(channel_rms))]
    return mono.astype(np.float32), int(sr)

def whisper_transcribe(model, signal, sr):
    waveform = prepare_for_asr(signal, sr)
    segments, info = model.transcribe(
        waveform, language="pt", beam_size=5, vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(part.text.strip() for part in segments).strip()

@st.cache_resource
def browser_asr():
    return components.declare_component(
        "smart_audio_browser_asr",
        path=str(Path(__file__).parent / "browser_asr_component"),
    )

with st.sidebar:
    st.header("Configuração")
    engine_choice = st.radio("Mecanismos de transcrição", ["Ambos", "Faster-Whisper", "Navegador (Web Speech API)"], index=0)
    model_size = st.selectbox("Modelo Whisper (CPU)", ["tiny", "base", "small"], index=1)
    st.caption("O modelo é mantido igual em todos os testes A/B.")
    st.divider()
    st.caption("O reconhecimento do navegador requer Chrome/Edge, permissão de microfone e conexão com o serviço de fala do navegador.")

st.markdown("""
**Como funciona:** grave as frases de referência → o sistema mede o sinal → testa perfis conservadores de DSP → compara as transcrições com o texto correto → seleciona o perfil com menor WER médio.
""")
st.info("Grave cada frase no mesmo ambiente, com o microfone na mesma posição e sem alterar o volume entre as amostras. O sistema preserva as gravações originais.")

recordings=[]
with st.expander("1. Gravação das frases de calibração", expanded=True):
    st.write("Fale naturalmente, sem acelerar. É possível começar com apenas uma frase; mais amostras tornam a calibração mais representativa.")
    for i, phrase in enumerate(PHRASES, 1):
        st.markdown(f"**Frase {i}.** {phrase}")
        upload=st.audio_input(f"Gravar frase {i}", key=f"calibration_audio_{i}")
        if upload is not None:
            recordings.append((phrase, upload))
    st.caption(f"{len(recordings)} de {len(PHRASES)} gravações recebidas.")

if recordings and st.button("Executar calibração automática", type="primary", use_container_width=True):
    if engine_choice == "Navegador (Web Speech API)":
        st.warning("A calibração DSP usa o Faster-Whisper como métrica objetiva. O reconhecimento do navegador é disponibilizado abaixo como comparação independente.")
    try:
        model=load_whisper(model_size)
        decoded=[]
        for reference, upload in recordings:
            samples,sr=decode_recording(upload)
            decoded.append((reference,samples,sr))
        # All candidates are evaluated against the same references and the same ASR model.
        rows=[]
        progress=st.progress(0.0, text="Avaliando perfis de processamento…")
        total=len(CANDIDATE_PROFILES)
        for idx, profile in enumerate(CANDIDATE_PROFILES):
            errors=[]; hypotheses=[]
            for reference,samples,sr in decoded:
                processed=apply_profile(samples,sr,profile)
                hypothesis=whisper_transcribe(model,processed,sr)
                hypotheses.append(hypothesis)
                errors.append(transcript_error(reference,hypothesis))
            finite_wer=[e["wer"] for e in errors if np.isfinite(e["wer"])]
            finite_cer=[e["cer"] for e in errors if np.isfinite(e["cer"])]
            rows.append({
                "profile":profile.name,
                "mean_wer":float(np.mean(finite_wer)) if finite_wer else None,
                "mean_cer":float(np.mean(finite_cer)) if finite_cer else None,
                "hypotheses":hypotheses,
                "parameters":{"highpass_hz":profile.highpass_hz,"denoise_strength":profile.denoise_strength,"gain_db":profile.gain_db},
            })
            progress.progress((idx+1)/total,text=f"Perfil {idx+1}/{total}: {profile.name}")
        best=select_profile(rows)
        st.session_state["calibration_results"]=rows
        st.session_state["calibration_best"]=best
        st.session_state["calibration_audio"]=[(ref,samples,sr) for ref,samples,sr in decoded]
    except Exception as exc:
        st.error(f"Não foi possível concluir a calibração: {exc}")

if "calibration_best" in st.session_state:
    rows=st.session_state["calibration_results"]
    best=st.session_state["calibration_best"]
    st.divider();st.subheader("2. Resultado da calibração")
    if best["profile"]=="Original (baseline)":
        st.warning("Nenhum perfil testado reduziu o WER em relação ao áudio original. O sistema manteve o baseline; não aplicou filtros por padrão.")
    else:
        st.success(f"Perfil recomendado: **{best['profile']}**")
    c1,c2=st.columns(2)
    c1.metric("WER médio", "—" if best["mean_wer"] is None else f"{best['mean_wer']*100:.1f}%")
    c2.metric("CER médio", "—" if best["mean_cer"] is None else f"{best['mean_cer']*100:.1f}%")
    st.dataframe([{"Perfil":r["profile"],"WER médio":r["mean_wer"],"CER médio":r["mean_cer"]} for r in rows],use_container_width=True,hide_index=True)
    st.caption("WER/CER são calculados contra as frases conhecidas. O resultado pode superajustar às frases de calibração; valide com frases novas antes de adotar o perfil.")
    st.subheader("3. Diagnóstico e comparação")
    for (reference,samples,sr), hypothesis in zip(st.session_state["calibration_audio"],best["hypotheses"]):
        with st.expander(reference):
            metrics=signal_diagnostics(samples,sr)
            st.write(f"Duração: {metrics['duration_s']:.2f} s · Pico: {metrics['peak_dbfs']:.1f} dBFS · RMS: {metrics['rms_dbfs']:.1f} dBFS")
            if metrics["clipping_suspected"]:
                st.warning("Possível clipping na gravação. Equalização não recupera áudio já distorcido; reduza o ganho de entrada ou afaste o microfone.")
            st.write("**Transcrição do perfil recomendado:**")
            st.write(hypothesis or "Sem texto reconhecido.")
            profile=next(p for p in CANDIDATE_PROFILES if p.name==best["profile"])
            processed=apply_profile(samples,sr,profile)
            audio_buffer=io.BytesIO();sf.write(audio_buffer,processed,sr,format="WAV")
            st.audio(audio_buffer.getvalue(),format="audio/wav")
    profile_json=json.dumps(best["parameters"],ensure_ascii=False,indent=2)
    st.download_button("Baixar perfil de calibração (JSON)",profile_json,file_name="smart-audio-profile.json",mime="application/json")

if engine_choice in ("Ambos","Navegador (Web Speech API)"):
    st.divider();st.subheader("4. Comparação com reconhecimento do navegador")
    st.write("Use o mesmo ambiente e fale uma frase de teste. O navegador reconhece diretamente o microfone; ele não recebe o áudio já filtrado pelo calibrador.")
    component=browser_asr()
    browser_result=component(key="browser_asr_ptbr")
    if isinstance(browser_result,dict):
        if browser_result.get("transcript"):
            st.markdown("**Resultado do navegador (pt-BR):**")
            st.write(browser_result["transcript"])
        elif browser_result.get("error"):
            st.warning(browser_result["error"])

st.divider()
st.subheader("Limites importantes")
st.markdown("""
- A equalização não remove todos os ruídos. Ruído de vento, reverberação, clipping e microfone distante exigem ações diferentes.
- O perfil automático compara apenas um conjunto pequeno e conservador de filtros; ele não altera o áudio original.
- O WER depende do modelo e das frases de referência. Grave frases de validação diferentes das usadas na calibração.
- O resultado do navegador é uma comparação independente em tempo real; por limitações da API, não é uma transcrição do arquivo processado.
""")
