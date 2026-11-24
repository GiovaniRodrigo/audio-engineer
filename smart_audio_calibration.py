"""DSP utilities for transcript-oriented automatic audio calibration."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


@dataclass(frozen=True)
class CalibrationProfile:
    name: str
    highpass_hz: int = 0
    denoise_strength: float = 0.0
    gain_db: float = 0.0


CANDIDATE_PROFILES = (
    CalibrationProfile("Original (baseline)"),
    CalibrationProfile("Passa-alta 70 Hz", highpass_hz=70),
    CalibrationProfile("Passa-alta 100 Hz", highpass_hz=100),
    CalibrationProfile("Redução leve de ruído", denoise_strength=0.30),
    CalibrationProfile("Passa-alta 70 Hz + redução leve", highpass_hz=70, denoise_strength=0.30),
    CalibrationProfile("Passa-alta 100 Hz + redução moderada", highpass_hz=100, denoise_strength=0.45),
)


def prepare_for_asr(samples: np.ndarray, sample_rate: int, target_rate: int = 16000) -> np.ndarray:
    signal = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate != target_rate:
        divisor = math.gcd(int(sample_rate), target_rate)
        signal = resample_poly(signal, target_rate // divisor, int(sample_rate) // divisor)
    return np.ascontiguousarray(signal, dtype=np.float32)


def apply_profile(samples: np.ndarray, sample_rate: int, profile: CalibrationProfile) -> np.ndarray:
    """Apply conservative speech-oriented DSP; never mutate the source array."""
    signal = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
    if signal.size < 32:
        raise ValueError("Áudio curto demais para processamento.")
    nyquist = sample_rate / 2
    if profile.highpass_hz:
        if profile.highpass_hz >= nyquist:
            raise ValueError("Taxa de amostragem insuficiente para o filtro selecionado.")
        sos = butter(2, profile.highpass_hz, btype="highpass", fs=sample_rate, output="sos")
        signal = sosfiltfilt(sos, signal).astype(np.float32)
    if profile.denoise_strength:
        try:
            import noisereduce as nr
        except ImportError as exc:
            raise RuntimeError("Instale noisereduce para habilitar redução de ruído.") from exc
        signal = nr.reduce_noise(
            y=signal, sr=sample_rate, prop_decrease=profile.denoise_strength,
            stationary=False, n_fft=512, win_length=512, hop_length=128,
        ).astype(np.float32)
    if profile.gain_db:
        signal *= np.float32(10 ** (profile.gain_db / 20))
    # Keep headroom; clipping is not a valid form of level normalization.
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 0.98:
        signal *= np.float32(0.98 / peak)
    return signal


def signal_diagnostics(samples: np.ndarray, sample_rate: int) -> dict[str, float | bool]:
    signal = np.asarray(samples, dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    rms = float(np.sqrt(np.mean(signal ** 2))) if signal.size else 0.0
    clipped = bool(np.mean(np.abs(signal) >= 0.999) > 0.001) if signal.size else False
    frame = max(1, int(sample_rate * 0.02))
    usable = signal[: (len(signal) // frame) * frame]
    if usable.size:
        frame_rms = np.sqrt(np.mean(usable.reshape(-1, frame) ** 2, axis=1))
        floor = float(np.percentile(frame_rms, 20))
        active = float(np.percentile(frame_rms, 80))
        dynamic_contrast_db = 20 * math.log10(max(active, 1e-8) / max(floor, 1e-8))
    else:
        dynamic_contrast_db = 0.0
    return {
        "duration_s": len(signal) / sample_rate if sample_rate else 0.0,
        "peak": peak,
        "rms": rms,
        "peak_dbfs": 20 * math.log10(max(peak, 1e-8)),
        "rms_dbfs": 20 * math.log10(max(rms, 1e-8)),
        "clipping_suspected": clipped,
        "dynamic_contrast_db": dynamic_contrast_db,
    }


def normalized_text(text: str) -> str:
    import re
    return re.sub(r"[^0-9a-záàâãéêíóôõúüç ]+", " ", text.lower()).strip()


def transcript_error(reference: str, hypothesis: str) -> dict[str, float]:
    """Return WER/CER using jiwer, with a safe fallback for empty references."""
    from jiwer import wer, cer
    ref, hyp = normalized_text(reference), normalized_text(hypothesis)
    if not ref:
        return {"wer": float("nan"), "cer": float("nan")}
    return {"wer": float(wer(ref, hyp)), "cer": float(cer(ref, hyp))}


def select_profile(results: Iterable[dict]) -> dict:
    """Select by mean WER; retain baseline on ties or when no candidate improves it."""
    rows = list(results)
    baseline = next((row for row in rows if row["profile"] == "Original (baseline)"), None)
    valid = [row for row in rows if row.get("mean_wer") is not None and np.isfinite(row["mean_wer"])]
    if not valid:
        return baseline or rows[0]
    best = min(valid, key=lambda row: row["mean_wer"])
    if baseline and (best["mean_wer"] >= baseline["mean_wer"] - 1e-9):
        return baseline
    return best
