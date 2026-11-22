from __future__ import annotations

import io
import math

import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt, spectrogram

st.set_page_config(page_title="Curso de Engenharia de Áudio", page_icon="🎧", layout="wide")

STAGES = [
    "0 · Comece aqui",
    "1 · Fundamentos e aquisição",
    "2 · Forma de onda",
    "3 · Frequência e espectrograma",
    "4 · Processamento digital (DSP)",
    "5 · Reconhecimento de fala (ASR)",
    "6 · Projeto final",
]

LESSONS = {
    1: {
        "title": "Fundamentos e aquisição",
        "goal": "Entender como o som analógico é representado por números.",
        "theory": """
O microfone transforma variações de pressão do ar em um sinal elétrico. O conversor
analógico-digital mede esse sinal em intervalos regulares e registra cada medida como
uma amostra.

- **Taxa de amostragem (Hz):** quantidade de amostras por segundo.
- **Profundidade de bits:** quantidade de níveis possíveis para representar a amplitude.
- **Canais:** mono tem um canal; estéreo normalmente tem dois.
- **Nyquist:** a maior frequência representável sem aliasing é, idealmente, menor que
  metade da taxa de amostragem.
- **Duração:** número de amostras dividido pela taxa de amostragem.
""",
        "task": "Carregue um áudio e identifique taxa de amostragem, duração e quantidade de canais.",
    },
    2: {
        "title": "Forma de onda",
        "goal": "Relacionar amplitude, tempo, silêncio, picos e clipping.",
        "theory": """
A forma de onda representa a amplitude do sinal ao longo do tempo. O eixo horizontal
é o tempo; o vertical é a amplitude digital.

- Valores próximos de zero podem representar pausas ou trechos de baixa energia.
- Picos maiores indicam maior amplitude, não necessariamente melhor qualidade.
- Clipping ocorre quando o sinal ultrapassa o limite que o sistema pode representar.
- RMS resume o nível médio quadrático; pico indica a maior amplitude absoluta.
""",
        "task": "Localize uma pausa e um trecho de fala. Compare a amplitude dos dois.",
    },
    3: {
        "title": "Frequência e espectrograma",
        "goal": "Identificar componentes de frequência e observar como mudam no tempo.",
        "theory": """
A FFT representa o conteúdo de frequência de um trecho. Ela responde “quais frequências
estão presentes?”, mas não mostra sozinha quando ocorreram.

O espectrograma divide o áudio em janelas e mostra a energia por frequência ao longo do
tempo. O tamanho da janela cria um compromisso: janelas maiores melhoram a resolução
em frequência, enquanto janelas menores acompanham melhor eventos rápidos.
""",
        "task": "Compare uma vogal sustentada com uma consoante fricativa, como /s/, se houver no áudio.",
    },
    4: {
        "title": "Processamento digital (DSP)",
        "goal": "Aplicar filtros e ajustes sem destruir informação útil da fala.",
        "theory": """
- **Passa-alta:** atenua frequências abaixo do corte; pode reduzir vento e vibrações graves.
- **Passa-baixa:** atenua frequências acima do corte; cortes excessivos prejudicam consoantes.
- **Redução espectral:** estima e atenua ruído, mas pode criar artefatos.
- **Ganho:** altera a amplitude. Não melhora, por si só, a relação sinal-ruído.
- **Normalização:** ajusta o nível de pico; não remove ruído nem recupera informação perdida.

Altere um parâmetro por vez, ouça o resultado e compare com o original.
""",
        "task": "Faça um único ajuste, ouça o resultado e explique se a voz melhorou ou perdeu algum detalhe.",
    },
    5: {
        "title": "Reconhecimento automático de fala (ASR)",
        "goal": "Entender como avaliar uma transcrição, sem confundir hipótese com verdade.",
        "theory": """
O ASR (Automatic Speech Recognition) converte áudio em texto. O resultado depende do
modelo, idioma, ruído, microfone, sotaque e qualidade da gravação.

- **WER:** taxa de erro de palavras, baseada em substituições, exclusões e inserções.
- **CER:** taxa de erro de caracteres.
- Uma referência revisada por uma pessoa é necessária para calcular métricas confiáveis.
- Menor WER/CER indica maior correspondência com a referência; não prova compreensão semântica.
""",
        "task": "Transcreva o original e o áudio processado. Compare ambos com uma referência revisada.",
    },
    6: {
        "title": "Projeto final",
        "goal": "Conduzir um experimento completo e justificar as decisões técnicas.",
        "theory": """
No projeto final, você deve documentar o problema, caracterizar o áudio, formular uma
hipótese, escolher um processamento, comparar original e resultado e avaliar a transcrição.

Um bom experimento muda poucas variáveis por vez, preserva o original e registra os
parâmetros. Não conclua que o áudio melhorou apenas porque ficou mais alto ou mais agradável.
""",
        "task": "Produza um relatório curto com objetivo, características do áudio, parâmetros, comparação auditiva e WER/CER quando houver referência.",
    },
}


def equipment_chain_figure():
    """Interactive isometric 3D schematic of an audio capture/playback chain."""
    blocks = [
        (0.0, "Microfone", "Converte variações de pressão do ar em sinal elétrico analógico."),
        (2.0, "Pré-amplificador", "Eleva o sinal fraco do microfone para um nível adequado."),
        (4.0, "Filtro anti-aliasing", "Limita frequências antes da conversão para evitar aliasing."),
        (6.0, "ADC", "Converte o sinal analógico em números: amostra e quantiza."),
        (8.0, "DSP / computador", "Armazena e processa as amostras digitais."),
        (10.0, "DAC", "Converte os números digitais novamente em sinal analógico."),
        (12.0, "Amplificador", "Fornece potência suficiente para movimentar o alto-falante."),
        (14.0, "Alto-falante", "Converte o sinal elétrico em vibrações do ar."),
    ]
    fig = go.Figure()
    # Connections are drawn first so the equipment blocks remain in the foreground.
    for x, *_ in blocks[:-1]:
        fig.add_trace(go.Scatter3d(
            x=[x + 0.72, x + 1.28], y=[0, 0], z=[0, 0],
            mode="lines", line=dict(color="#64748b", width=7),
            hoverinfo="skip", showlegend=False
        ))
    faces = (
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)
    )
    for idx, (x, name, detail) in enumerate(blocks):
        dx, dy, dz = 0.72, 0.55, 0.42
        verts = [
            (x-dx, -dy, -dz), (x+dx, -dy, -dz),
            (x+dx, dy, -dz), (x-dx, dy, -dz),
            (x-dx, -dy, dz), (x+dx, -dy, dz),
            (x+dx, dy, dz), (x-dx, dy, dz),
        ]
        color = ["#0ea5e9", "#14b8a6", "#14b8a6", "#f59e0b",
                 "#8b5cf6", "#f59e0b", "#14b8a6", "#0ea5e9"][idx]
        fig.add_trace(go.Mesh3d(
            x=[v[0] for v in verts], y=[v[1] for v in verts], z=[v[2] for v in verts],
            i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
            color=color, opacity=0.92, flatshading=True,
            hovertext=f"<b>{name}</b><br>{detail}", hoverinfo="text",
            showscale=False, showlegend=False
        ))
    fig.add_trace(go.Scatter3d(
        x=[x for x, _, _ in blocks], y=[0]*len(blocks), z=[0.83]*len(blocks),
        mode="text", text=[name for _, name, _ in blocks],
        textfont=dict(size=11, color="#e2e8f0"),
        hoverinfo="skip", showlegend=False
    ))
    fig.update_layout(
        height=520, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="manual", aspectratio=dict(x=5.8, y=1.4, z=1.6),
            camera=dict(eye=dict(x=1.4, y=1.8, z=1.2), projection=dict(type="orthographic"))
        ),
        showlegend=False
    )
    return fig




EQUIPMENT_DETAILS = {
    "Microfone": {
        "kind": "microphone",
        "role": "Capta a variação de pressão do ar e a transforma em um sinal elétrico analógico.",
        "input": "Ondas sonoras no ar",
        "output": "Sinal elétrico analógico de baixa intensidade",
        "analogy": "É como um ouvido eletrônico: percebe vibrações no ar e as transforma em eletricidade.",
        "note": "O microfone ainda não produz áudio digital. A conversão para números acontece depois, no ADC."
    },
    "Pré-amplificador": {
        "kind": "preamp",
        "role": "Eleva o sinal muito fraco do microfone para um nível adequado ao restante do circuito.",
        "input": "Sinal analógico de baixa intensidade",
        "output": "Sinal analógico amplificado",
        "analogy": "É como aumentar o volume de uma voz muito baixa antes de gravá-la.",
        "note": "Ganho excessivo pode amplificar também o ruído e pode causar clipping."
    },
    "Filtro anti-aliasing": {
        "kind": "filter",
        "role": "Atenua componentes de frequência acima da faixa que o sistema consegue representar.",
        "input": "Sinal analógico antes do ADC",
        "output": "Sinal limitado em banda",
        "analogy": "É como uma peneira que impede frequências altas demais de entrar na conversão.",
        "note": "É normalmente um circuito analógico integrado ao caminho de entrada da interface, não necessariamente uma caixa separada."
    },
    "ADC — conversor A/D": {
        "kind": "adc",
        "role": "Amostra o sinal analógico em instantes regulares e quantiza cada medição em um valor digital.",
        "input": "Tensão analógica variável",
        "output": "Sequência de números (amostras)",
        "analogy": "É como tirar várias fotografias por segundo da amplitude do sinal.",
        "note": "Taxa de amostragem define quantas medições são feitas por segundo; profundidade de bits define os níveis de amplitude disponíveis."
    },
    "DSP / computador": {
        "kind": "computer",
        "role": "Armazena e processa as amostras digitais por software ou hardware especializado.",
        "input": "Amostras digitais",
        "output": "Áudio digital processado",
        "analogy": "É a bancada de trabalho: nela você corta, filtra, mede e analisa o áudio.",
        "note": "Filtros digitais, redução de ruído, equalização e análise espectral podem acontecer nesta etapa."
    },
    "DAC — conversor D/A": {
        "kind": "dac",
        "role": "Converte os valores digitais em um sinal elétrico analógico variável.",
        "input": "Sequência de números (amostras)",
        "output": "Sinal elétrico analógico",
        "analogy": "Faz o caminho inverso do ADC: reconstrói uma tensão variável a partir dos números.",
        "note": "O sinal passa por circuitos de reconstrução e filtragem antes de alimentar a saída."
    },
    "Amplificador": {
        "kind": "amplifier",
        "role": "Fornece tensão e corrente suficientes para acionar um alto-falante ou fones passivos.",
        "input": "Sinal analógico de nível de linha",
        "output": "Sinal com potência adequada à carga",
        "analogy": "É o músculo do sistema: fornece energia para movimentar o alto-falante.",
        "note": "Não deve ser confundido com o pré-amplificador de microfone: eles trabalham com níveis e finalidades diferentes."
    },
    "Alto-falante": {
        "kind": "speaker",
        "role": "Transforma a energia elétrica do amplificador em movimento mecânico e ondas sonoras no ar.",
        "input": "Sinal elétrico com potência",
        "output": "Ondas sonoras",
        "analogy": "É o caminho inverso do microfone: recebe eletricidade e movimenta o ar.",
        "note": "A bobina movimenta o cone; o cone desloca o ar e produz som."
    },
}


def add_box(fig, x, y, z, dx, dy, dz, color, opacity=1.0):
    vertices = [
        (x-dx/2, y-dy/2, z-dz/2), (x+dx/2, y-dy/2, z-dz/2),
        (x+dx/2, y+dy/2, z-dz/2), (x-dx/2, y+dy/2, z-dz/2),
        (x-dx/2, y-dy/2, z+dz/2), (x+dx/2, y-dy/2, z+dz/2),
        (x+dx/2, y+dy/2, z+dz/2), (x-dx/2, y+dy/2, z+dz/2),
    ]
    faces = [(0,1,2),(0,2,3),(4,6,5),(4,7,6),(0,4,5),(0,5,1),
             (1,5,6),(1,6,2),(2,6,7),(2,7,3),(3,7,4),(3,4,0)]
    fig.add_trace(go.Mesh3d(
        x=[p[0] for p in vertices], y=[p[1] for p in vertices], z=[p[2] for p in vertices],
        i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
        color=color, opacity=opacity, flatshading=True, showscale=False, hoverinfo="skip"
    ))


def add_cylinder(fig, x, y, z, radius, height, color, segments=32):
    angles = np.linspace(0, 2*np.pi, segments, endpoint=False)
    xs = list(x + radius*np.cos(angles)) * 2
    ys = list(y + radius*np.sin(angles)) * 2
    zs = [z-height/2]*segments + [z+height/2]*segments
    i, j, k = [], [], []
    for n in range(segments):
        nxt = (n+1) % segments
        i.extend([n, n+segments, n])
        j.extend([nxt, nxt+segments, nxt+segments])
        k.extend([n+segments, nxt, nxt+segments])
    fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k,
                            color=color, opacity=0.98, showscale=False, hoverinfo="skip"))


def add_cone(fig, x, y, z, radius, depth, color):
    angles = np.linspace(0, 2*np.pi, 48, endpoint=False)
    outer = [(x+radius*np.cos(a), y+radius*np.sin(a), z) for a in angles]
    inner = [(x+0.12*radius*np.cos(a), y+0.12*radius*np.sin(a), z-depth) for a in angles]
    verts = outer + inner
    faces_i, faces_j, faces_k = [], [], []
    for n in range(len(angles)):
        nxt = (n+1) % len(angles)
        faces_i.extend([n, n])
        faces_j.extend([nxt, n+len(angles)])
        faces_k.extend([nxt+len(angles), nxt])
    fig.add_trace(go.Mesh3d(
        x=[p[0] for p in verts], y=[p[1] for p in verts], z=[p[2] for p in verts],
        i=faces_i, j=faces_j, k=faces_k, color=color, opacity=0.95,
        showscale=False, hoverinfo="skip"
    ))


def set_equipment_step(step: int):
    """Update the synchronized text/3D guided lesson step."""
    st.session_state.guided_equipment_step = int(step)


def equipment_detail_figure(name):
    """Create an illustrative 3D model of the selected audio component."""
    kind = EQUIPMENT_DETAILS[name]["kind"]
    fig = go.Figure()
    if kind == "microphone":
        add_cylinder(fig, 0, 0, 0.35, 0.32, 1.25, "#334155")
        add_cylinder(fig, 0, 0, 1.08, 0.34, 0.48, "#94a3b8")
        add_cylinder(fig, 0, 0, -0.42, 0.09, 0.55, "#64748b")
        add_box(fig, 0, 0, -0.78, 0.9, 0.08, 0.08, "#475569")
        for zline in np.linspace(0.9, 1.25, 5):
            fig.add_trace(go.Scatter3d(x=[-0.25,0.25], y=[0,0], z=[zline,zline],
                mode="lines", line=dict(color="#e2e8f0", width=3), hoverinfo="skip", showlegend=False))
    elif kind in ("preamp", "amplifier"):
        add_box(fig, 0, 0, 0, 2.2, 1.25, 0.65, "#334155")
        for x in [-0.7, 0, 0.7]:
            add_cylinder(fig, x, -0.68, 0.08, 0.16, 0.14, "#f59e0b", 24)
        add_cylinder(fig, -0.82, -0.69, -0.1, 0.055, 0.13, "#cbd5e1")
        add_cylinder(fig, 0.82, -0.69, -0.1, 0.055, 0.13, "#cbd5e1")
        add_box(fig, 0, 0.18, 0.36, 0.65, 0.35, 0.06, "#0f172a")
        if kind == "amplifier":
            add_box(fig, 0, 0.7, -0.02, 1.5, 0.12, 0.3, "#64748b")
    elif kind == "filter":
        add_box(fig, 0, 0, 0, 2.0, 1.2, 0.12, "#15803d")
        for x in [-0.65, -0.2, 0.25, 0.7]:
            add_box(fig, x, 0, 0.18, 0.22, 0.3, 0.25, "#1e293b")
        for x in np.linspace(-0.8, 0.8, 9):
            add_cylinder(fig, x, -0.48, 0.12, 0.045, 0.08, "#fbbf24", 12)
        add_box(fig, 0, 0.72, 0.1, 0.35, 0.18, 0.2, "#cbd5e1")
    elif kind in ("adc", "dac"):
        add_box(fig, 0, 0, 0, 1.55, 1.05, 0.16, "#166534")
        add_box(fig, 0, 0, 0.18, 0.68, 0.62, 0.28, "#111827")
        for x in np.linspace(-0.58, 0.58, 8):
            for y in [-0.43, 0.43]:
                add_box(fig, x, y, 0.12, 0.07, 0.18, 0.08, "#e2e8f0")
        for x in [-0.55, 0.55]:
            for y in np.linspace(-0.3, 0.3, 5):
                add_box(fig, x, y, 0.12, 0.18, 0.06, 0.08, "#e2e8f0")
        add_box(fig, 0, 0, 0.38, 0.35, 0.25, 0.08, "#f59e0b")
    elif kind == "computer":
        add_box(fig, 0, 0, -0.12, 1.8, 0.95, 0.18, "#475569")
        add_box(fig, 0, 0.05, 0.55, 1.8, 0.12, 1.15, "#64748b")
        add_box(fig, 0, -0.02, 0.58, 1.58, 0.035, 0.92, "#0ea5e9")
        add_box(fig, 0, -0.52, -0.22, 1.0, 0.05, 0.32, "#94a3b8")
    elif kind == "speaker":
        add_box(fig, 0, 0, 0, 1.7, 1.1, 2.0, "#334155")
        add_cylinder(fig, 0, -0.58, 0.05, 0.58, 0.12, "#0f172a", 48)
        add_cone(fig, 0, -0.68, 0.08, 0.49, 0.48, "#64748b")
        add_cylinder(fig, 0, -0.92, -0.12, 0.12, 0.12, "#cbd5e1", 32)
        add_cylinder(fig, 0, -0.58, 0.72, 0.18, 0.12, "#0f172a", 32)
    fig.update_layout(
        height=440, margin=dict(l=0, r=0, t=5, b=5),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=2.0, z=1.25), projection=dict(type="orthographic"))
        ),
        showlegend=False
    )
    return fig


def sampling_demo(sample_rate: int):
    """Show a continuous analog-like waveform and regularly spaced digital samples."""
    signal_hz = 2.0
    t_cont = np.linspace(0, 1, 1200)
    y_cont = np.sin(2 * np.pi * signal_hz * t_cont)
    t_samples = np.arange(0, 1 + 1e-9, 1 / sample_rate)
    y_samples = np.sin(2 * np.pi * signal_hz * t_samples)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_cont, y=y_cont, mode="lines", name="Sinal contínuo (analógico)",
        line=dict(width=3, color="#0ea5e9")
    ))
    fig.add_trace(go.Scatter(
        x=t_samples, y=y_samples, mode="markers", name="Amostras medidas pelo ADC",
        marker=dict(size=10, color="#f97316", line=dict(width=1, color="#fff"))
    ))
    stem_x, stem_y = [], []
    for tx, ty in zip(t_samples, y_samples):
        stem_x.extend([tx, tx, None])
        stem_y.extend([0, ty, None])
    fig.add_trace(go.Scatter(
        x=stem_x, y=stem_y, mode="lines", name="Instantes de amostragem",
        line=dict(color="#f97316", width=1, dash="dot"), hoverinfo="skip"
    ))
    fig.update_layout(
        height=380, xaxis_title="Tempo (segundos)", yaxis_title="Amplitude",
        hovermode="x unified", legend=dict(orientation="h", y=-0.25),
        margin=dict(l=10, r=10, t=20, b=65)
    )
    return fig, 1 / sample_rate, signal_hz


def decode(raw: bytes, filename: str):
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(raw))
        sr = int(seg.frame_rate)
        channels = int(seg.channels)
        pcm = np.asarray(seg.get_array_of_samples(), dtype=np.float32)
        pcm /= float(1 << (8 * seg.sample_width - 1))
        data = pcm.reshape(-1, channels)
    if data.size == 0:
        raise ValueError("O arquivo não contém amostras.")
    channels = int(data.shape[1])
    mono = data.mean(axis=1) if channels > 1 else data[:, 0]
    return np.asarray(mono, dtype=np.float32), int(sr), channels

def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0

def db(x):
    return -120.0 if x <= 0 else 20 * math.log10(x)

def wav_bytes(x, sr):
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def quiz(stage: int, question: str, options: list[str], answer: str, explanation: str):
    key = f"quiz_{stage}"
    selected = st.radio(question, options, key=f"radio_{stage}")
    if st.button("Conferir resposta", key=f"check_{stage}"):
        if selected == answer:
            st.session_state[key] = True
            st.success("Correto. " + explanation)
        else:
            st.session_state[key] = False
            st.error("Ainda não. " + explanation)
    return st.session_state.get(key, False)

if "completed_stages" not in st.session_state:
    st.session_state.completed_stages = []
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "processed_audio" not in st.session_state:
    st.session_state.processed_audio = None

# Apply navigation requests before the radio widget is created.
if "navigation_target" in st.session_state:
    st.session_state.selected_stage = st.session_state.pop("navigation_target")

with st.sidebar:
    st.title("🎧 Sua trilha")
    done = len(st.session_state.completed_stages)
    st.progress(done / 6, text=f"{done} de 6 etapas concluídas")
    available = [STAGES[0]]
    for idx in range(1, 7):
        if idx == 1 or (idx - 1) in st.session_state.completed_stages:
            available.append(STAGES[idx])
        else:
            break
    selected_stage = st.radio("Etapas", available, key="selected_stage")
    st.divider()
    st.caption("Avance no seu ritmo. Você pode voltar às etapas já liberadas.")

stage = STAGES.index(selected_stage)
st.title("Audio Engineering Lab · Curso prático")
st.caption("Aprenda engenharia de áudio do básico ao avançado, experimentando com seus próprios arquivos.")

if stage == 0:
    st.header("Bem-vindo ao laboratório")
    st.write("""
Este curso transforma conceitos de áudio em experimentos. Você vai começar entendendo
como o som vira dado digital e avançar até processamento e avaliação automática de fala.

**Como funciona cada etapa**
1. Leia a explicação curta.
2. Observe a demonstração no seu próprio áudio.
3. Faça a atividade prática.
4. Responda à verificação de aprendizagem.
5. Conclua a etapa para liberar a próxima.
""")
    st.info("Você precisará de um arquivo de áudio com fala. WAV é recomendado para começar; MP3, OGG, FLAC, WebM e M4A também podem funcionar com FFmpeg instalado.")
    if st.button("Começar o curso", type="primary"):
        st.session_state.navigation_target = STAGES[1]
        st.rerun()

else:
    st.header(LESSONS[stage]["title"])
    st.caption("Objetivo: " + LESSONS[stage]["goal"])
    with st.expander("📘 Aula: conceitos essenciais", expanded=True):
        st.markdown(LESSONS[stage]["theory"])

    if stage == 1:
        st.subheader("A cadeia de áudio: aula visual guiada")
        st.write(
            "Avance etapa por etapa. O texto à esquerda e o equipamento 3D à direita "
            "mudam juntos para mostrar o caminho percorrido pelo som."
        )
        equipment_names = list(EQUIPMENT_DETAILS.keys())
        if "guided_equipment_step" not in st.session_state:
            st.session_state.guided_equipment_step = 0
        current_step = int(st.session_state.guided_equipment_step)
        nav_prev, nav_next, nav_count = st.columns([1, 1, 2])
        with nav_prev:
            st.button(
                "← Anterior", key="equipment_prev",
                disabled=current_step == 0,
                on_click=set_equipment_step, args=(max(0, current_step - 1),)
            )
        with nav_next:
            st.button(
                "Próxima etapa →", key="equipment_next",
                disabled=current_step == len(equipment_names) - 1,
                type="primary",
                on_click=set_equipment_step,
                args=(min(len(equipment_names) - 1, current_step + 1),)
            )
        with nav_count:
            st.caption(f"Etapa {current_step + 1} de {len(equipment_names)} · siga o sinal da esquerda para a direita")
        current_step = st.select_slider(
            "Etapa da cadeia de áudio",
            options=list(range(len(equipment_names))),
            value=current_step,
            format_func=lambda idx: f"{idx + 1}. {equipment_names[idx]}",
            key="guided_equipment_step"
        )
        selected_equipment = equipment_names[current_step]
        detail = EQUIPMENT_DETAILS[selected_equipment]
        text_col, model_col = st.columns([1, 1.15], gap="large")
        with text_col:
            st.markdown(f"### {current_step + 1}. {selected_equipment}")
            st.markdown(detail["role"])
            st.markdown("**O que entra**")
            st.info(detail["input"])
            st.markdown("**O que sai**")
            st.success(detail["output"])
            st.markdown("**Uma forma de entender**")
            st.write(detail["analogy"])
            st.warning(detail["note"])
        with model_col:
            st.plotly_chart(
                equipment_detail_figure(selected_equipment),
                use_container_width=True,
                config={"displayModeBar": True}
            )
            st.caption(
                f"Modelo 3D: {selected_equipment}. Arraste para girar e use o scroll para aproximar. "
                "Ao avançar, o modelo acompanha a explicação."
            )
        progress_labels = "  →  ".join(
            [f"**{name}**" if idx == current_step else name
             for idx, name in enumerate(equipment_names)]
        )
        st.markdown("**Percurso do sinal**")
        st.markdown(progress_labels)
        st.divider()
        st.markdown("**Sequência completa do sinal**")
        st.markdown(
            "Microfone → Pré-amplificador → Filtro anti-aliasing → ADC → "
            "DSP/computador → DAC → Amplificador → Alto-falante"
        )
        st.caption("Observação: em interfaces reais, pré-amplificador, filtros e conversores costumam estar integrados na mesma placa ou circuito.")
        tab_sampling = st.tabs(["Demonstração: amostragem do ADC"])[0]
        with tab_sampling:
            sample_rate_demo = st.select_slider(
                "Quantidade de amostras por segundo (taxa ilustrativa)",
                options=[8, 10, 20, 40], value=10, format_func=lambda v: f"{v} Hz"
            )
            sample_fig, interval, signal_hz = sampling_demo(sample_rate_demo)
            st.plotly_chart(sample_fig, use_container_width=True)
            st.markdown(
                f"**O que observar:** o sinal azul é contínuo. Os pontos laranja são as medições feitas pelo ADC. "
                f"Com {sample_rate_demo} amostras por segundo, o intervalo entre medições é "
                f"**{interval:.3f} s**. A onda demonstrada tem {signal_hz:.0f} Hz."
            )
            st.info(
                "Aumente a taxa de amostragem e observe os pontos ficarem mais próximos. "
                "O ADC não mede o sinal o tempo todo: ele registra valores em instantes igualmente espaçados."
            )
            st.markdown("""
**E o que acontece depois da medição?**

O ADC também precisa quantizar: arredondar cada medição para um dos níveis digitais permitidos
pela profundidade de bits. Por exemplo, 8 bits permitem 256 níveis; 16 bits permitem 65.536 níveis.
A taxa de amostragem controla quantas medições existem por segundo; a profundidade de bits controla
quantos valores de amplitude podem ser representados.
""")
        st.divider()
        st.subheader("Agora, carregue um áudio para experimentar")
        upload = st.file_uploader("Envie seu áudio para as atividades", type=["wav", "mp3", "ogg", "flac", "webm", "m4a", "aac"], key="course_audio")
        if upload is not None:
            try:
                samples, sample_rate, channels = decode(upload.getvalue(), upload.name)
                st.session_state.audio_data = {
                    "samples": samples, "sr": sample_rate, "channels": channels,
                    "name": upload.name, "raw": upload.getvalue()
                }
                st.session_state.processed_audio = samples.copy()
            except Exception as exc:
                st.error(f"Não consegui abrir o arquivo: {exc}")
        audio = st.session_state.audio_data
        if audio:
            samples, sr, channels = audio["samples"], audio["sr"], audio["channels"]
            peak = float(np.max(np.abs(samples)))
            a, b, c, d = st.columns(4)
            a.metric("Duração", f"{len(samples)/sr:.2f} s")
            b.metric("Amostragem", f"{sr} Hz")
            c.metric("Canais", str(channels))
            d.metric("Nyquist", f"{sr/2:.0f} Hz")
            st.audio(audio["raw"])
            st.markdown("**Atividade:** registre os quatro valores acima no seu caderno de experimentos.")
            st.session_state.audio_loaded = True
        else:
            st.warning("Envie um áudio para liberar as próximas atividades.")
        passed = quiz(1, "Se a taxa de amostragem é 16.000 Hz, qual é a frequência de Nyquist?", ["4.000 Hz", "8.000 Hz", "16.000 Hz"], "8.000 Hz", "Nyquist é metade da taxa de amostragem.")
        if not audio:
            passed = False

    elif stage == 2:
        audio = st.session_state.audio_data
        if not audio:
            st.warning("Primeiro conclua a etapa de aquisição e carregue um áudio.")
            st.stop()
        x, sr = audio["samples"], audio["sr"]
        step = max(1, int(np.ceil(len(x) / 15000)))
        t = np.arange(0, len(x), step) / sr
        fig = go.Figure(go.Scattergl(x=t, y=x[::step], mode="lines", name="Áudio original"))
        fig.update_layout(xaxis_title="Tempo (s)", yaxis_title="Amplitude", height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.metric("RMS", f"{rms(x):.5f}", f"{db(rms(x)):.1f} dBFS")
        st.metric("Pico", f"{np.max(np.abs(x)):.5f}", f"{db(float(np.max(np.abs(x)))):.1f} dBFS")
        st.markdown("**Atividade:** ouça o áudio e identifique um trecho de silêncio e um trecho de maior amplitude.")
        passed = quiz(2, "O que um pico alto na forma de onda indica?", ["Necessariamente melhor inteligibilidade", "Maior amplitude naquele instante", "Maior taxa de amostragem"], "Maior amplitude naquele instante", "Amplitude não é sinônimo de qualidade ou inteligibilidade.")

    elif stage == 3:
        audio = st.session_state.audio_data
        if not audio:
            st.warning("Primeiro conclua a etapa de aquisição e carregue um áudio.")
            st.stop()
        x, sr = audio["samples"], audio["sr"]
        nyquist = sr / 2
        start = st.slider("Início do trecho (s)", 0.0, max(0.0, len(x)/sr - 0.1), 0.0, 0.05)
        duration = min(3.0, max(0.1, len(x)/sr - start))
        first = int(start * sr)
        segment = x[first:first + int(duration * sr)]
        if len(segment) >= 8:
            window = np.hanning(len(segment))
            spectrum = np.fft.rfft(segment * window)
            frequencies = np.fft.rfftfreq(len(segment), 1/sr)
            magnitude = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-8))
            fig = go.Figure(go.Scattergl(x=frequencies, y=magnitude, mode="lines"))
            fig.update_layout(xaxis_title="Frequência (Hz)", yaxis_title="Magnitude (dB)", height=320)
            fig.update_xaxes(range=[0, min(10000, nyquist)])
            st.plotly_chart(fig, use_container_width=True)
        nperseg = min(1024, len(x))
        if nperseg >= 32:
            f, times, power = spectrogram(x, fs=sr, nperseg=nperseg, noverlap=nperseg//2)
            fig2 = go.Figure(go.Heatmap(x=times, y=f, z=10*np.log10(np.maximum(power, 1e-12)), colorbar=dict(title="dB/Hz")))
            fig2.update_layout(xaxis_title="Tempo (s)", yaxis_title="Frequência (Hz)", height=380)
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("**Atividade:** compare o espectro de uma pausa com o de um trecho com fala.")
        passed = quiz(3, "Qual visualização mostra como a energia por frequência muda ao longo do tempo?", ["Forma de onda", "Espectrograma", "Métrica RMS"], "Espectrograma", "O espectrograma combina tempo, frequência e energia.")

    elif stage == 4:
        audio = st.session_state.audio_data
        if not audio:
            st.warning("Primeiro conclua a etapa de aquisição e carregue um áudio.")
            st.stop()
        x, sr = audio["samples"], audio["sr"]
        hp_on = st.checkbox("Ativar passa-alta", value=False)
        hp = st.slider("Corte passa-alta (Hz)", 20, min(500, max(21, sr//3)), min(80, max(20, sr//3-1)), 5, disabled=not hp_on)
        lp_on = st.checkbox("Ativar passa-baixa", value=False)
        lp_max = max(1001, min(10000, sr//2-1))
        lp = st.slider("Corte passa-baixa (Hz)", 1000, lp_max, min(7500, lp_max), 100, disabled=not lp_on)
        gain = st.slider("Ganho (dB)", -12, 12, 0)
        reduction = st.slider("Redução de ruído (%)", 0, 90, 0, 5)
        normalize = st.checkbox("Normalizar pico para -3 dBFS")
        y = x.copy()
        try:
            if hp_on:
                y = sosfiltfilt(butter(4, hp, btype="highpass", fs=sr, output="sos"), y)
            if lp_on:
                y = sosfiltfilt(butter(4, lp, btype="lowpass", fs=sr, output="sos"), y)
            if reduction:
                import noisereduce as nr
                y = nr.reduce_noise(y=y, sr=sr, prop_decrease=reduction/100, stationary=True)
            y *= 10 ** (gain/20)
            if normalize and np.max(np.abs(y)) > 0:
                y *= (10**(-3/20)) / np.max(np.abs(y))
            st.session_state.processed_audio = y
        except Exception as exc:
            st.error(f"Falha no processamento: {exc}")
            st.stop()
        st.write("Ouça e compare:")
        left, right = st.columns(2)
        with left:
            st.markdown("**Original**")
            st.audio(audio["raw"])
        with right:
            st.markdown("**Processado**")
            st.audio(wav_bytes(y, sr), format="audio/wav")
        st.download_button("Baixar áudio processado", wav_bytes(y, sr), "audio_processado.wav", "audio/wav")
        st.markdown("**Atividade:** teste primeiro apenas um filtro. Depois compare com o áudio sem processamento.")
        passed = quiz(4, "O que a normalização de pico faz?", ["Remove todo o ruído", "Ajusta o nível do pico", "Aumenta a taxa de amostragem"], "Ajusta o nível do pico", "Normalização altera o nível; não remove ruído.")

    elif stage == 5:
        audio = st.session_state.audio_data
        if not audio:
            st.warning("Primeiro conclua a etapa de aquisição e carregue um áudio.")
            st.stop()
        st.info("A primeira execução baixa os pesos do modelo. Em computadores modestos, comece com tiny ou base.")
        model_size = st.selectbox("Modelo", ["tiny", "base", "small", "medium"], index=1)
        language_label = st.selectbox("Idioma", ["Auto detectar", "Português", "Inglês", "Espanhol"])
        language = {"Auto detectar": None, "Português": "pt", "Inglês": "en", "Espanhol": "es"}[language_label]
        beam = st.select_slider("Beam size", [1, 3, 5, 8], value=5)
        reference = st.text_area("Transcrição de referência revisada (necessária para WER/CER)")
        if st.button("Transcrever original e processado", type="primary"):
            try:
                from faster_whisper import WhisperModel
                from jiwer import wer, cer
                with st.spinner("Carregando o modelo e transcrevendo..."):
                    model = WhisperModel(model_size, device="cpu", compute_type="int8")
                    def transcribe(signal):
                        if audio["sr"] != 16000:
                            divisor = math.gcd(audio["sr"], 16000)
                            signal = resample_poly(signal, 16000//divisor, audio["sr"]//divisor)
                        segments, _ = model.transcribe(np.asarray(signal, dtype=np.float32), language=language, beam_size=beam, vad_filter=True)
                        return " ".join(s.text.strip() for s in segments).strip()
                    original_text = transcribe(audio["samples"])
                    processed = st.session_state.processed_audio
                    processed_text = transcribe(processed if processed is not None else audio["samples"])
                st.session_state.asr_pair = (original_text, processed_text)
                st.success("Transcrição concluída.")
            except Exception as exc:
                st.error(f"Não foi possível transcrever: {exc}")
        if "asr_pair" in st.session_state:
            original_text, processed_text = st.session_state.asr_pair
            a, b = st.columns(2)
            a.text_area("Original", original_text, height=140)
            b.text_area("Processado", processed_text, height=140)
            if reference.strip():
                from jiwer import wer, cer
                a.metric("WER original", f"{wer(reference, original_text)*100:.2f}%")
                b.metric("WER processado", f"{wer(reference, processed_text)*100:.2f}%")
                a.metric("CER original", f"{cer(reference, original_text)*100:.2f}%")
                b.metric("CER processado", f"{cer(reference, processed_text)*100:.2f}%")
        passed = quiz(5, "Para calcular WER de forma válida, você precisa de:", ["Apenas o áudio", "Uma transcrição de referência revisada", "Somente a taxa de amostragem"], "Uma transcrição de referência revisada", "WER compara a hipótese do ASR com um texto de referência.")

    elif stage == 6:
        st.markdown("### Entrega do projeto")
        st.write("Use este roteiro para documentar um experimento reproduzível.")
        st.text_area("Problema e objetivo", key="final_problem")
        st.text_area("Características do áudio e hipótese", key="final_characteristics")
        st.text_area("Parâmetros de processamento e justificativa", key="final_parameters")
        st.text_area("Comparação auditiva e visual", key="final_comparison")
        st.text_area("Resultado de ASR / WER / CER (se aplicável)", key="final_metrics")
        report = "\n".join([
            "# Relatório — Projeto final",
            "## Problema e objetivo\n" + st.session_state.get("final_problem", ""),
            "## Características e hipótese\n" + st.session_state.get("final_characteristics", ""),
            "## Parâmetros\n" + st.session_state.get("final_parameters", ""),
            "## Comparação\n" + st.session_state.get("final_comparison", ""),
            "## Métricas\n" + st.session_state.get("final_metrics", ""),
        ])
        st.download_button("Baixar relatório em Markdown", report, "relatorio_experimento_audio.md", "text/markdown")
        passed = all(st.session_state.get(k, "").strip() for k in ["final_problem", "final_characteristics", "final_parameters", "final_comparison"])

    st.divider()
    st.markdown("### Atividade desta etapa")
    st.info(LESSONS[stage]["task"])
    if passed:
        st.success("Verificação concluída. Você pode marcar esta etapa como concluída.")
        if st.button("Concluir etapa e liberar a próxima", type="primary", key=f"complete_{stage}"):
            if stage not in st.session_state.completed_stages:
                st.session_state.completed_stages.append(stage)
            if stage < 6:
                st.session_state.navigation_target = STAGES[stage + 1]
            st.rerun()
    else:
        st.caption("Conclua a atividade e acerte a verificação para liberar a próxima etapa.")

st.divider()
st.caption("O laboratório técnico original continua disponível em audio_engineering_lab.py.")
