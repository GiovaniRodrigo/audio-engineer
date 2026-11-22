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


def _wave_surface(phase, offset=0.0, direction=1.0, amplitude=0.22):
    """A flowing 3D ribbon: a visual metaphor for pressure-wave propagation."""
    x = np.linspace(-1.75, 1.75, 42)
    y = np.linspace(-0.34, 0.34, 9) + offset
    xx, yy = np.meshgrid(x, y)
    zz = amplitude * np.sin(2.6 * xx - phase * direction)
    # A subtle second harmonic gives the ribbon a richer, less mechanical profile.
    zz += 0.045 * np.sin(5.2 * xx - phase * direction * 1.7)
    return go.Surface(
        x=xx, y=yy, z=zz, surfacecolor=zz,
        colorscale=[[0.0, "#075985"], [0.35, "#0ea5e9"], [0.65, "#67e8f9"], [1.0, "#0c4a6e"]],
        cmin=-0.30, cmax=0.30, showscale=False, opacity=0.78,
        hoverinfo="skip", lighting=dict(ambient=0.48, diffuse=0.8, specular=0.45, roughness=0.35)
    )


def equipment_animation_figure(name, camera_view="isometric"):
    """Animate 3D audio-wave ribbons around the selected equipment."""
    fig = equipment_detail_figure(name)
    kind = EQUIPMENT_DETAILS[name]["kind"]
    frame_count = 36
    frames = []
    wave_offsets = [-0.48, 0.0, 0.48]
    wave_amplitude = 0.20 if kind == "microphone" else 0.24
    wave_indices = []
    for offset in wave_offsets:
        fig.add_trace(_wave_surface(0.0, offset, amplitude=wave_amplitude))
        wave_indices.append(len(fig.data) - 1)

    # Additional component-specific motion remains visible alongside the traveling waves.
    motion_index = None
    if kind == "adc":
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[], mode="markers",
            marker=dict(size=7, color="#fbbf24", line=dict(width=1, color="#fff")),
            name="Amostras digitais", hoverinfo="skip"
        ))
        motion_index = len(fig.data) - 1
    elif kind == "speaker":
        angles = np.linspace(0, 2*np.pi, 64)
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[], mode="lines",
            line=dict(color="#fbbf24", width=6), name="Vibração do cone", hoverinfo="skip"
        ))
        motion_index = len(fig.data) - 1

    for i in range(frame_count):
        phase = 2*np.pi*i/frame_count
        data = [_wave_surface(phase, offset, amplitude=wave_amplitude) for offset in wave_offsets]
        traces = list(wave_indices)
        if kind == "adc":
            xs = np.linspace(-0.75, 0.75, 13)
            ys = np.full_like(xs, -0.78)
            analog = 0.34*np.sin(2*np.pi*xs - phase)
            quantized = np.round(analog*6)/6
            data.append(go.Scatter3d(x=xs, y=ys, z=0.34+quantized, mode="markers"))
            traces.append(motion_index)
        elif kind == "speaker":
            displacement = 0.09*np.sin(phase*2)
            x = 0.48*np.cos(angles)
            z = 0.10 + displacement + 0.48*np.sin(angles)
            y = np.full_like(x, -0.82-displacement)
            data.append(go.Scatter3d(x=x, y=y, z=z, mode="lines"))
            traces.append(motion_index)
        frames.append(go.Frame(data=data, traces=traces, name=str(i)))

    camera_presets = {
        "isometric": dict(eye=dict(x=1.65, y=2.05, z=1.35), up=dict(x=0, y=0, z=1), projection=dict(type="perspective")),
        "front": dict(eye=dict(x=0, y=3.2, z=0.35), up=dict(x=0, y=0, z=1), projection=dict(type="perspective")),
        "side": dict(eye=dict(x=3.2, y=0, z=0.35), up=dict(x=0, y=0, z=1), projection=dict(type="perspective")),
        "top": dict(eye=dict(x=0.05, y=0.05, z=3.6), up=dict(x=0, y=1, z=0), projection=dict(type="perspective")),
    }
    camera = camera_presets.get(camera_view, camera_presets["isometric"])
    fig.frames = frames
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", camera=camera, dragmode="orbit"
        ),
        updatemenus=[
            dict(
                type="buttons", showactive=False, x=0.01, y=0.02,
                xanchor="left", yanchor="bottom", direction="right",
                buttons=[
                    dict(label="▶ Reproduzir", method="animate",
                         args=[None, dict(frame=dict(duration=90, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                    dict(label="Ⅱ Pausar", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")])
                ]
            ),
            dict(
                type="buttons", showactive=False, x=0.99, y=1.02,
                xanchor="right", yanchor="top", direction="down",
                buttons=[
                    dict(label="Isométrica", method="relayout", args=[{"scene.camera": camera_presets["isometric"]}]),
                    dict(label="Frontal", method="relayout", args=[{"scene.camera": camera_presets["front"]}]),
                    dict(label="Lateral", method="relayout", args=[{"scene.camera": camera_presets["side"]}]),
                    dict(label="Superior", method="relayout", args=[{"scene.camera": camera_presets["top"]}])
                ]
            )
        ],
        height=650, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=28, b=35),
        uirevision=f"audio-equipment-{name}"
    )
    return fig

def set_equipment_step(step: int):
    """Update the synchronized text/3D guided lesson step."""
    st.session_state.guided_equipment_step = int(step)


def add_torus(fig, x, y, z, major_radius, minor_radius, color, axis="z", resolution=48, tube=12):
    """Add a shaded torus mesh; useful for bezels, grille rims and speaker surrounds."""
    u = np.linspace(0, 2*np.pi, resolution, endpoint=False)
    v = np.linspace(0, 2*np.pi, tube, endpoint=False)
    uu, vv = np.meshgrid(u, v)
    ring = major_radius + minor_radius*np.cos(vv)
    a = ring*np.cos(uu)
    b = ring*np.sin(uu)
    c = minor_radius*np.sin(vv)
    if axis == "y":
        xx, yy, zz = x+a, y+c, z+b
    else:
        xx, yy, zz = x+a, y+b, z+c
    xx, yy, zz = xx.ravel(), yy.ravel(), zz.ravel()
    ii, jj, kk = [], [], []
    for j in range(tube):
        for i in range(resolution):
            p = j*resolution+i
            pn = j*resolution+(i+1)%resolution
            q = ((j+1)%tube)*resolution+i
            qn = ((j+1)%tube)*resolution+(i+1)%resolution
            ii.extend([p, pn]); jj.extend([q, q]); kk.extend([qn, qn])
    fig.add_trace(go.Mesh3d(
        x=xx, y=yy, z=zz, i=ii, j=jj, k=kk, color=color,
        flatshading=False, lighting=dict(ambient=0.42, diffuse=0.78, specular=0.72, roughness=0.28, fresnel=0.18),
        lightposition=dict(x=120, y=180, z=240), showscale=False, hoverinfo="skip"
    ))


def add_sphere(fig, x, y, z, sx, sy, sz, color, opacity=1.0):
    u = np.linspace(0, 2*np.pi, 28)
    v = np.linspace(0, np.pi, 18)
    uu, vv = np.meshgrid(u, v)
    xx = x + sx*np.cos(uu)*np.sin(vv)
    yy = y + sy*np.sin(uu)*np.sin(vv)
    zz = z + sz*np.cos(vv)
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz, surfacecolor=np.zeros_like(xx),
        colorscale=[[0, color], [1, color]], showscale=False,
        opacity=opacity, hoverinfo="skip",
        lighting=dict(ambient=0.35, diffuse=0.8, specular=0.85, roughness=0.22, fresnel=0.2)
    ))


def equipment_detail_figure(name):
    """Create a more product-like 3D model with recognizable physical components."""
    kind = EQUIPMENT_DETAILS[name]["kind"]
    fig = go.Figure()

    if kind == "microphone":
        # Studio condenser microphone: cylindrical body, capsule, grille, rings and yoke.
        add_cylinder(fig, 0, 0, 0.12, 0.255, 1.42, "#202a36", 64)
        add_cylinder(fig, 0, 0, 0.83, 0.275, 0.48, "#8795a5", 64)
        add_cylinder(fig, 0, 0, 1.12, 0.27, 0.12, "#c4ced8", 64)
        add_cylinder(fig, 0, 0, 0.56, 0.263, 0.055, "#b7c1cb", 64)
        add_cylinder(fig, 0, 0, -0.66, 0.075, 0.48, "#8795a5", 32)
        add_box(fig, 0, 0, -0.94, 0.78, 0.07, 0.07, "#354253")
        # Fine grille slots and dark acoustic mesh.
        for zline in np.linspace(0.68, 1.02, 13):
            fig.add_trace(go.Scatter3d(
                x=[-0.20, 0.20], y=[-0.19, -0.19], z=[zline, zline],
                mode="lines", line=dict(color="#e5edf5", width=2), hoverinfo="skip", showlegend=False
            ))
        for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
            x = 0.278*np.cos(angle); y = 0.278*np.sin(angle)
            fig.add_trace(go.Scatter3d(
                x=[x, x], y=[y, y], z=[0.66, 1.03], mode="lines",
                line=dict(color="#536273", width=1), hoverinfo="skip", showlegend=False
            ))
        add_torus(fig, 0, 0, 1.17, 0.245, 0.025, "#dbe3eb")
        add_torus(fig, 0, 0, 0.54, 0.258, 0.018, "#dbe3eb")

    elif kind in ("preamp", "amplifier"):
        # Brushed-metal rack enclosure with front panel, rotary controls and connectors.
        add_box(fig, 0, 0, 0, 2.35, 1.38, 0.72, "#263342")
        add_box(fig, 0, -0.705, 0.02, 2.22, 0.045, 0.57, "#9aa8b6")
        add_box(fig, 0, 0.05, 0.385, 2.15, 1.1, 0.035, "#3d4b5b")
        for x in [-0.78, -0.25, 0.28, 0.81]:
            add_cylinder(fig, x, -0.77, 0.06, 0.135, 0.12, "#202a36", 40)
            add_cylinder(fig, x, -0.84, 0.06, 0.09, 0.08, "#c2a56c", 40)
            add_torus(fig, x, -0.845, 0.06, 0.105, 0.012, "#e6d4ad", axis="y")
            fig.add_trace(go.Scatter3d(
                x=[x, x], y=[-0.89, -0.89], z=[0.06, 0.15],
                mode="lines", line=dict(color="#f8fafc", width=3), hoverinfo="skip", showlegend=False
            ))
        for x in [-1.02, 1.02]:
            add_cylinder(fig, x, -0.77, -0.18, 0.075, 0.12, "#d5dde5", 32)
            add_cylinder(fig, x, -0.84, -0.18, 0.035, 0.08, "#202a36", 24)
        for x in np.linspace(-0.9, 0.9, 7):
            add_sphere(fig, x, -0.765, 0.27, 0.025, 0.018, 0.025, "#4ade80" if x < 0 else "#f59e0b")
        if kind == "amplifier":
            add_box(fig, 0, 0.70, -0.05, 1.45, 0.12, 0.3, "#788797")
            for x in np.linspace(-0.55, 0.55, 9):
                add_box(fig, x, 0.73, 0.13, 0.035, 0.08, 0.26, "#c1ccd6")

    elif kind == "filter":
        # PCB with traces, capacitors, resistors and an inductor.
        add_box(fig, 0, 0, -0.08, 2.05, 1.35, 0.16, "#12634f")
        for y in [-0.56, 0.56]:
            add_box(fig, 0, y, 0.015, 1.92, 0.025, 0.025, "#d5a84b")
        for x in np.linspace(-0.78, 0.78, 8):
            add_cylinder(fig, x, -0.61, 0.02, 0.035, 0.07, "#d8c7a0", 16)
        for x in [-0.66, -0.18, 0.32, 0.72]:
            add_box(fig, x, 0.0, 0.13, 0.18, 0.24, 0.22, "#202a36")
            add_box(fig, x, -0.17, 0.14, 0.08, 0.08, 0.08, "#c7a56a")
        for x, y, h in [(-0.42, 0.32, 0.34), (0.08, 0.32, 0.28), (0.55, -0.32, 0.38)]:
            add_cylinder(fig, x, y, 0.12+h/2, 0.095, h, "#b9c7d5", 32)
            add_cylinder(fig, x, y, 0.12+h+0.015, 0.098, 0.035, "#64748b", 32)
        add_cylinder(fig, 0.0, -0.35, 0.22, 0.16, 0.32, "#d5b36b", 32)

    elif kind in ("adc", "dac"):
        # Converter IC mounted on a PCB, with pin rows and input/output connectors.
        add_box(fig, 0, 0, -0.08, 1.85, 1.35, 0.16, "#12634f")
        add_box(fig, 0, 0, 0.12, 0.92, 0.72, 0.25, "#202a36")
        add_box(fig, 0, 0, 0.27, 0.74, 0.55, 0.08, "#39495a")
        add_box(fig, 0, 0, 0.325, 0.43, 0.28, 0.035, "#111827")
        for x in [-0.40, 0.40]:
            for y in np.linspace(-0.27, 0.27, 8):
                add_box(fig, x, y, 0.12, 0.13, 0.035, 0.045, "#cbd5e1")
        for x in np.linspace(-0.72, 0.72, 9):
            for y in [-0.53, 0.53]:
                add_box(fig, x, y, 0.10, 0.055, 0.16, 0.06, "#d9e1e8")
        for x in [-0.77, 0.77]:
            add_cylinder(fig, x, 0, 0.17, 0.11, 0.22, "#c1ccd6", 32)
            add_cylinder(fig, x, 0, 0.29, 0.055, 0.05, "#263342", 24)
        add_sphere(fig, -0.58, 0.34, 0.18, 0.055, 0.055, 0.055, "#f59e0b")
        add_sphere(fig, 0.58, -0.34, 0.18, 0.055, 0.055, 0.055, "#4ade80")

    elif kind == "computer":
        # Desktop monitor with bezel, glass display, stand and keyboard.
        add_box(fig, 0, 0.05, 0.58, 1.95, 0.15, 1.28, "#202a36")
        add_box(fig, 0, -0.04, 0.59, 1.78, 0.035, 1.10, "#10283b")
        # Screen visualization: subtle waveform and signal trace.
        xs = np.linspace(-0.78, 0.78, 90)
        ys = -0.067*np.ones_like(xs)
        zs = 0.59 + 0.20*np.sin(xs*8)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="#38bdf8", width=5), hoverinfo="skip", showlegend=False))
        add_box(fig, 0, 0.1, -0.18, 0.20, 0.48, 0.28, "#8997a5")
        add_box(fig, 0, 0.12, -0.34, 0.85, 0.58, 0.08, "#aeb9c4")
        add_box(fig, 0, -0.65, -0.30, 1.35, 0.42, 0.10, "#374557")
        for x in np.linspace(-0.58, 0.58, 12):
            for z in [-0.27, -0.30]:
                add_box(fig, x, -0.88, z, 0.065, 0.025, 0.018, "#d1dae3")

    elif kind == "speaker":
        # Loudspeaker cabinet with mounting rim, surround, cone, dust cap and terminals.
        add_box(fig, 0, 0.02, 0, 1.78, 1.18, 2.12, "#263342")
        add_box(fig, 0, -0.59, 0, 1.64, 0.055, 1.98, "#111827")
        add_torus(fig, 0, -0.635, 0.10, 0.67, 0.065, "#8997a5", axis="y")
        add_torus(fig, 0, -0.675, 0.10, 0.55, 0.045, "#202a36", axis="y")
        add_cone(fig, 0, -0.69, 0.10, 0.52, 0.40, "#8795a5")
        add_torus(fig, 0, -0.72, 0.10, 0.17, 0.035, "#c1ccd6", axis="y")
        add_cylinder(fig, 0, -0.79, 0.10, 0.15, 0.12, "#263342", 48)
        add_cylinder(fig, 0, -0.86, 0.10, 0.09, 0.035, "#c1ccd6", 40)
        for x in [-0.72, 0.72]:
            for z in [-0.82, 0.98]:
                add_cylinder(fig, x, -0.64, z, 0.035, 0.06, "#d5dde5", 20)

    fig.update_layout(
        height=590, margin=dict(l=0, r=0, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=2.15, z=1.35), projection=dict(type="perspective"))
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
        st.subheader("A cadeia de áudio")
        st.markdown(
            "Explore cada componente da cadeia. A explicação, o modelo e a animação acompanham a etapa selecionada."
        )
        equipment_names = list(EQUIPMENT_DETAILS.keys())
        if "guided_equipment_step" not in st.session_state:
            st.session_state.guided_equipment_step = 0
        current_step = int(st.session_state.guided_equipment_step)

        st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] { align-items: stretch; }
        .equipment-panel {
            border: 1px solid rgba(148,163,184,.24);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: linear-gradient(145deg, rgba(30,41,59,.52), rgba(15,23,42,.22));
            min-height: 100%;
        }
        .equipment-kicker {
            color: #38bdf8; font-size: .76rem; font-weight: 700;
            letter-spacing: .10em; text-transform: uppercase;
        }
        .equipment-flow {
            border: 1px solid rgba(148,163,184,.22); border-radius: 12px;
            padding: .8rem; background: rgba(15,23,42,.35);
        }
        </style>
        """, unsafe_allow_html=True)

        nav_prev, nav_next, nav_count = st.columns([1, 1, 2])
        with nav_prev:
            st.button(
                "← Anterior", key="equipment_prev",
                disabled=current_step == 0,
                on_click=set_equipment_step, args=(max(0, current_step - 1),),
                use_container_width=True
            )
        with nav_next:
            st.button(
                "Próxima etapa →", key="equipment_next",
                disabled=current_step == len(equipment_names) - 1,
                type="primary", on_click=set_equipment_step,
                args=(min(len(equipment_names) - 1, current_step + 1),),
                use_container_width=True
            )
        with nav_count:
            st.caption(f"COMPONENTE {current_step + 1:02d} / {len(equipment_names):02d}")
            st.progress((current_step + 1) / len(equipment_names))

        current_step = st.select_slider(
            "Percurso da cadeia",
            options=list(range(len(equipment_names))),
            format_func=lambda idx: f"{idx + 1}. {equipment_names[idx]}",
            key="guided_equipment_step"
        )
        selected_equipment = equipment_names[current_step]
        detail = EQUIPMENT_DETAILS[selected_equipment]
        text_col, model_col = st.columns([0.82, 1.55], gap="large")

        with text_col:
            with st.container(border=True):
                st.markdown(f'<div class="equipment-kicker">Etapa {current_step + 1} · cadeia de áudio</div>', unsafe_allow_html=True)
                st.markdown(f"### {selected_equipment}")
                st.markdown(detail["role"])
                st.divider()
                st.markdown("**SINAL DE ENTRADA**")
                st.info(detail["input"], icon="↘️")
                st.markdown("**SINAL DE SAÍDA**")
                st.success(detail["output"], icon="↗️")
                st.markdown("**Como entender**")
                st.write(detail["analogy"])
                with st.expander("Detalhe técnico", expanded=False):
                    st.write(detail["note"])

        with model_col:
            with st.container(border=True):
                st.markdown('<div class="equipment-kicker">Visualização tridimensional interativa</div>', unsafe_allow_html=True)
                st.markdown(f"#### {selected_equipment}")
                st.plotly_chart(
                    equipment_animation_figure(selected_equipment),
                    use_container_width=True,
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["toImage", "lasso3d", "select3d"]
                    },
                    key=f"equipment_model_{current_step}"
                )
                st.caption("Use ▶ Reproduzir e Ⅱ Pausar no gráfico. Arraste para girar e use a roda do mouse para aproximar.")
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
