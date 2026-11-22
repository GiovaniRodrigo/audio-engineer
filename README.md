# Audio Engineering Lab

Curso prático e laboratório de processamento digital de sinais de áudio, desenvolvido em Python e Streamlit.

O projeto possui duas experiências:

- **Curso guiado** (`audio_engineering_course.py`): trilha progressiva do básico ao avançado, com explicações, experimentos, atividades e verificações de aprendizagem. A etapa inicial permite selecionar equipamentos (microfone, pré-amplificador, filtro, ADC, computador/DSP, DAC, amplificador e alto-falante), visualizar um modelo 3D individual e ler a explicação ao lado. Também inclui uma demonstração visual da amostragem do ADC.
- **Laboratório completo** (`audio_engineering_lab.py`): ambiente técnico para explorar aquisição, forma de onda, FFT, espectrograma, DSP e reconhecimento automático de fala.

## Trilha de aprendizagem

1. **Comece aqui** — apresentação da metodologia.
2. **Fundamentos e aquisição** — amostragem, canais, duração e Nyquist.
3. **Forma de onda** — amplitude, RMS, pico e clipping.
4. **Frequência e espectrograma** — FFT, energia e resolução tempo-frequência.
5. **Processamento digital (DSP)** — passa-alta, passa-baixa, redução de ruído, ganho e normalização.
6. **Reconhecimento de fala (ASR)** — transcrição, WER e CER.
7. **Projeto final** — experimento documentado e relatório em Markdown.

As etapas são liberadas progressivamente após a atividade e a verificação de aprendizagem. O áudio original é preservado.

## Requisitos

- Ubuntu 22.04 ou superior (ou distribuição Debian compatível);
- Python 3.10 ou superior;
- conexão com a internet durante a instalação;
- FFmpeg para formatos como MP3, WebM, M4A e AAC.

## Instalação rápida (Ubuntu)

```bash
git clone https://github.com/GiovaniRodrigo/audio-engineer.git
cd audio-engineer
chmod +x instalar.sh iniciar.sh
./instalar.sh
```

Para iniciar o curso guiado:

```bash
./iniciar.sh
```

O Streamlit exibirá um endereço local, normalmente http://localhost:8501.

## Abrir o laboratório técnico completo

Depois de executar a instalação, use:

```bash
.venv/bin/streamlit run audio_engineering_lab.py
```

## Atualizar uma cópia já clonada

```bash
git pull
chmod +x instalar.sh iniciar.sh
./instalar.sh
./iniciar.sh
```

Se as dependências não tiverem mudado, não é necessário executar novamente o instalador após cada atualização.

## Instalação manual

```bash
sudo apt update
sudo apt install -y ffmpeg python3-venv python3-pip
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run audio_engineering_course.py
```

## Estrutura principal

```text
.
├── audio_engineering_course.py
├── audio_engineering_lab.py
├── requirements.txt
├── instalar.sh
├── iniciar.sh
└── README.md
```

## Observações científicas

- RMS e pico descrevem amplitude; não demonstram, isoladamente, melhoria da inteligibilidade.
- SNR exige uma referência adequada de sinal/ruído ou um método explícito de estimação.
- WER/CER exigem uma transcrição de referência revisada.
- A conversão multicanal para mono pela média pode causar cancelamento de fase.
- A primeira execução de ASR pode baixar os pesos do modelo e levar mais tempo, especialmente em CPU.

## Licença

Nenhuma licença foi definida neste repositório ainda. Consulte o proprietário antes de reutilizar ou redistribuir o código.
