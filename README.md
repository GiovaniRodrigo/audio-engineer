# Audio Engineering Lab

Laboratório didático de processamento digital de sinais de áudio, desenvolvido em Python e Streamlit.

## Estado do projeto

A versão inicial disponibiliza:
- aquisição e caracterização de arquivos de áudio;
- visualização da forma de onda;
- análise espectral por FFT;
- espectrograma;
- filtros passa-alta e passa-baixa;
- redução espectral de ruído, ganho e normalização;
- comparação entre áudio original e processado.

O reconhecimento automático de fala (ASR), métricas WER/CER e histórico persistente de experimentos ainda são etapas futuras.

## Requisitos

- Python 3.10 ou superior;
- FFmpeg para formatos como MP3, WebM, M4A e AAC;
- dependências listadas em `requirements.txt`.

## Instalação no Ubuntu

```bash
sudo apt update
sudo apt install -y ffmpeg python3-venv python3-pip

git clone https://github.com/GiovaniRodrigo/audio-engineer.git
cd audio-engineer

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

streamlit run audio_engineering_lab.py
```

Abra o endereço local exibido pelo Streamlit (normalmente http://localhost:8501).

## Observações científicas

- RMS e pico caracterizam nível/amplitude; não demonstram, isoladamente, melhoria da inteligibilidade.
- SNR exige referência adequada de sinal/ruído ou um método explícito de estimação.
- WER/CER exigem uma transcrição de referência revisada.
- O áudio original é preservado; os ajustes são aplicados a uma cópia mono.
- A conversão de multicanal para mono pela média pode causar cancelamento de fase em alguns arquivos.

## Estrutura

```text
.
├── audio_engineering_lab.py
├── requirements.txt
└── README.md
```

## Licença

Nenhuma licença foi definida neste repositório ainda. Consulte o proprietário antes de reutilizar ou redistribuir o código.
