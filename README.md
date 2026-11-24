# Audio Engineering Lab

Curso prático e laboratório de processamento digital de sinais de áudio, desenvolvido em Python e Streamlit.

O projeto possui três experiências:

- **Curso guiado** (`audio_engineering_course.py`): trilha progressiva do básico ao avançado, com explicações, experimentos e verificações de aprendizagem.
- **Laboratório completo** (`audio_engineering_lab.py`): ambiente técnico para explorar aquisição, forma de onda, FFT, espectrograma, DSP e reconhecimento automático de fala.
- **Smart Audio Calibration** (`smart_audio_calibration_app.py`): calibração automática orientada à transcrição. O usuário grava frases conhecidas, o sistema compara perfis conservadores de DSP com Faster-Whisper, calcula WER/CER e recomenda um perfil somente quando há melhora mensurável. Inclui reconhecimento independente pelo Web Speech API do navegador.

## Smart Audio Calibration

Após instalar as dependências, inicie:

```bash
chmod +x iniciar_calibracao.sh
./iniciar_calibracao.sh
```

Grave uma ou mais frases de referência na interface e execute a calibração. Para uma avaliação mais confiável, grave todas as frases, mantenha microfone e ambiente constantes e valide o perfil recomendado com frases diferentes das usadas na calibração.

**Importante:** a calibração automática utiliza Faster-Whisper como mecanismo de avaliação do WER/CER. O reconhecimento do navegador (Web Speech API) é uma comparação em tempo real, não uma transcrição do arquivo filtrado. A API do navegador depende de suporte do Chrome/Edge, permissão do microfone e conectividade. A primeira execução do Whisper pode baixar o modelo e demorar em CPU.

O perfil atual testa uma pequena grade de configurações: áudio original, passa-alta em 70/100 Hz e redução leve/moderada de ruído em combinações conservadoras. O áudio original nunca é sobrescrito. Se nenhum candidato melhorar o WER do baseline, o sistema recomenda manter o original.

## Documentação para desenvolvimento\n\n- [AGENTS.md](AGENTS.md): instruções, arquitetura, regras de segurança, testes e critérios de conclusão para agentes de desenvolvimento.\n- [Especificação do Smart Audio Calibration](docs/SMART_AUDIO_CALIBRATION.md): pipeline, perfis DSP, métricas, mecanismos ASR e protocolo de validação.\n\n## Trilha de aprendizagem

1. **Comece aqui** — apresentação da metodologia.
2. **Fundamentos e aquisição** — amostragem, canais, duração e Nyquist.
3. **Forma de onda** — amplitude, RMS, pico e clipping.
4. **Frequência e espectrograma** — FFT, energia e resolução tempo-frequência.
5. **Processamento digital (DSP)** — filtros, redução de ruído, ganho e normalização.
6. **Reconhecimento de fala (ASR)** — transcrição, WER e CER.
7. **Projeto final** — experimento documentado e relatório em Markdown.

## Requisitos

- Ubuntu 22.04 ou superior (ou distribuição Debian compatível);
- Python 3.10 ou superior;
- conexão com a internet durante a instalação;
- FFmpeg para formatos como MP3, WebM, M4A e AAC.

## Instalação rápida (Ubuntu)

```bash
git clone https://github.com/GiovaniRodrigo/audio-engineer.git
cd audio-engineer
chmod +x instalar.sh iniciar.sh iniciar_calibracao.sh
./instalar.sh
```

Para iniciar o curso guiado:

```bash
./iniciar.sh
```

Para iniciar a calibração inteligente:

```bash
./iniciar_calibracao.sh
```

O Streamlit exibirá um endereço local, normalmente http://localhost:8501.

## Abrir o laboratório técnico completo

```bash
.venv/bin/streamlit run audio_engineering_lab.py
```

## Atualizar uma cópia já clonada

```bash
git pull
chmod +x instalar.sh iniciar.sh iniciar_calibracao.sh
./instalar.sh
```

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
├── smart_audio_calibration.py
├── smart_audio_calibration_app.py
├── browser_asr_component/
│   └── index.html
├── requirements.txt
├── instalar.sh
├── iniciar.sh
└── iniciar_calibracao.sh
```

## Observações científicas e limitações

- RMS e pico descrevem amplitude; não demonstram, isoladamente, melhoria da inteligibilidade.
- SNR exige uma referência adequada de sinal/ruído ou um método explícito de estimação.
- WER/CER exigem uma transcrição de referência revisada. O WER da calibração é medido nas frases fornecidas e pode superajustar a esse conjunto.
- A conversão multicanal para mono pela média pode causar cancelamento de fase; a calibração seleciona o canal com maior RMS para evitar essa média.
- Equalização não recupera informação perdida por clipping, não resolve microfone distante e não remove qualquer tipo de ruído.
- A API Web Speech é dependente do navegador e pode enviar áudio ao serviço de reconhecimento do próprio navegador.
- A primeira execução de ASR pode baixar os pesos do modelo e levar mais tempo, especialmente em CPU.

## Licença

Nenhuma licença foi definida neste repositório ainda. Consulte o proprietário antes de reutilizar ou redistribuir o código.
