# Audio Engineering Lab

Laboratório didático de processamento digital de sinais de áudio, desenvolvido em Python e Streamlit.

## Requisitos

- Ubuntu 22.04 ou superior (ou distribuição Debian compatível);
- Python 3.10 ou superior;
- conexão com a internet durante a instalação das dependências;
- FFmpeg (o instalador tenta instalá-lo automaticamente).

## Instalação rápida (Ubuntu)

Clone o repositório e entre na pasta:

```bash
git clone https://github.com/GiovaniRodrigo/audio-engineer.git
cd audio-engineer
```

Dê permissão de execução aos scripts e instale as dependências:

```bash
chmod +x instalar.sh iniciar.sh
./instalar.sh
```

O instalador cria o ambiente virtual `.venv`, instala/atualiza o pip e instala as bibliotecas de `requirements.txt`. Se necessário, solicitará sua senha para instalar FFmpeg e os pacotes do sistema.

Para abrir o laboratório:

```bash
./iniciar.sh
```

O Streamlit exibirá um endereço local, normalmente http://localhost:8501.

## Atualizar uma cópia já clonada

```bash
git pull
chmod +x instalar.sh iniciar.sh
./instalar.sh
./iniciar.sh
```

Se as dependências não tiverem mudado, não é necessário executar novamente o instalador após cada `git pull`.

## Instalação manual

Caso prefira não usar os scripts:

```bash
sudo apt update
sudo apt install -y ffmpeg python3-venv python3-pip
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run audio_engineering_lab.py
```

## Estrutura principal

```text
.
├── audio_engineering_lab.py
├── requirements.txt
├── instalar.sh
├── iniciar.sh
└── README.md
```

## Observações

- A primeira execução pode levar mais tempo devido ao download de modelos de reconhecimento de fala, quando essa funcionalidade for utilizada.
- Preserve os arquivos de áudio originais ao realizar experimentos de processamento.
- Nenhum ambiente virtual é versionado no Git; a pasta `.venv` é criada localmente pelo instalador.

## Licença

Nenhuma licença foi definida neste repositório ainda. Consulte o proprietário antes de reutilizar ou redistribuir o código.
