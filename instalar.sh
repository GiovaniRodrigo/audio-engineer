#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "==> Preparando o Audio Engineering Lab"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "Erro: este instalador foi preparado para Ubuntu/Debian (apt-get)." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1 || ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Erro: é necessário Python 3.10 ou superior." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! dpkg -s python3-venv >/dev/null 2>&1; then
  echo "==> Instalando FFmpeg e suporte a ambientes virtuais"
  sudo apt-get update
  sudo apt-get install -y ffmpeg python3-venv python3-pip
fi

if [[ ! -f requirements.txt ]]; then
  echo "Erro: requirements.txt não encontrado na raiz do projeto." >&2
  exit 1
fi

echo "==> Criando ambiente virtual"
python3 -m venv .venv

echo "==> Instalando dependências Python"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

chmod +x iniciar.sh 2>/dev/null || true

echo
echo "Instalação concluída."
echo "Para iniciar: ./iniciar.sh"
