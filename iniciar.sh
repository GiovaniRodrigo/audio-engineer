#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/streamlit" ]]; then
  echo "Ambiente virtual ou Streamlit não encontrado."
  echo "Execute primeiro: ./instalar.sh"
  exit 1
fi

APP_FILE="audio_engineering_lab.py"
if [[ ! -f "$APP_FILE" ]]; then
  echo "Erro: $APP_FILE não foi encontrado na raiz do projeto." >&2
  exit 1
fi

echo "Iniciando Audio Engineering Lab..."
exec .venv/bin/streamlit run "$APP_FILE" "$@"
