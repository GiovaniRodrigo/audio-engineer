#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
if [[ ! -x ".venv/bin/streamlit" ]]; then
  echo "Ambiente virtual ou Streamlit não encontrado. Execute ./instalar.sh" >&2
  exit 1
fi
exec .venv/bin/streamlit run smart_audio_calibration_app.py "$@"
