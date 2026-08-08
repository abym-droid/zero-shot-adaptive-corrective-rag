#!/usr/bin/env bash
# setup_env.sh — build the local `adarag` thesis-system env on Apple silicon.
# Idempotent-ish: recreates the env if --fresh is passed, else installs into it.
#
#   bash environment/setup_env.sh          # create/update env + install core
#   bash environment/setup_env.sh --fresh  # remove and rebuild from scratch
#   bash environment/setup_env.sh --optional  # also try requirements-optional.txt
set -uo pipefail

ENV_NAME="adarag"
PY_VER="3.11"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRESH="no"; OPTIONAL="no"
for a in "$@"; do
  [ "$a" = "--fresh" ] && FRESH="yes"
  [ "$a" = "--optional" ] && OPTIONAL="yes"
done

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ "$FRESH" = "yes" ]; then
  echo ">> removing existing env $ENV_NAME"
  conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
fi

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo ">> creating conda env $ENV_NAME (python $PY_VER)"
  conda create -n "$ENV_NAME" "python=${PY_VER}" -y
fi

conda activate "$ENV_NAME"
echo ">> using python: $(which python)  ($(python --version))"

python -m pip install --upgrade pip wheel setuptools

echo ">> installing core requirements"
python -m pip install -r "${HERE}/requirements-core.txt"

if [ "$OPTIONAL" = "yes" ]; then
  echo ">> installing OPTIONAL requirements (failures tolerated)"
  while IFS= read -r line; do
    case "$line" in ''|\#*) continue;; esac
    pkg="${line%%#*}"; pkg="$(echo "$pkg" | xargs)"
    [ -z "$pkg" ] && continue
    echo ">>   pip install $pkg"
    python -m pip install "$pkg" || echo ">>   (skipped: $pkg failed to install)"
  done < "${HERE}/requirements-optional.txt"
fi

echo ">> registering Jupyter kernel"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python (adarag)" >/dev/null 2>&1 || true

echo ">> freezing lock file"
python -m pip freeze > "${HERE}/requirements-lock.txt"

echo ">> running env self-check"
python "${HERE}/check_env.py" || true

echo ">> done. Activate with:  conda activate ${ENV_NAME}"
