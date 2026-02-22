#!/usr/bin/env bash
#
# install.sh - Set up the Cabinet Door Opening Robot project
#
# Works on macOS and WSL/Linux. Creates a virtual environment,
# installs robosuite + robocasa, and downloads kitchen assets.
#
# Usage:
#   ./install.sh
#
set -euo pipefail

# ─── Helpers ────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# ─── Detect platform ───────────────────────────────────────────────

OS="$(uname -s)"
case "$OS" in
    Darwin) PLATFORM="macos" ;;
    Linux)  PLATFORM="linux" ;;
    *)      error "Unsupported OS: $OS. This script supports macOS and Linux/WSL." ;;
esac

info "Detected platform: $PLATFORM"

# Check for WSL
if [ "$PLATFORM" = "linux" ] && grep -qi microsoft /proc/version 2>/dev/null; then
    info "Running inside WSL"
fi

# ─── Check Python ──────────────────────────────────────────────────

PYTHON=""
for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python 3.10+ not found. Please install Python 3.10, 3.11, or 3.12."
fi

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$("$PYTHON" -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MINOR" -lt 10 ]; then
    error "Python >= 3.10 required, found $PY_VERSION"
fi

info "Using $PYTHON ($PY_VERSION)"

# ─── Check system dependencies (Linux/WSL) ─────────────────────────

if [ "$PLATFORM" = "linux" ]; then
    sudo apt-get update -qq

    apt_has_candidate() {
        apt-cache policy "$1" 2>/dev/null | awk -F': ' '/Candidate:/ {print $2}' | grep -vq '(none)'
    }

    PKGS=(git cmake build-essential python3-dev linux-libc-dev libosmesa6-dev libglew-dev)

    if apt_has_candidate libgl1-mesa-glx; then
        PKGS+=(libgl1-mesa-glx)
    else
        PKGS+=(libgl1 libglx-mesa0)
    fi

    MISSING_PKGS=()
    for pkg in "${PKGS[@]}"; do
        if ! dpkg -s "$pkg" &>/dev/null; then
            MISSING_PKGS+=("$pkg")
        fi
    done

    if [ "${#MISSING_PKGS[@]}" -gt 0 ]; then
        warn "Missing system packages: ${MISSING_PKGS[*]}"
        info "Installing with apt (may need sudo)..."
        sudo apt-get install -y -qq "${MISSING_PKGS[@]}"
    fi
fi

# ─── Create virtual environment ────────────────────────────────────

VENV_DIR="$REPO_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
    info "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
info "Activated virtual environment"

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_PY -m pip"

# Make sure pip exists in the venv (some distros create venvs without pip)
"$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
$VENV_PIP install --upgrade pip --quiet

# Avoid conda poisoning native builds (evdev, mujoco deps, etc.)
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export PATH="/usr/bin:$PATH"

# Upgrade pip
$VENV_PIP install --upgrade pip --quiet

# ─── Clone and install robosuite ────────────────────────────────────

if [ -d "$REPO_DIR/robosuite" ]; then
    info "robosuite directory already exists, skipping clone"
else
    info "Cloning robosuite..."
    git clone https://github.com/ARISE-Initiative/robosuite.git "$REPO_DIR/robosuite"
fi

info "Installing robosuite (this will likely pull NumPy 1.x)..."
$VENV_PIP install -e "$REPO_DIR/robosuite" --quiet

# ─── Clone and install robocasa ─────────────────────────────────────

if [ -d "$REPO_DIR/robocasa" ]; then
    info "robocasa directory already exists, skipping clone"
else
    info "Cloning robocasa..."
    git clone https://github.com/robocasa/robocasa.git "$REPO_DIR/robocasa"
fi

info "Installing robocasa..."
$VENV_PIP install -e "$REPO_DIR/robocasa" --quiet

# ─── The "Intentional Override" ─────────────────────────────────────
# We MUST force NumPy 2.2.5 last. Pip will print a scary red ERROR 
# complaining about the mink dependency conflict. This is expected 
# and necessary to make RoboCasa work.

info "Resolving NumPy version conflict (EXPECT A PIP ERROR HERE)..."
$VENV_PIP install "numpy==2.2.5" "opencv-python-headless>=4.10" --quiet

# ─── Install additional Python dependencies ─────────────────────────

info "Installing additional dependencies..."
$VENV_PIP install --quiet \
    torch torchvision \
    matplotlib \
    pyarrow \
    imageio[ffmpeg] \
    jupyter \
    ipykernel

# ─── Download kitchen assets ────────────────────────────────────────

info "Downloading RoboCasa kitchen assets (~5 GB)..."
info "This may take a while on a slow connection."
"$VENV_PY" -m robocasa.scripts.download_kitchen_assets

# ─── Set up macros ──────────────────────────────────────────────────

info "Setting up RoboCasa macros..."
"$VENV_PY" -m robocasa.scripts.setup_macros

# ─── Platform-specific notes ────────────────────────────────────────

echo ""
echo "========================================"
echo "  Installation complete!"
echo "========================================"
echo ""
echo "To get started:"
echo ""
echo "  source .venv/bin/activate"
echo "  cd cabinet_door_project"
echo "  python 00_verify_installation.py"
echo ""

if [ "$PLATFORM" = "macos" ]; then
    echo "NOTE (macOS): Scripts that open a rendering window"
    echo "(03_teleop, 05_playback) require mjpython:"
    echo ""
    echo "  mjpython 03_teleop_collect_demos.py"
    echo ""
fi

if [ "$PLATFORM" = "linux" ]; then
    echo "NOTE (headless Linux/WSL): If you have no display, set:"
    echo ""
    echo "  export MUJOCO_GL=egl    # for GPU rendering"
    echo "  export MUJOCO_GL=osmesa # for CPU rendering"
    echo ""
fi
