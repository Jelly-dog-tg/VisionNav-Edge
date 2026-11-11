> 这是原作者基础仿真安装/平台排错说明；VisionNav-Edge最终ACT系统请先阅读 [README.md](README.md)。历史命令或本机路径可能不适用于最终入口。

# Installation Guide

## Issue: PyBullet Installation Failures

If you encounter build errors or "no matching distribution" errors when installing PyBullet, try these solutions in order:

### Step 0: Diagnose the Issue

First, run the diagnostic script to understand your environment:

```bash
source venv/bin/activate
python scripts/diagnose_pybullet.py
```

This will show your Python version, architecture, and provide specific recommendations.

## Solution 1: Upgrade pip and Install (Most Common Fix)

```bash
source venv/bin/activate

# Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# Clear pip cache and install
pip install pybullet --no-cache-dir

# Then install other dependencies
pip install -r requirements.txt
```

## Solution 2: Install Without Build Isolation

If Solution 1 fails, try without build isolation:

```bash
pip install pybullet --no-build-isolation --no-cache-dir
```

## Solution 3: Use Conda (Recommended for Apple Silicon or Python 3.12+)

If you're on Apple Silicon (M1/M2/M3) or using Python 3.12+, conda is often more reliable:

```bash
# If you don't have conda, install Miniconda first from https://docs.conda.io/en/latest/miniconda.html

# Create new environment
conda create -n pybullet_env python=3.11
conda activate pybullet_env

# Install PyBullet via conda
conda install -c conda-forge pybullet

# Install other dependencies via pip
pip install -r requirements.txt
```

## Solution 4: Install Packages Individually

If the requirements file causes issues, install packages one by one:

```bash
pip install --upgrade pip setuptools wheel
pip install pybullet --no-cache-dir
pip install gymnasium
pip install "stable-baselines3[extra]"
pip install numpy matplotlib pyyaml tensorboard opencv-python
```

## Solution 5: Use Python 3.11 (If Using Python 3.12+)

PyBullet may not have wheels for Python 3.12+. Create a new venv with Python 3.11:

```bash
# Find Python 3.11 (might be python3.11 or installed via Homebrew)
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pybullet
pip install -r requirements.txt
```

## Verify Installation

After installation, verify everything works:

```bash
python -c "import pybullet, numpy, cv2, torch, stable_baselines3; print('setup ok')"
```

## Common Issues

**"No matching distribution found":**
- This usually means no pre-built wheel for your Python/architecture combo
- Try: `pip install --upgrade pip` then `pip install pybullet --no-cache-dir`
- Or use conda (Solution 3 above)

**macOS Build Errors (Compilation Failures):**
- PyBullet is trying to build from source
- Ensure Xcode Command Line Tools: `xcode-select --install`
- Try: `pip install pybullet --no-build-isolation`
- Or use conda which handles this better

**Apple Silicon (M1/M2/M3) Issues:**
- Some PyBullet versions may have issues on ARM
- Use conda: `conda install -c conda-forge pybullet`
- Or ensure you're using the ARM version of Python (not x86_64 via Rosetta)

**Python Version:**
- PyBullet works best with Python 3.8-3.11
- Python 3.12+ may not have pre-built wheels - use conda or Python 3.11

**Pip Cache Issues:**
- Clear cache: `pip cache purge`
- Install with: `pip install pybullet --no-cache-dir`

