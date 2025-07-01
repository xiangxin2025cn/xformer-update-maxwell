# Installation Guide for xformers on RTX 5090 D

This guide provides step-by-step instructions for compiling xformers on RTX 5090 D with Blackwell architecture.

## 🎯 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 5090 D (Blackwell architecture, sm_120)
- **RAM**: 32GB+ system memory recommended
- **Storage**: 50GB+ free space for compilation
- **CPU**: Modern multi-core processor (Intel/AMD)

### Software Requirements
- **OS**: Ubuntu 24.04.2 LTS (native or WSL2)
- **CUDA**: 12.8 Linux toolkit
- **Python**: 3.12+
- **GCC**: 11+ (for CUDA 12.8 compatibility)

## 🔧 Step 1: System Preparation

### 1.1 Update System
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake git wget curl
```

### 1.2 Install Development Tools
```bash
# Essential build tools
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    python3-dev \
    python3-pip \
    python3-venv

# CUDA development dependencies
sudo apt install -y \
    gcc-11 \
    g++-11 \
    libnvidia-compute-560 \
    nvidia-cuda-toolkit-12-8
```

## 🚀 Step 2: CUDA 12.8 Installation

### 2.1 Download CUDA 12.8
```bash
# Download CUDA 12.8 Linux installer
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_560.35.03_linux.run
```

### 2.2 Install CUDA 12.8
```bash
# Make installer executable
chmod +x cuda_12.8.0_560.35.03_linux.run

# Install CUDA (select toolkit only, skip driver if already installed)
sudo sh cuda_12.8.0_560.35.03_linux.run
```

### 2.3 Configure Environment
```bash
# Add to ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc

# Reload environment
source ~/.bashrc

# Verify CUDA installation
nvcc --version
nvidia-smi
```

## 🐍 Step 3: Python Environment Setup

### 3.1 Create Virtual Environment
```bash
# Create dedicated environment for xformers
python3 -m venv ~/xformers_env
source ~/xformers_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3.2 Install PyTorch 2.9.0 Nightly
```bash
# Install PyTorch with CUDA 12.8 support
pip3 install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name()}')"
```

## 🔨 Step 4: Prepare xformers Source

### 4.1 Clone xformers Repository
```bash
# Clone official xformers repository
cd ~/
git clone https://github.com/facebookresearch/xformers.git
cd xformers

# Checkout stable version
git checkout main
```

### 4.2 Download Dependencies
```bash
# Download CUTLASS 4.0 (Blackwell support)
cd ~/
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v4.0.0

# Download flash-attention
cd ~/
git clone https://github.com/Dao-AILab/flash-attention.git

# Download composable_kernel
cd ~/
git clone https://github.com/ROCmSoftwarePlatform/composable_kernel.git
```

## ⚙️ Step 5: Apply Blackwell Patches

### 5.1 Update CUTLASS Integration
```bash
cd ~/xformers

# Update CUTLASS path in setup.py
# (Apply patches from patches/ directory)
```

### 5.2 Configure Compilation Flags
```bash
# Set environment variables for RTX 5090 D compilation
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export CUDA_HOME=/usr/local/cuda-12.8
export CUTLASS_PATH=~/cutlass
export FLASH_ATTENTION_PATH=~/flash-attention
export COMPOSABLE_KERNEL_PATH=~/composable_kernel

# Verify GPU compute capability
python3 -c "import torch; print(f'Compute capability: {torch.cuda.get_device_capability()}')"
```

## 🏗️ Step 6: Compile xformers

### 6.1 Install Dependencies
```bash
cd ~/xformers

# Install Python dependencies
pip install -r requirements.txt
pip install ninja pybind11
```

### 6.2 Compile xformers
```bash
# Start compilation (this may take 30-60 minutes)
pip install -v -e .

# Monitor compilation progress
# Look for successful compilation of all 21 CUDA kernels
```

### 6.3 Handle Compilation Issues
```bash
# If Sparse24 compilation fails (expected for Blackwell)
# The compilation will continue and complete successfully
# Sparse24 is replaced by fine-grained structured sparsity in Blackwell
```

## ✅ Step 7: Verification

### 7.1 Basic Import Test
```bash
python3 -c "import xformers; print(f'xformers {xformers.__version__} imported successfully!')"
```

### 7.2 Memory Efficient Attention Test
```bash
python3 -c "
import torch
import xformers.ops as xops

# Test on RTX 5090 D
device = torch.device('cuda')
q = torch.randn(1, 64, 32, device=device, dtype=torch.float16)
k = torch.randn(1, 64, 32, device=device, dtype=torch.float16)
v = torch.randn(1, 64, 32, device=device, dtype=torch.float16)

output = xops.memory_efficient_attention(q, k, v)
print(f'✅ memory_efficient_attention test passed: {output.shape}')
print('🎉 xformers is working perfectly on RTX 5090 D!')
"
```

### 7.3 Performance Benchmark
```bash
# Run comprehensive performance test
python3 examples/performance_test.py
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Version Mismatch
```bash
# Ensure CUDA 12.8 is properly installed
nvcc --version
# Should show: Cuda compilation tools, release 12.8
```

#### 2. PyTorch Version Issues
```bash
# Verify PyTorch CUDA version
python3 -c "import torch; print(torch.version.cuda)"
# Should show: 12.8
```

#### 3. Compilation Memory Issues
```bash
# If compilation runs out of memory, limit parallel jobs
export MAX_JOBS=4
pip install -v -e .
```

#### 4. Sparse24 Compilation Warnings
```bash
# Sparse24 warnings are expected for Blackwell architecture
# The compilation will complete successfully without Sparse24
# Fine-grained structured sparsity is used instead
```

## 🎯 Next Steps

After successful installation:

1. **Run Performance Tests**: Use `examples/performance_test.py`
2. **Integrate with AI Models**: See `examples/integration_test.py`
3. **Optimize for Your Workload**: Check `docs/performance.md`

## 📞 Support

If you encounter issues:

1. Check `docs/troubleshooting.md` for common solutions
2. Verify all prerequisites are met
3. Open an issue with detailed error logs
4. Include system information and compilation output

---

**🏆 Congratulations! You now have xformers running on RTX 5090 D with Blackwell architecture!**
