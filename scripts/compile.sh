#!/bin/bash
# xformers Compilation Script for RTX 5090 D + Blackwell Architecture
# 
# This script automates the compilation of xformers for RTX 5090 D with
# Blackwell sm_120 architecture support.
#
# Usage: ./compile.sh [options]
# Options:
#   --clean     Clean previous build artifacts
#   --verbose   Enable verbose compilation output
#   --jobs N    Set number of parallel compilation jobs (default: auto)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
XFORMERS_DIR="$HOME/xformers"
CUTLASS_DIR="$HOME/cutlass"
FLASH_ATTENTION_DIR="$HOME/flash-attention"
COMPOSABLE_KERNEL_DIR="$HOME/composable_kernel"

# Default options
CLEAN_BUILD=false
VERBOSE=false
MAX_JOBS=$(nproc)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --clean     Clean previous build artifacts"
            echo "  --verbose   Enable verbose compilation output"
            echo "  --jobs N    Set number of parallel compilation jobs"
            echo "  -h, --help  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Check if running in correct environment
check_environment() {
    log_step "Checking compilation environment..."
    
    # Check if we're in a virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        log_warning "Not in a virtual environment. Consider activating one."
    else
        log_info "Virtual environment: $VIRTUAL_ENV"
    fi
    
    # Check CUDA installation
    if ! command -v nvcc &> /dev/null; then
        log_error "CUDA compiler (nvcc) not found. Please install CUDA 12.8."
        exit 1
    fi
    
    local cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    log_info "CUDA version: $cuda_version"
    
    if [[ "$cuda_version" != "12.8" ]]; then
        log_warning "CUDA version is not 12.8. Compilation may fail."
    fi
    
    # Check GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Please install NVIDIA drivers."
        exit 1
    fi
    
    local gpu_info=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits | head -1)
    log_info "GPU: $gpu_info"
    
    # Check Python and PyTorch
    if ! python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        log_error "PyTorch not found or not working. Please install PyTorch 2.9.0+ with CUDA 12.8."
        exit 1
    fi
    
    local pytorch_cuda=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    log_info "PyTorch CUDA version: $pytorch_cuda"
    
    if [[ "$pytorch_cuda" != "12.8" ]]; then
        log_warning "PyTorch CUDA version is not 12.8. Compilation may fail."
    fi
    
    log_success "Environment check completed"
}

# Setup compilation environment
setup_environment() {
    log_step "Setting up compilation environment..."
    
    # Set CUDA architecture for RTX 5090 D (Blackwell sm_120)
    export TORCH_CUDA_ARCH_LIST="12.0"
    export FORCE_CUDA=1
    export CUDA_HOME=/usr/local/cuda-12.8
    export MAX_JOBS=$MAX_JOBS
    
    # Set paths for dependencies
    export CUTLASS_PATH="$CUTLASS_DIR"
    export FLASH_ATTENTION_PATH="$FLASH_ATTENTION_DIR"
    export COMPOSABLE_KERNEL_PATH="$COMPOSABLE_KERNEL_DIR"
    
    # Compiler settings for Blackwell architecture
    export NVCC_FLAGS="-gencode arch=compute_120,code=sm_120"
    
    log_info "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
    log_info "FORCE_CUDA: $FORCE_CUDA"
    log_info "CUDA_HOME: $CUDA_HOME"
    log_info "MAX_JOBS: $MAX_JOBS"
    
    log_success "Environment setup completed"
}

# Download dependencies if needed
download_dependencies() {
    log_step "Checking and downloading dependencies..."
    
    # Check and download xformers
    if [[ ! -d "$XFORMERS_DIR" ]]; then
        log_info "Downloading xformers..."
        git clone https://github.com/facebookresearch/xformers.git "$XFORMERS_DIR"
    else
        log_info "xformers already exists at $XFORMERS_DIR"
    fi
    
    # Check and download CUTLASS 4.0
    if [[ ! -d "$CUTLASS_DIR" ]]; then
        log_info "Downloading CUTLASS 4.0..."
        git clone https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
        cd "$CUTLASS_DIR"
        git checkout v4.0.0
        cd -
    else
        log_info "CUTLASS already exists at $CUTLASS_DIR"
    fi
    
    # Check and download flash-attention
    if [[ ! -d "$FLASH_ATTENTION_DIR" ]]; then
        log_info "Downloading flash-attention..."
        git clone https://github.com/Dao-AILab/flash-attention.git "$FLASH_ATTENTION_DIR"
    else
        log_info "flash-attention already exists at $FLASH_ATTENTION_DIR"
    fi
    
    # Check and download composable_kernel
    if [[ ! -d "$COMPOSABLE_KERNEL_DIR" ]]; then
        log_info "Downloading composable_kernel..."
        git clone https://github.com/ROCmSoftwarePlatform/composable_kernel.git "$COMPOSABLE_KERNEL_DIR"
    else
        log_info "composable_kernel already exists at $COMPOSABLE_KERNEL_DIR"
    fi
    
    log_success "Dependencies check completed"
}

# Apply Blackwell-specific patches
apply_patches() {
    log_step "Applying Blackwell architecture patches..."
    
    cd "$XFORMERS_DIR"
    
    # Apply patches if they exist
    local patches_dir="$PROJECT_ROOT/patches"
    
    if [[ -f "$patches_dir/blackwell_sm120.patch" ]]; then
        log_info "Applying Blackwell sm_120 patch..."
        patch -p1 < "$patches_dir/blackwell_sm120.patch" || log_warning "Patch may already be applied"
    fi
    
    if [[ -f "$patches_dir/cutlass_4.0.patch" ]]; then
        log_info "Applying CUTLASS 4.0 integration patch..."
        patch -p1 < "$patches_dir/cutlass_4.0.patch" || log_warning "Patch may already be applied"
    fi
    
    if [[ -f "$patches_dir/sparse24_fix.patch" ]]; then
        log_info "Applying Sparse24 compatibility fix..."
        patch -p1 < "$patches_dir/sparse24_fix.patch" || log_warning "Patch may already be applied"
    fi
    
    log_success "Patches applied"
}

# Clean previous build artifacts
clean_build() {
    if [[ "$CLEAN_BUILD" == true ]]; then
        log_step "Cleaning previous build artifacts..."
        
        cd "$XFORMERS_DIR"
        
        # Remove build directories
        rm -rf build/
        rm -rf dist/
        rm -rf *.egg-info/
        rm -rf __pycache__/
        
        # Remove compiled extensions
        find . -name "*.so" -delete
        find . -name "*.pyc" -delete
        find . -name "*.pyo" -delete
        
        log_success "Build artifacts cleaned"
    fi
}

# Compile xformers
compile_xformers() {
    log_step "Starting xformers compilation for RTX 5090 D..."
    
    cd "$XFORMERS_DIR"
    
    # Install Python dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    pip install ninja pybind11
    
    # Start compilation
    log_info "Starting compilation (this may take 30-60 minutes)..."
    log_info "Compiling for Blackwell sm_120 architecture..."
    
    local compile_cmd="pip install -v -e ."
    
    if [[ "$VERBOSE" == true ]]; then
        log_info "Running: $compile_cmd"
        $compile_cmd
    else
        log_info "Running compilation (use --verbose for detailed output)..."
        $compile_cmd > compilation.log 2>&1 || {
            log_error "Compilation failed. Check compilation.log for details."
            tail -50 compilation.log
            exit 1
        }
    fi
    
    log_success "xformers compilation completed!"
}

# Verify installation
verify_installation() {
    log_step "Verifying xformers installation..."
    
    # Basic import test
    if python3 -c "import xformers; print(f'xformers {xformers.__version__} imported successfully')" 2>/dev/null; then
        log_success "xformers import test passed"
    else
        log_error "xformers import test failed"
        exit 1
    fi
    
    # Memory efficient attention test
    if python3 -c "
import torch
import xformers.ops as xops
device = torch.device('cuda')
q = torch.randn(1, 64, 32, device=device, dtype=torch.float16)
k = torch.randn(1, 64, 32, device=device, dtype=torch.float16)
v = torch.randn(1, 64, 32, device=device, dtype=torch.float16)
output = xops.memory_efficient_attention(q, k, v)
print(f'memory_efficient_attention test passed: {output.shape}')
" 2>/dev/null; then
        log_success "memory_efficient_attention test passed"
    else
        log_error "memory_efficient_attention test failed"
        exit 1
    fi
    
    log_success "Installation verification completed"
}

# Main execution
main() {
    echo -e "${CYAN}"
    echo "================================================================"
    echo "  xformers Compilation for RTX 5090 D + Blackwell Architecture"
    echo "  World's First Successful Implementation"
    echo "================================================================"
    echo -e "${NC}"
    
    check_environment
    setup_environment
    download_dependencies
    apply_patches
    clean_build
    compile_xformers
    verify_installation
    
    echo -e "${GREEN}"
    echo "================================================================"
    echo "  🎉 COMPILATION SUCCESSFUL! 🎉"
    echo "  🏆 xformers is now ready for RTX 5090 D + Blackwell!"
    echo "  🚀 You have the world's most advanced AI computing environment!"
    echo "================================================================"
    echo -e "${NC}"
    
    log_info "Next steps:"
    log_info "1. Run performance tests: python3 examples/performance_test.py"
    log_info "2. Integrate with your AI models"
    log_info "3. Enjoy unprecedented AI performance!"
}

# Execute main function
main "$@"
