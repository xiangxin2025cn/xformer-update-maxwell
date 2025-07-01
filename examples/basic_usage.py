#!/usr/bin/env python3
"""
Basic Usage Example for xformers on RTX 5090 D + Blackwell Architecture

This script demonstrates basic usage of xformers memory_efficient_attention
on RTX 5090 D with Blackwell sm_120 architecture.

Usage:
    python3 basic_usage.py

Requirements:
    - RTX 5090 D GPU with Blackwell architecture
    - xformers compiled for sm_120
    - PyTorch 2.9.0+ with CUDA 12.8
"""

import torch
import time
import sys

def check_environment():
    """Check if the environment is properly configured."""
    print("🔍 Environment Check")
    print("=" * 40)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    # Check GPU
    gpu_name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    print(f"🚀 GPU: {gpu_name}")
    print(f"🏆 Compute Capability: sm_{capability[0]}{capability[1]}")
    
    # Check xformers
    try:
        import xformers
        print(f"⚡ xformers Version: {xformers.__version__}")
        return True
    except ImportError:
        print("❌ xformers is not available")
        return False

def basic_attention_example():
    """Demonstrate basic memory efficient attention usage."""
    print("\n📝 Basic Memory Efficient Attention Example")
    print("=" * 50)
    
    import xformers.ops as xops
    
    # Configuration
    batch_size = 2
    seq_len = 1024
    head_dim = 64
    device = torch.device("cuda")
    dtype = torch.float16
    
    print(f"Configuration: batch={batch_size}, seq_len={seq_len}, head_dim={head_dim}")
    
    # Create input tensors
    q = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    
    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    
    # Perform attention
    start_time = time.time()
    output = xops.memory_efficient_attention(q, k, v)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Output shape: {output.shape}")
    print(f"Computation time: {(end_time - start_time) * 1000:.2f} ms")
    print("✅ Basic attention example completed successfully!")
    
    return output

def attention_with_bias_example():
    """Demonstrate attention with bias."""
    print("\n🎯 Attention with Bias Example")
    print("=" * 40)
    
    import xformers.ops as xops
    
    # Configuration
    batch_size = 1
    seq_len = 512
    head_dim = 64
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Create input tensors
    q = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    
    # Create attention bias (causal mask)
    attn_bias = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    attn_bias = attn_bias.to(device=device, dtype=dtype)
    
    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"Attention bias shape: {attn_bias.shape}")
    
    # Perform attention with bias
    start_time = time.time()
    output = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Output shape: {output.shape}")
    print(f"Computation time: {(end_time - start_time) * 1000:.2f} ms")
    print("✅ Attention with bias example completed successfully!")
    
    return output

def multi_head_attention_example():
    """Demonstrate multi-head attention pattern."""
    print("\n🧠 Multi-Head Attention Example")
    print("=" * 40)
    
    import xformers.ops as xops
    
    # Configuration
    batch_size = 1
    seq_len = 2048
    num_heads = 8
    head_dim = 64
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Create input tensors (batch, seq_len, num_heads * head_dim)
    hidden_dim = num_heads * head_dim
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    
    # Linear projections (simplified)
    q = x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (batch, heads, seq, dim)
    k = x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Reshape for xformers (batch * heads, seq, dim)
    q = q.reshape(batch_size * num_heads, seq_len, head_dim)
    k = k.reshape(batch_size * num_heads, seq_len, head_dim)
    v = v.reshape(batch_size * num_heads, seq_len, head_dim)
    
    print(f"Multi-head configuration: {num_heads} heads, {head_dim} dim per head")
    print(f"Reshaped input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    
    # Perform multi-head attention
    start_time = time.time()
    output = xops.memory_efficient_attention(q, k, v)
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Reshape back to multi-head format
    output = output.reshape(batch_size, num_heads, seq_len, head_dim)
    output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
    
    print(f"Final output shape: {output.shape}")
    print(f"Computation time: {(end_time - start_time) * 1000:.2f} ms")
    print("✅ Multi-head attention example completed successfully!")
    
    return output

def performance_comparison():
    """Compare xformers vs PyTorch native attention."""
    print("\n⚡ Performance Comparison: xformers vs PyTorch")
    print("=" * 55)
    
    import xformers.ops as xops
    import torch.nn.functional as F
    
    # Configuration
    batch_size = 1
    seq_len = 2048
    head_dim = 64
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Create input tensors
    q = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(3):
        _ = xops.memory_efficient_attention(q, k, v)
        _ = F.scaled_dot_product_attention(q, k, v)
    
    torch.cuda.synchronize()
    
    # Benchmark xformers
    num_iterations = 10
    xformers_times = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        output_xformers = xops.memory_efficient_attention(q, k, v)
        torch.cuda.synchronize()
        end_time = time.time()
        xformers_times.append(end_time - start_time)
    
    # Benchmark PyTorch native
    pytorch_times = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        output_pytorch = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        end_time = time.time()
        pytorch_times.append(end_time - start_time)
    
    # Calculate statistics
    xformers_avg = sum(xformers_times) / len(xformers_times)
    pytorch_avg = sum(pytorch_times) / len(pytorch_times)
    speedup = pytorch_avg / xformers_avg
    
    print(f"Configuration: {batch_size}×{seq_len}×{head_dim}")
    print(f"xformers average time: {xformers_avg * 1000:.2f} ms")
    print(f"PyTorch average time:  {pytorch_avg * 1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x faster with xformers")
    
    # Verify correctness
    max_diff = torch.max(torch.abs(output_xformers - output_pytorch)).item()
    print(f"Maximum difference: {max_diff:.6f}")
    
    if max_diff < 1e-3:
        print("✅ Results are numerically equivalent")
    else:
        print("⚠️ Results have significant differences")
    
    print("✅ Performance comparison completed!")

def memory_usage_example():
    """Demonstrate memory usage monitoring."""
    print("\n💾 Memory Usage Example")
    print("=" * 30)
    
    import xformers.ops as xops
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Configuration
    batch_size = 1
    seq_len = 4096
    head_dim = 64
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Monitor initial memory
    initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    print(f"Initial GPU memory: {initial_memory:.1f} MB")
    
    # Create input tensors
    q = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    
    after_tensors = torch.cuda.memory_allocated() / 1024**2  # MB
    print(f"After creating tensors: {after_tensors:.1f} MB (+{after_tensors - initial_memory:.1f} MB)")
    
    # Perform attention
    output = xops.memory_efficient_attention(q, k, v)
    
    after_attention = torch.cuda.memory_allocated() / 1024**2  # MB
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"After attention: {after_attention:.1f} MB (+{after_attention - after_tensors:.1f} MB)")
    print(f"Peak memory usage: {peak_memory:.1f} MB")
    print(f"Total memory used: {peak_memory - initial_memory:.1f} MB")
    
    print("✅ Memory usage example completed!")

def main():
    """Main function to run all examples."""
    print("🎯 xformers Basic Usage Examples")
    print("🏆 RTX 5090 D + Blackwell Architecture")
    print("=" * 50)
    
    # Environment check
    if not check_environment():
        print("\n❌ Environment check failed. Please ensure:")
        print("   1. RTX 5090 D GPU is available")
        print("   2. CUDA 12.8 is properly installed")
        print("   3. PyTorch 2.9.0+ with CUDA 12.8 support")
        print("   4. xformers compiled for sm_120 architecture")
        sys.exit(1)
    
    try:
        # Run examples
        basic_attention_example()
        attention_with_bias_example()
        multi_head_attention_example()
        performance_comparison()
        memory_usage_example()
        
        print("\n🎉 All examples completed successfully!")
        print("🏆 xformers is working perfectly on RTX 5090 D + Blackwell!")
        print("🚀 You're ready to accelerate your AI workloads!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
