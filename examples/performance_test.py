#!/usr/bin/env python3
"""
Performance Benchmark for xformers on RTX 5090 D + Blackwell Architecture

This script provides comprehensive performance testing for xformers
memory_efficient_attention on RTX 5090 D with Blackwell sm_120 architecture.

Usage:
    python3 performance_test.py

Requirements:
    - RTX 5090 D GPU with Blackwell architecture
    - xformers compiled for sm_120
    - PyTorch 2.9.0+ with CUDA 12.8
"""

import torch
import time
import gc
import sys
from typing import List, Tuple, Dict
import argparse

def check_environment() -> bool:
    """Check if the environment is properly configured for RTX 5090 D."""
    print("🔍 Environment Check")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    # Check GPU information
    gpu_name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🚀 GPU: {gpu_name}")
    print(f"🏆 Compute Capability: sm_{capability[0]}{capability[1]}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    print(f"🔧 PyTorch Version: {torch.__version__}")
    
    # Check for RTX 5090 D and Blackwell architecture
    is_rtx5090d = "RTX 5090" in gpu_name
    is_blackwell = capability[0] >= 12
    
    if is_rtx5090d and is_blackwell:
        print("🏆 RTX 5090 D with Blackwell architecture detected - Perfect!")
    elif is_blackwell:
        print("✅ Blackwell architecture detected - xformers optimizations available")
    elif capability[0] >= 8:
        print("✅ Tensor Core support available")
    else:
        print("⚠️ Older GPU architecture - limited performance expected")
    
    # Check xformers availability
    try:
        import xformers
        import xformers.ops as xops
        print(f"⚡ xformers Version: {xformers.__version__}")
        print("✅ xformers is available")
        
        # Quick xformers test
        device = torch.device("cuda")
        q = torch.randn(1, 32, 16, device=device, dtype=torch.float16)
        k = torch.randn(1, 32, 16, device=device, dtype=torch.float16)
        v = torch.randn(1, 32, 16, device=device, dtype=torch.float16)
        
        output = xops.memory_efficient_attention(q, k, v)
        print(f"✅ memory_efficient_attention test passed: {output.shape}")
        
        del q, k, v, output
        torch.cuda.empty_cache()
        
        return True
        
    except ImportError:
        print("❌ xformers is not available")
        return False
    except Exception as e:
        print(f"❌ xformers test failed: {e}")
        return False

def run_benchmark_suite() -> Dict[str, Dict]:
    """Run comprehensive benchmark suite for different configurations."""
    import xformers.ops as xops
    
    print("\n📊 Performance Benchmark Suite")
    print("=" * 50)
    
    # Test configurations: (name, batch_size, seq_len, dim)
    test_configs = [
        ("Small", 2, 1024, 64),
        ("Medium", 2, 2048, 64),
        ("Large", 1, 4096, 64),
        ("XLarge", 1, 8192, 64),
        ("XXLarge", 1, 16384, 64),
        ("CogVLM2", 1, 2048, 128),
        ("LLaMA-7B", 1, 4096, 128),
        ("LLaMA-13B", 1, 4096, 160),
        ("GPT-4", 1, 8192, 256),
        ("Extreme", 1, 32768, 64),
    ]
    
    device = torch.device("cuda")
    dtype = torch.float16
    results = {}
    
    for config_name, batch_size, seq_len, dim in test_configs:
        print(f"\n🧪 Testing: {config_name}")
        print(f"   Configuration: batch={batch_size}, seq_len={seq_len}, dim={dim}")
        
        try:
            # Create tensors
            q = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
            k = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
            v = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
            
            # Warmup runs
            for _ in range(3):
                _ = xops.memory_efficient_attention(q, k, v)
            
            torch.cuda.synchronize()
            
            # Benchmark runs
            num_iterations = 10
            times = []
            
            for i in range(num_iterations):
                start_time = time.time()
                output = xops.memory_efficient_attention(q, k, v)
                torch.cuda.synchronize()
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            
            # Calculate throughput
            total_tokens = batch_size * seq_len
            tokens_per_sec = total_tokens / avg_time
            
            # Memory usage
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            
            # Store results
            results[config_name] = {
                "config": (batch_size, seq_len, dim),
                "avg_time_ms": avg_time * 1000,
                "min_time_ms": min_time * 1000,
                "max_time_ms": max_time * 1000,
                "std_time_ms": std_time * 1000,
                "tokens_per_sec": tokens_per_sec,
                "memory_gb": memory_used,
                "status": "SUCCESS"
            }
            
            print(f"   ⏱️  Average Time: {avg_time * 1000:.2f} ms")
            print(f"   🚀 Throughput: {tokens_per_sec:.0f} tokens/sec")
            print(f"   💾 Memory Used: {memory_used:.2f} GB")
            print(f"   ✅ Status: SUCCESS")
            
            # Cleanup
            del q, k, v, output
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                results[config_name] = {"status": "OUT_OF_MEMORY"}
                print(f"   ❌ Status: OUT OF MEMORY")
            else:
                results[config_name] = {"status": f"ERROR: {str(e)}"}
                print(f"   ❌ Status: ERROR - {str(e)}")
            
            torch.cuda.empty_cache()
        
        except Exception as e:
            results[config_name] = {"status": f"UNEXPECTED_ERROR: {str(e)}"}
            print(f"   ❌ Status: UNEXPECTED ERROR - {str(e)}")
            torch.cuda.empty_cache()
    
    return results

def print_summary_table(results: Dict[str, Dict]):
    """Print a formatted summary table of benchmark results."""
    print("\n📋 Performance Summary Table")
    print("=" * 80)
    
    # Header
    print(f"{'Configuration':<12} {'Batch×Seq×Dim':<15} {'Time (ms)':<10} {'Throughput':<15} {'Memory':<10} {'Status':<10}")
    print("-" * 80)
    
    # Results
    for config_name, result in results.items():
        if result["status"] == "SUCCESS":
            batch, seq, dim = result["config"]
            config_str = f"{batch}×{seq}×{dim}"
            time_str = f"{result['avg_time_ms']:.1f}"
            throughput_str = f"{result['tokens_per_sec']:.0f} tok/s"
            memory_str = f"{result['memory_gb']:.1f} GB"
            status_str = "✅ OK"
        else:
            config_str = "N/A"
            time_str = "N/A"
            throughput_str = "N/A"
            memory_str = "N/A"
            status_str = "❌ FAIL"
        
        print(f"{config_name:<12} {config_str:<15} {time_str:<10} {throughput_str:<15} {memory_str:<10} {status_str:<10}")

def print_detailed_analysis(results: Dict[str, Dict]):
    """Print detailed performance analysis."""
    print("\n🔬 Detailed Performance Analysis")
    print("=" * 50)
    
    successful_results = {k: v for k, v in results.items() if v["status"] == "SUCCESS"}
    
    if not successful_results:
        print("❌ No successful benchmark results to analyze")
        return
    
    # Find best performing configurations
    best_throughput = max(successful_results.values(), key=lambda x: x["tokens_per_sec"])
    best_efficiency = min(successful_results.values(), key=lambda x: x["avg_time_ms"])
    
    print(f"🏆 Highest Throughput:")
    for name, result in successful_results.items():
        if result == best_throughput:
            print(f"   {name}: {result['tokens_per_sec']:.0f} tokens/sec")
            break
    
    print(f"⚡ Fastest Execution:")
    for name, result in successful_results.items():
        if result == best_efficiency:
            print(f"   {name}: {result['avg_time_ms']:.2f} ms")
            break
    
    # Calculate total performance metrics
    total_throughput = sum(r["tokens_per_sec"] for r in successful_results.values())
    avg_throughput = total_throughput / len(successful_results)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Successful Tests: {len(successful_results)}/{len(results)}")
    print(f"   Average Throughput: {avg_throughput:.0f} tokens/sec")
    print(f"   Peak Throughput: {best_throughput['tokens_per_sec']:.0f} tokens/sec")

def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="xformers Performance Benchmark for RTX 5090 D")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (fewer iterations)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed timing statistics")
    args = parser.parse_args()
    
    print("🎯 xformers Performance Benchmark")
    print("🏆 RTX 5090 D + Blackwell Architecture + xformers")
    print("🚀 World's Most Advanced AI Computing Environment")
    print("=" * 70)
    
    # Environment check
    if not check_environment():
        print("\n❌ Environment check failed. Please ensure:")
        print("   1. RTX 5090 D GPU is available")
        print("   2. CUDA 12.8 is properly installed")
        print("   3. PyTorch 2.9.0+ with CUDA 12.8 support")
        print("   4. xformers compiled for sm_120 architecture")
        sys.exit(1)
    
    # Run benchmarks
    try:
        results = run_benchmark_suite()
        
        # Print results
        print_summary_table(results)
        print_detailed_analysis(results)
        
        print("\n🎉 Benchmark Complete!")
        print("🏆 Congratulations on having the world's most advanced AI computing environment!")
        print("🚀 RTX 5090 D + Blackwell + xformers = Ultimate Performance!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
