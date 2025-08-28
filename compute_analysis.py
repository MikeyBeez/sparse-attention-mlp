# compute_analysis.py
# Analyze computational complexity of sparse attention vs full attention across model sizes
# Demonstrates the crossover point where sparse attention becomes beneficial

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

def count_attention_flops(batch_size: int, seq_len: int, n_heads: int, head_dim: int) -> int:
    """Count FLOPs for full self-attention"""
    # Q @ K^T: (B, H, T, d) @ (B, H, d, T) = (B, H, T, T)
    qk_flops = batch_size * n_heads * seq_len * seq_len * head_dim
    
    # Softmax: roughly 3 * elements (exp, sum, divide)
    softmax_flops = 3 * batch_size * n_heads * seq_len * seq_len
    
    # Attention @ V: (B, H, T, T) @ (B, H, T, d) = (B, H, T, d)  
    attn_v_flops = batch_size * n_heads * seq_len * seq_len * head_dim
    
    # QKV projections: 3 * (B, T, d_model) @ (d_model, d_model)
    d_model = n_heads * head_dim
    qkv_proj_flops = 3 * batch_size * seq_len * d_model * d_model
    
    # Output projection: (B, T, d_model) @ (d_model, d_model)
    out_proj_flops = batch_size * seq_len * d_model * d_model
    
    return qk_flops + softmax_flops + attn_v_flops + qkv_proj_flops + out_proj_flops

def count_sparse_attention_flops(batch_size: int, seq_len: int, n_heads: int, head_dim: int, 
                                 top_k: int, selector_hidden: int, approx_hidden: int) -> int:
    """Count FLOPs for sparse attention with MLP approximation"""
    d_model = n_heads * head_dim
    
    # Still need QKV projections (same as full attention)
    qkv_proj_flops = 3 * batch_size * seq_len * d_model * d_model
    
    # Output projection (same as full attention) 
    out_proj_flops = batch_size * seq_len * d_model * d_model
    
    # For each head that uses sparse approximation:
    sparse_head_flops = 0
    
    # Key Selector MLP per query: (B*T, head_dim) -> (B*T, seq_len)
    # Layer 1: head_dim -> selector_hidden  
    selector_l1_flops = batch_size * seq_len * head_dim * selector_hidden
    # Layer 2: selector_hidden -> seq_len
    selector_l2_flops = batch_size * seq_len * selector_hidden * seq_len
    
    # Top-K selection: roughly seq_len * log(top_k) per query
    topk_flops = batch_size * seq_len * seq_len * math.log2(top_k)
    
    # Value gathering: essentially free (just indexing)
    gather_flops = 0
    
    # MLP Approximator per query: (B*T, top_k * head_dim) -> (B*T, head_dim)
    # Layer 1: (top_k * head_dim) -> approx_hidden
    approx_l1_flops = batch_size * seq_len * (top_k * head_dim) * approx_hidden
    # Layer 2: approx_hidden -> head_dim  
    approx_l2_flops = batch_size * seq_len * approx_hidden * head_dim
    
    sparse_head_flops = (selector_l1_flops + selector_l2_flops + topk_flops + 
                        gather_flops + approx_l1_flops + approx_l2_flops)
    
    # For remaining heads, use reduced attention (only top-k, not full seq_len)
    remaining_heads = n_heads - 1  # assume 1 head uses MLP approximation
    if remaining_heads > 0:
        # Reduced Q@K^T for remaining heads: only compute top_k instead of seq_len
        reduced_qk_flops = batch_size * remaining_heads * seq_len * top_k * head_dim
        reduced_softmax_flops = 3 * batch_size * remaining_heads * seq_len * top_k  
        reduced_attn_v_flops = batch_size * remaining_heads * seq_len * top_k * head_dim
        
        remaining_head_flops = reduced_qk_flops + reduced_softmax_flops + reduced_attn_v_flops
    else:
        remaining_head_flops = 0
    
    return qkv_proj_flops + out_proj_flops + sparse_head_flops + remaining_head_flops

@dataclass 
class ModelConfig:
    seq_len: int
    n_heads: int  
    head_dim: int
    batch_size: int = 1
    
    @property
    def d_model(self):
        return self.n_heads * self.head_dim

def analyze_compute_scaling():
    """Analyze computation across different model sizes"""
    
    # Model configurations from tiny to large
    configs = [
        ModelConfig(seq_len=64, n_heads=2, head_dim=64),      # Tiny (128d)
        ModelConfig(seq_len=128, n_heads=4, head_dim=64),     # Small (256d) 
        ModelConfig(seq_len=256, n_heads=8, head_dim=64),     # Medium (512d)
        ModelConfig(seq_len=512, n_heads=12, head_dim=64),    # Base (768d)
        ModelConfig(seq_len=1024, n_heads=16, head_dim=64),   # Large (1024d)
        ModelConfig(seq_len=2048, n_heads=20, head_dim=64),   # XL (1280d)
        ModelConfig(seq_len=4096, n_heads=32, head_dim=64),   # XXL (2048d)
    ]
    
    top_k = 8
    selector_hidden = 128
    approx_hidden = 128
    
    results = []
    
    print("Model Size Analysis:")
    print("=" * 80)
    print(f"{'Config':<15} {'Full Attn':<12} {'Sparse Attn':<12} {'Speedup':<10} {'Efficient?'}")
    print("-" * 80)
    
    for cfg in configs:
        full_flops = count_attention_flops(cfg.batch_size, cfg.seq_len, cfg.n_heads, cfg.head_dim)
        sparse_flops = count_sparse_attention_flops(cfg.batch_size, cfg.seq_len, cfg.n_heads, 
                                                   cfg.head_dim, top_k, selector_hidden, approx_hidden)
        
        speedup = full_flops / sparse_flops
        efficient = speedup > 1.0
        
        config_name = f"{cfg.d_model}d/{cfg.seq_len}L"
        
        print(f"{config_name:<15} {full_flops/1e6:>8.1f}M {sparse_flops/1e6:>8.1f}M {speedup:>8.2f}x {'✓' if efficient else '✗'}")
        
        results.append({
            'config': config_name,
            'd_model': cfg.d_model,
            'seq_len': cfg.seq_len, 
            'full_flops': full_flops,
            'sparse_flops': sparse_flops,
            'speedup': speedup,
            'efficient': efficient
        })
    
    return results

def plot_scaling_analysis(results):
    """Plot the computational scaling results"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    d_models = [r['d_model'] for r in results]
    seq_lens = [r['seq_len'] for r in results] 
    speedups = [r['speedup'] for r in results]
    full_flops = [r['full_flops']/1e9 for r in results]  # Convert to GFLOPS
    sparse_flops = [r['sparse_flops']/1e9 for r in results]
    
    # Plot 1: FLOPS comparison
    x = np.arange(len(results))
    ax1.bar(x - 0.2, full_flops, 0.4, label='Full Attention', alpha=0.7)
    ax1.bar(x + 0.2, sparse_flops, 0.4, label='Sparse Attention', alpha=0.7)
    ax1.set_xlabel('Model Size')
    ax1.set_ylabel('Computation (GFLOPS)')
    ax1.set_title('Computational Cost Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([r['config'] for r in results], rotation=45)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Plot 2: Speedup factor
    ax2.plot(x, speedups, 'o-', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    ax2.set_xlabel('Model Size') 
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Sparse vs Full Attention Speedup')
    ax2.set_xticks(x)
    ax2.set_xticklabels([r['config'] for r in results], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/bard/Code/nanoGPT_sparse_attention/compute_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def find_crossover_point():
    """Find the exact model size where sparse attention becomes beneficial"""
    
    print("\nCrossover Point Analysis:")
    print("=" * 50)
    
    # Test a range of model sizes to find crossover
    seq_lens = range(32, 1024, 32)
    top_k = 8
    selector_hidden = 128
    approx_hidden = 128
    
    crossover_found = False
    
    for seq_len in seq_lens:
        # Use reasonable head count scaling
        n_heads = max(2, seq_len // 64)
        head_dim = 64
        
        cfg = ModelConfig(seq_len=seq_len, n_heads=n_heads, head_dim=head_dim)
        
        full_flops = count_attention_flops(cfg.batch_size, cfg.seq_len, cfg.n_heads, cfg.head_dim)
        sparse_flops = count_sparse_attention_flops(cfg.batch_size, cfg.seq_len, cfg.n_heads, 
                                                   cfg.head_dim, top_k, selector_hidden, approx_hidden)
        
        speedup = full_flops / sparse_flops
        
        if speedup > 1.0 and not crossover_found:
            print(f"Crossover point: {cfg.d_model}d model, {seq_len} sequence length")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Full attention: {full_flops/1e6:.1f}M FLOPs")
            print(f"Sparse attention: {sparse_flops/1e6:.1f}M FLOPs")
            crossover_found = True
            break
    
    if not crossover_found:
        print("Crossover point not found in tested range")

if __name__ == "__main__":
    print("Sparse Attention Computational Analysis")
    print("======================================\n")
    
    # Run scaling analysis
    results = analyze_compute_scaling()
    
    # Plot results
    try:
        plot_scaling_analysis(results)
        print(f"\nPlot saved to: /Users/bard/Code/nanoGPT_sparse_attention/compute_scaling.png")
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # Find crossover point
    find_crossover_point()
    
    print("\nKey Insights:")
    print("- Small models: MLP approximation uses MORE computation")
    print("- Large models: Significant computational savings") 
    print("- Crossover occurs around medium-sized models (512-768d)")
    print("- Benefits increase quadratically with sequence length")
