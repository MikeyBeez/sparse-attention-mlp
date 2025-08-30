#!/usr/bin/env python3
"""
Quick Bespoke Token Theory Validation
====================================

Fast test to validate the core hypothesis that different token types
can use different embedding dimensions while maintaining quality.

This is a minimal proof-of-concept before running the full experiment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import time

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[info] using device: {device}")

torch.manual_seed(42)

def quick_test():
    """Run a minimal test of the bespoke embedding concept"""
    
    # Small test parameters
    vocab_size = 100
    seq_len = 32
    batch_size = 8
    embed_dim = 64
    
    # Generate test data with Zipfian distribution (realistic)
    ranks = np.arange(1, vocab_size + 1)
    zipf_probs = 1 / ranks
    zipf_probs = zipf_probs / zipf_probs.sum()
    
    # Create test corpus
    corpus_size = 1000
    tokens = torch.multinomial(
        torch.tensor(zipf_probs, dtype=torch.float), 
        corpus_size, 
        replacement=True
    )
    
    # Analyze token frequencies
    token_counts = Counter(tokens.tolist())
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Categorize tokens by frequency
    high_freq_tokens = set(token for token, _ in sorted_tokens[:20])  # Top 20%
    low_freq_tokens = set(token for token, _ in sorted_tokens[-20:])  # Bottom 20%
    
    print(f"Token distribution: {len(high_freq_tokens)} high-freq, {len(low_freq_tokens)} low-freq")
    
    # Standard embedding (baseline)
    standard_embed = nn.Embedding(vocab_size, embed_dim).to(device)
    
    # Bespoke embedding system
    class QuickBespokeEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            # Different dimensions for different token types
            self.high_freq_embed = nn.Embedding(vocab_size, embed_dim + 16)  # Bigger for common tokens
            self.low_freq_embed = nn.Embedding(vocab_size, embed_dim - 16)   # Smaller for rare tokens
            self.mid_freq_embed = nn.Embedding(vocab_size, embed_dim)        # Standard for others
            
            # Projections to standard size
            self.high_proj = nn.Linear(embed_dim + 16, embed_dim)
            self.low_proj = nn.Linear(embed_dim - 16, embed_dim)
            
        def forward(self, token_ids):
            batch_size, seq_len = token_ids.shape
            output = torch.zeros(batch_size, seq_len, embed_dim, device=token_ids.device)
            
            # Process each token type
            for i in range(batch_size):
                for j in range(seq_len):
                    token_id = token_ids[i, j].item()
                    
                    if token_id in high_freq_tokens:
                        embed_val = self.high_freq_embed(token_ids[i:i+1, j:j+1])
                        output[i, j] = self.high_proj(embed_val).squeeze()
                    elif token_id in low_freq_tokens:
                        embed_val = self.low_freq_embed(token_ids[i:i+1, j:j+1])  
                        output[i, j] = self.low_proj(embed_val).squeeze()
                    else:
                        output[i, j] = self.mid_freq_embed(token_ids[i:i+1, j:j+1]).squeeze()
                        
            return output
    
    bespoke_embed = QuickBespokeEmbed().to(device)
    
    # Simple test model
    class QuickLM(nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embedding = embedding
            self.classifier = nn.Linear(embed_dim, vocab_size)
            
        def forward(self, x):
            x = self.embedding(x)
            x = x.mean(dim=1)  # Simple pooling
            return self.classifier(x)
    
    # Create models
    standard_model = QuickLM(standard_embed).to(device)
    bespoke_model = QuickLM(bespoke_embed).to(device)
    
    # Count parameters
    standard_params = sum(p.numel() for p in standard_model.parameters())
    bespoke_params = sum(p.numel() for p in bespoke_model.parameters())
    
    print(f"Parameters: Standard {standard_params:,}, Bespoke {bespoke_params:,}")
    print(f"Parameter ratio: {bespoke_params/standard_params:.3f}")
    
    # Quick training test
    def train_quick(model, steps=100):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        losses = []
        
        for step in range(steps):
            # Random batch
            start_idx = torch.randint(0, corpus_size - seq_len, (batch_size,))
            batch = torch.stack([tokens[i:i+seq_len] for i in start_idx]).to(device)
            targets = torch.randint(0, vocab_size, (batch_size,)).to(device)
            
            # Forward pass
            logits = model(batch)
            loss = F.cross_entropy(logits, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
        return losses
    
    # Train both models
    print("\nTraining standard model...")
    start_time = time.time()
    standard_losses = train_quick(standard_model)
    standard_time = time.time() - start_time
    
    print("Training bespoke model...")
    start_time = time.time()  
    bespoke_losses = train_quick(bespoke_model)
    bespoke_time = time.time() - start_time
    
    # Results
    print(f"\nResults:")
    print(f"Standard final loss: {standard_losses[-1]:.4f} (time: {standard_time:.2f}s)")
    print(f"Bespoke final loss: {bespoke_losses[-1]:.4f} (time: {bespoke_time:.2f}s)")
    print(f"Loss ratio: {bespoke_losses[-1]/standard_losses[-1]:.3f} (lower is better)")
    
    # Memory efficiency 
    memory_ratio = bespoke_params / standard_params
    performance_ratio = bespoke_losses[-1] / standard_losses[-1]
    efficiency_score = (1 / memory_ratio) / performance_ratio
    
    print(f"\nEfficiency Analysis:")
    print(f"Memory ratio: {memory_ratio:.3f} ({'saves' if memory_ratio < 1 else 'uses more'} memory)")
    print(f"Performance ratio: {performance_ratio:.3f} ({'better' if performance_ratio < 1 else 'worse'} performance)")
    print(f"Efficiency score: {efficiency_score:.3f} (higher is better)")
    
    # Theory validation
    theory_valid = (
        abs(performance_ratio - 1.0) < 0.2 and  # Performance within 20%
        memory_ratio <= 1.1  # Memory usage not significantly worse
    )
    
    print(f"\n🎯 Theory Validation: {'✅ PASS' if theory_valid else '❌ FAIL'}")
    
    if theory_valid:
        print("✨ The theory shows promise! Different token types can use different dimensions.")
    else:
        print("⚠️ Theory needs refinement. Consider adjusting dimension allocation strategy.")
        
    return {
        'standard_params': standard_params,
        'bespoke_params': bespoke_params,
        'memory_ratio': memory_ratio,
        'performance_ratio': performance_ratio,
        'efficiency_score': efficiency_score,
        'theory_valid': theory_valid
    }

if __name__ == "__main__":
    print("🧪 Quick Bespoke Token Dimension Test")
    print("="*50)
    results = quick_test()
    print("="*50)
