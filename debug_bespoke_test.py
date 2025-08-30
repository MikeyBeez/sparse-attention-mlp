#!/usr/bin/env python3
"""
Debug Bespoke Token Test - Simplified version to identify issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import time

def debug_test():
    print("🔍 Debug: Starting bespoke token test")
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ MPS device available")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ CUDA device available")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU device")
    
    print(f"Device: {device}")
    
    # Test basic functionality
    try:
        vocab_size = 50
        embed_dim = 32
        batch_size = 4
        seq_len = 16
        
        print(f"Testing with vocab_size={vocab_size}, embed_dim={embed_dim}")
        
        # Create simple embeddings
        standard_embed = nn.Embedding(vocab_size, embed_dim).to(device)
        print("✅ Standard embedding created")
        
        # Test data
        test_tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        print(f"✅ Test tokens created: shape {test_tokens.shape}")
        
        # Test forward pass
        standard_output = standard_embed(test_tokens)
        print(f"✅ Standard forward pass: shape {standard_output.shape}")
        
        # Simple bespoke embedding
        class SimpleBespoke(nn.Module):
            def __init__(self, vocab_size, embed_dim):
                super().__init__()
                # Split vocabulary into frequent (0-24) and rare (25-49)
                self.freq_embed = nn.Embedding(vocab_size, embed_dim + 8)  # Bigger for frequent
                self.rare_embed = nn.Embedding(vocab_size, embed_dim - 8)  # Smaller for rare
                self.freq_proj = nn.Linear(embed_dim + 8, embed_dim)
                self.rare_proj = nn.Linear(embed_dim - 8, embed_dim)
                self.threshold = vocab_size // 2
                
            def forward(self, token_ids):
                # Separate frequent and rare tokens
                freq_mask = token_ids < self.threshold
                rare_mask = token_ids >= self.threshold
                
                output = torch.zeros(token_ids.shape[0], token_ids.shape[1], embed_dim, 
                                   device=token_ids.device)
                
                # Process frequent tokens
                if freq_mask.any():
                    freq_tokens = token_ids[freq_mask]
                    freq_embeds = self.freq_embed(freq_tokens)
                    freq_projected = self.freq_proj(freq_embeds)
                    output[freq_mask] = freq_projected
                
                # Process rare tokens
                if rare_mask.any():
                    rare_tokens = token_ids[rare_mask]
                    rare_embeds = self.rare_embed(rare_tokens)
                    rare_projected = self.rare_proj(rare_embeds)
                    output[rare_mask] = rare_projected
                
                return output
        
        bespoke_embed = SimpleBespoke(vocab_size, embed_dim).to(device)
        print("✅ Bespoke embedding created")
        
        # Test bespoke forward pass
        bespoke_output = bespoke_embed(test_tokens)
        print(f"✅ Bespoke forward pass: shape {bespoke_output.shape}")
        
        # Count parameters
        standard_params = sum(p.numel() for p in standard_embed.parameters())
        bespoke_params = sum(p.numel() for p in bespoke_embed.parameters())
        
        print(f"Parameters - Standard: {standard_params:,}, Bespoke: {bespoke_params:,}")
        param_ratio = bespoke_params / standard_params
        print(f"Parameter ratio: {param_ratio:.3f}")
        
        # Quick quality test with a simple task
        class SimpleClassifier(nn.Module):
            def __init__(self, embedding, vocab_size, embed_dim):
                super().__init__()
                self.embedding = embedding
                self.classifier = nn.Linear(embed_dim, vocab_size)
                
            def forward(self, x):
                x = self.embedding(x)
                x = x.mean(dim=1)  # Global average pooling
                return self.classifier(x)
        
        # Create models
        standard_model = SimpleClassifier(standard_embed, vocab_size, embed_dim).to(device)
        bespoke_model = SimpleClassifier(bespoke_embed, vocab_size, embed_dim).to(device)
        
        print("✅ Models created")
        
        # Quick training test
        optimizer_std = torch.optim.Adam(standard_model.parameters(), lr=0.01)
        optimizer_bes = torch.optim.Adam(bespoke_model.parameters(), lr=0.01)
        
        print("🏋️ Starting quick training test...")
        
        # Training loop
        std_losses = []
        bes_losses = []
        
        for step in range(20):  # Very short training
            # Generate batch
            batch_tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
            # Simple task: predict the most frequent token in sequence
            targets = torch.mode(batch_tokens, dim=1)[0]
            
            # Train standard model
            optimizer_std.zero_grad()
            std_logits = standard_model(batch_tokens)
            std_loss = F.cross_entropy(std_logits, targets)
            std_loss.backward()
            optimizer_std.step()
            std_losses.append(std_loss.item())
            
            # Train bespoke model
            optimizer_bes.zero_grad()
            bes_logits = bespoke_model(batch_tokens)
            bes_loss = F.cross_entropy(bes_logits, targets)
            bes_loss.backward()
            optimizer_bes.step()
            bes_losses.append(bes_loss.item())
            
            if step % 5 == 0:
                print(f"  Step {step}: Standard {std_loss.item():.4f}, Bespoke {bes_loss.item():.4f}")
        
        # Results
        final_std_loss = std_losses[-1]
        final_bes_loss = bes_losses[-1]
        performance_ratio = final_bes_loss / final_std_loss
        
        print(f"\n📊 Results:")
        print(f"Standard final loss: {final_std_loss:.4f}")
        print(f"Bespoke final loss: {final_bes_loss:.4f}")
        print(f"Performance ratio: {performance_ratio:.3f} ({'better' if performance_ratio < 1 else 'worse'})")
        print(f"Parameter efficiency: {param_ratio:.3f}")
        
        # Theory validation
        memory_efficient = param_ratio <= 1.2  # Allow slight increase due to projection layers
        quality_maintained = performance_ratio <= 1.3  # Allow 30% degradation for proof-of-concept
        
        theory_valid = memory_efficient and quality_maintained
        
        print(f"\n🎯 Theory Validation:")
        print(f"  Memory efficient: {memory_efficient} (ratio: {param_ratio:.3f})")
        print(f"  Quality maintained: {quality_maintained} (ratio: {performance_ratio:.3f})")
        print(f"  Overall: {'✅ PASS' if theory_valid else '❌ FAIL'}")
        
        if theory_valid:
            print("🚀 Theory shows promise! Bespoke dimensions can work.")
            print("   Next step: Optimize dimension allocation and test on larger scales.")
        else:
            print("⚠️ Theory needs refinement.")
            if not memory_efficient:
                print("   Issue: Parameter count not efficiently reduced")
            if not quality_maintained:
                print("   Issue: Quality degradation too high")
        
        return theory_valid, {
            'param_ratio': param_ratio,
            'performance_ratio': performance_ratio,
            'memory_efficient': memory_efficient,
            'quality_maintained': quality_maintained
        }
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

if __name__ == "__main__":
    print("🧪 Debug Bespoke Token Dimension Test")
    print("=" * 50)
    success, results = debug_test()
    print("=" * 50)
    if success:
        print("✅ Test completed successfully")
    else:
        print("❌ Test failed - check output above")
