#!/usr/bin/env python3
"""
Focused Convergence Test
========================

Quick but meaningful test with proper language modeling
to see if bespoke benefits emerge with reasonable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import time

def focused_convergence_test():
    print("🎯 FOCUSED CONVERGENCE TEST")
    print("=" * 50)
    
    device = torch.device("cpu")
    
    # Optimized for quick but meaningful results
    vocab_size = 200
    embed_dim = 64
    seq_len = 16
    batch_size = 8
    corpus_size = 10000
    
    # Focus on convergence behavior - moderate epochs
    num_epochs = 50
    eval_every = 10
    
    print(f"Quick config: {vocab_size} vocab, {num_epochs} epochs")
    
    # Generate strong frequency patterns
    ranks = np.arange(1, vocab_size + 1)
    zipf_probs = 1 / ranks**1.4  # Very strong Zipfian
    zipf_probs = zipf_probs / zipf_probs.sum()
    
    corpus = torch.multinomial(
        torch.tensor(zipf_probs, dtype=torch.float), 
        corpus_size, 
        replacement=True
    )
    
    # Clear frequency analysis
    token_counts = Counter(corpus.tolist())
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Strong categorization - clear separation
    high_freq_tokens = set(token for token, _ in sorted_tokens[:20])    # Top 20 tokens
    low_freq_tokens = set(token for token, _ in sorted_tokens[-50:])   # Bottom 50 tokens
    mid_freq_tokens = set(range(vocab_size)) - high_freq_tokens - low_freq_tokens
    
    total_tokens = sum(count for _, count in sorted_tokens)
    high_coverage = sum(token_counts[t] for t in high_freq_tokens) / total_tokens
    
    print(f"Clear categories: {len(high_freq_tokens)} high ({high_coverage:.1%}), {len(mid_freq_tokens)} mid, {len(low_freq_tokens)} low")
    
    class FocusedBespokeEmbedding(nn.Module):
        def __init__(self, vocab_size, embed_dim, high_tokens, mid_tokens, low_tokens):
            super().__init__()
            
            self.categories = {
                'high': high_tokens,
                'mid': mid_tokens, 
                'low': low_tokens
            }
            
            # Aggressive dimension differences for clear signal
            self.high_dim = embed_dim * 2      # 2x for frequent
            self.mid_dim = embed_dim           # Standard
            self.low_dim = embed_dim // 4      # 1/4 for rare
            
            # Direct embeddings (no complex indexing)
            self.embeddings = nn.ModuleDict({
                'high': nn.Embedding(vocab_size, self.high_dim),
                'mid': nn.Embedding(vocab_size, self.mid_dim),
                'low': nn.Embedding(vocab_size, self.low_dim)
            })
            
            # Simple projections
            self.projections = nn.ModuleDict({
                'high': nn.Linear(self.high_dim, embed_dim),
                'mid': nn.Identity(),  # No projection needed
                'low': nn.Linear(self.low_dim, embed_dim)
            })
            
        def forward(self, token_ids):
            batch_size, seq_len = token_ids.shape
            output = torch.zeros(batch_size, seq_len, embed_dim, device=token_ids.device)
            
            # Vectorized approach for efficiency
            for i in range(batch_size):
                for j in range(seq_len):
                    token = token_ids[i, j].item()
                    
                    if token in self.categories['high']:
                        embed_val = self.embeddings['high'](token_ids[i:i+1, j:j+1])
                        output[i, j] = self.projections['high'](embed_val).squeeze()
                    elif token in self.categories['low']:
                        embed_val = self.embeddings['low'](token_ids[i:i+1, j:j+1])  
                        output[i, j] = self.projections['low'](embed_val).squeeze()
                    else:  # mid frequency
                        output[i, j] = self.embeddings['mid'](token_ids[i:i+1, j:j+1]).squeeze()
            
            return output
    
    class SimpleLanguageModel(nn.Module):
        def __init__(self, embedding, vocab_size, embed_dim):
            super().__init__()
            self.embedding = embedding
            self.output_head = nn.Linear(embed_dim, vocab_size)
            
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = x.mean(dim=1)  # Simple pooling
            return self.output_head(x)
    
    # Create models
    standard_embed = nn.Embedding(vocab_size, embed_dim)
    bespoke_embed = FocusedBespokeEmbedding(vocab_size, embed_dim, high_freq_tokens, mid_freq_tokens, low_freq_tokens)
    
    standard_model = SimpleLanguageModel(standard_embed, vocab_size, embed_dim).to(device)
    bespoke_model = SimpleLanguageModel(bespoke_embed, vocab_size, embed_dim).to(device)
    
    # Parameter check
    std_params = sum(p.numel() for p in standard_model.parameters())
    bes_params = sum(p.numel() for p in bespoke_model.parameters())
    print(f"Parameters: Standard {std_params:,}, Bespoke {bes_params:,} ({bes_params/std_params:.2f}x)")
    
    # Training data
    train_size = int(0.8 * corpus_size)
    train_data = corpus[:train_size]
    val_data = corpus[train_size:]
    
    def create_next_token_batches(data, batch_size, seq_len):
        """Simple next-token prediction batches"""
        batches = []
        max_start = len(data) - seq_len - 1
        
        for _ in range(50):  # 50 batches per epoch
            batch_x = []
            batch_y = []
            
            for _ in range(batch_size):
                start = torch.randint(0, max_start, (1,)).item()
                x = data[start:start+seq_len]
                y = data[start+seq_len]  # Next token
                
                batch_x.append(x)
                batch_y.append(y)
            
            batches.append((
                torch.stack(batch_x).to(device),
                torch.stack(batch_y).to(device)
            ))
        
        return batches
    
    def evaluate_next_token(model, val_data):
        """Evaluate next-token prediction"""
        model.eval()
        total_loss = 0
        count = 0
        
        val_batches = create_next_token_batches(val_data, batch_size, seq_len)[:10]
        
        with torch.no_grad():
            for x, y in val_batches:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                total_loss += loss.item()
                count += 1
        
        model.train()
        return total_loss / count if count > 0 else float('inf')
    
    def train_focused(model, train_data, val_data, name):
        """Focused training with convergence tracking"""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        train_losses = []
        val_losses = []
        
        print(f"\n🏃 Training {name} ({num_epochs} epochs)...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            model.train()
            train_batches = create_next_token_batches(train_data, batch_size, seq_len)
            
            epoch_loss = 0
            for x, y in train_batches:
                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_batches)
            train_losses.append(avg_train_loss)
            
            if epoch % eval_every == 0 or epoch == num_epochs - 1:
                val_loss = evaluate_next_token(model, val_data)
                val_losses.append(val_loss)
                
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1:2d}: train={avg_train_loss:.4f}, val={val_loss:.4f} ({elapsed:.1f}s)")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_loss': val_losses[-1] if val_losses else float('inf')
        }
    
    # Train both models
    standard_results = train_focused(standard_model, train_data, val_data, "Standard")
    bespoke_results = train_focused(bespoke_model, train_data, val_data, "Bespoke")
    
    # Analysis
    std_final = standard_results['final_val_loss']
    bes_final = bespoke_results['final_val_loss']
    
    if std_final > 0 and bes_final > 0:
        loss_ratio = bes_final / std_final
        param_ratio = bes_params / std_params
        
        print(f"\n📊 FOCUSED CONVERGENCE RESULTS:")
        print(f"Standard final validation loss: {std_final:.4f}")
        print(f"Bespoke final validation loss:  {bes_final:.4f}")
        print(f"Loss ratio (bespoke/standard): {loss_ratio:.4f}")
        print(f"Parameter ratio: {param_ratio:.2f}x")
        
        # Check for improvement patterns
        std_improved = len(standard_results['val_losses']) >= 3 and \
                      standard_results['val_losses'][-1] < standard_results['val_losses'][0]
        bes_improved = len(bespoke_results['val_losses']) >= 3 and \
                      bespoke_results['val_losses'][-1] < bespoke_results['val_losses'][0]
        
        print(f"Standard improved: {std_improved}")
        print(f"Bespoke improved: {bes_improved}")
        
        # Conservative validation
        clear_benefit = loss_ratio < 0.85  # At least 15% better
        reasonable_overhead = param_ratio < 3.0  # Less than 3x parameters
        both_learning = std_improved and bes_improved
        
        validated = clear_benefit and reasonable_overhead and both_learning
        
        print(f"\n🎯 FOCUSED VALIDATION:")
        print(f"Clear benefit: {clear_benefit} ({loss_ratio:.4f} < 0.85)")
        print(f"Reasonable overhead: {reasonable_overhead} ({param_ratio:.2f}x < 3.0)")
        print(f"Both learning: {both_learning}")
        print(f"Theory validated: {'✅ YES' if validated else '❌ NO'}")
        
        if validated:
            improvement = (1 - loss_ratio) * 100
            print(f"\n🎉 FOCUSED SUCCESS!")
            print(f"   ✅ {improvement:.1f}% improvement with {num_epochs} epochs")
            print(f"   ✅ Clear convergence pattern observed")
            print(f"   ✅ Reasonable parameter overhead: {param_ratio:.2f}x")
            print(f"   🔬 Bespoke embeddings show promise with proper training!")
        else:
            print(f"\n🤔 FOCUSED RESULTS:")
            if not clear_benefit:
                improvement = (1 - loss_ratio) * 100
                print(f"   - Modest improvement: {improvement:.1f}% (needed ≥15%)")
            if not reasonable_overhead:
                print(f"   - High parameter overhead: {param_ratio:.2f}x")
            if not both_learning:
                print(f"   - Learning issues detected")
            print(f"   📊 May need longer training or architecture changes")
        
        return validated, {
            'loss_ratio': loss_ratio,
            'param_ratio': param_ratio,
            'improvement_percent': (1 - loss_ratio) * 100,
            'validated': validated
        }
    else:
        print("❌ Training failed - invalid loss values")
        return False, {}

if __name__ == "__main__":
    print("🎯 Running focused convergence test...")
    success, results = focused_convergence_test()
    
    print("\n" + "="*50)
    if success:
        print("✅ FOCUSED TEST: Benefits detected with proper training!")
    else:
        print("⚠️  FOCUSED TEST: Benefits unclear or need more training")
    print("="*50)
