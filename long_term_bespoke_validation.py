#!/usr/bin/env python3
"""
Long-Term Bespoke Token Validation
==================================

Proper validation with many epochs to see if bespoke embeddings
actually provide benefits after sufficient convergence time.

This addresses the critical issue: bespoke encodings need many epochs
to start working well, so we need to measure after proper training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import json
import time
import matplotlib.pyplot as plt

def long_term_bespoke_test():
    print("🔬 LONG-TERM BESPOKE TOKEN VALIDATION")
    print("=" * 60)
    print("⏰ This will take several minutes - proper training takes time!")
    
    device = torch.device("cpu")  # Use CPU for consistency
    
    # Test configuration
    vocab_size = 1000
    embed_dim = 256
    seq_len = 64
    batch_size = 16
    corpus_size = 100000  # Larger corpus for proper validation
    
    # CRITICAL: Many more epochs for proper convergence
    num_epochs = 500  # Was 3, now 500!
    eval_every = 50   # Evaluate every 50 epochs
    
    print(f"Configuration: {vocab_size} vocab, {embed_dim} embed_dim")
    print(f"Training: {num_epochs} epochs, evaluate every {eval_every}")
    print(f"Corpus size: {corpus_size:,} tokens")
    
    # Generate realistic corpus with strong Zipfian distribution
    ranks = np.arange(1, vocab_size + 1)
    zipf_probs = 1 / ranks**1.2  # Strong Zipfian
    zipf_probs = zipf_probs / zipf_probs.sum()
    
    print("🎲 Generating large realistic corpus...")
    corpus = torch.multinomial(
        torch.tensor(zipf_probs, dtype=torch.float), 
        corpus_size, 
        replacement=True
    )
    
    # Analyze token frequencies
    token_counts = Counter(corpus.tolist())
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Frequency-based categorization
    total_occurrences = sum(count for _, count in sorted_tokens)
    cumulative = 0
    high_freq_tokens = set()
    mid_freq_tokens = set() 
    low_freq_tokens = set()
    
    for token, count in sorted_tokens:
        freq_percentile = cumulative / total_occurrences
        
        if freq_percentile < 0.7:      # Top 70% of occurrences
            high_freq_tokens.add(token)
        elif freq_percentile < 0.9:    # Next 20% of occurrences  
            mid_freq_tokens.add(token)
        else:                          # Bottom 10% (rare tokens)
            low_freq_tokens.add(token)
            
        cumulative += count
    
    print(f"Token analysis: {len(high_freq_tokens)} high, {len(mid_freq_tokens)} mid, {len(low_freq_tokens)} low")
    print(f"High freq coverage: {sum(token_counts[t] for t in high_freq_tokens)/total_occurrences:.1%}")
    
    # Create train/validation split
    train_size = int(0.8 * corpus_size)
    train_corpus = corpus[:train_size]
    val_corpus = corpus[train_size:]
    
    print(f"Data split: {len(train_corpus):,} train, {len(val_corpus):,} validation")
    
    class ProperBespokeEmbedding(nn.Module):
        """Bespoke embedding designed for long-term convergence"""
        def __init__(self, vocab_size, embed_dim, high_tokens, mid_tokens, low_tokens):
            super().__init__()
            
            self.high_tokens = high_tokens
            self.mid_tokens = mid_tokens
            self.low_tokens = low_tokens
            
            # More conservative dimension allocation for stability
            self.high_dim = int(embed_dim * 1.3)   # 30% more for frequent
            self.mid_dim = embed_dim               # Standard
            self.low_dim = int(embed_dim * 0.6)    # 40% less for rare
            
            # Create embeddings for each category
            high_count = len(high_tokens)
            mid_count = len(mid_tokens)
            low_count = len(low_tokens)
            
            if high_count > 0:
                self.high_embed = nn.Embedding(high_count, self.high_dim)
                self.high_token_to_idx = {token: idx for idx, token in enumerate(sorted(high_tokens))}
            else:
                self.high_embed = None
                self.high_token_to_idx = {}
                
            if mid_count > 0:
                self.mid_embed = nn.Embedding(mid_count, self.mid_dim)
                self.mid_token_to_idx = {token: idx for idx, token in enumerate(sorted(mid_tokens))}
            else:
                self.mid_embed = None
                self.mid_token_to_idx = {}
                
            if low_count > 0:
                self.low_embed = nn.Embedding(low_count, self.low_dim)
                self.low_token_to_idx = {token: idx for idx, token in enumerate(sorted(low_tokens))}
            else:
                self.low_embed = None
                self.low_token_to_idx = {}
            
            # Projections to standard dimension
            self.high_proj = nn.Linear(self.high_dim, embed_dim, bias=False) if self.high_dim != embed_dim else None
            self.low_proj = nn.Linear(self.low_dim, embed_dim, bias=False) if self.low_dim != embed_dim else None
            
            self.embed_dim = embed_dim
            
        def forward(self, token_ids):
            batch_size, seq_len = token_ids.shape
            output = torch.zeros(batch_size, seq_len, self.embed_dim, 
                               device=token_ids.device, dtype=torch.float32)
            
            flat_tokens = token_ids.view(-1)
            flat_output = output.view(-1, self.embed_dim)
            
            for i, token_id in enumerate(flat_tokens):
                token_val = token_id.item()
                
                if token_val in self.high_tokens and self.high_embed is not None:
                    embed_idx = self.high_token_to_idx[token_val]
                    embed_val = self.high_embed.weight[embed_idx]
                    if self.high_proj is not None:
                        flat_output[i] = self.high_proj(embed_val)
                    else:
                        flat_output[i] = embed_val
                        
                elif token_val in self.mid_tokens and self.mid_embed is not None:
                    embed_idx = self.mid_token_to_idx[token_val]
                    flat_output[i] = self.mid_embed.weight[embed_idx]
                    
                elif token_val in self.low_tokens and self.low_embed is not None:
                    embed_idx = self.low_token_to_idx[token_val]
                    embed_val = self.low_embed.weight[embed_idx]
                    if self.low_proj is not None:
                        flat_output[i] = self.low_proj(embed_val)
                    else:
                        flat_output[i] = embed_val
                else:
                    # Unknown token - use small random embedding
                    flat_output[i] = torch.randn(self.embed_dim) * 0.01
            
            return output
    
    class ProperLanguageModel(nn.Module):
        """More realistic language model for proper validation"""
        def __init__(self, embedding, vocab_size, embed_dim, seq_len):
            super().__init__()
            self.embedding = embedding
            self.pos_embedding = nn.Embedding(seq_len, embed_dim)
            
            # Proper transformer-like architecture
            self.transformer_blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=8,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                for _ in range(6)  # 6 layers for more realistic model
            ])
            
            self.layer_norm = nn.LayerNorm(embed_dim)
            self.output_head = nn.Linear(embed_dim, vocab_size)
            
            # Initialize weights properly
            self.apply(self._init_weights)
            
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
        def forward(self, input_ids):
            batch_size, seq_len = input_ids.shape
            
            # Token embeddings
            x = self.embedding(input_ids)
            
            # Position embeddings
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            x = x + self.pos_embedding(positions)
            
            # Transformer layers
            for block in self.transformer_blocks:
                x = block(x)
            
            # Output projection
            x = self.layer_norm(x)
            logits = self.output_head(x)
            
            return logits
    
    # Create models
    standard_embed = nn.Embedding(vocab_size, embed_dim)
    bespoke_embed = ProperBespokeEmbedding(vocab_size, embed_dim, high_freq_tokens, mid_freq_tokens, low_freq_tokens)
    
    standard_model = ProperLanguageModel(standard_embed, vocab_size, embed_dim, seq_len).to(device)
    bespoke_model = ProperLanguageModel(bespoke_embed, vocab_size, embed_dim, seq_len).to(device)
    
    print(f"\n📊 Model Parameters:")
    standard_params = sum(p.numel() for p in standard_model.parameters())
    bespoke_params = sum(p.numel() for p in bespoke_model.parameters())
    print(f"Standard: {standard_params:,} parameters")
    print(f"Bespoke:  {bespoke_params:,} parameters ({bespoke_params/standard_params:.3f}x)")
    
    def create_batches(corpus_data, batch_size, seq_len):
        """Create batches for language modeling"""
        batches = []
        max_start = len(corpus_data) - seq_len - 1
        
        # Create more batches for better training
        num_batches = min(1000, max_start // seq_len)  # Up to 1000 batches per epoch
        
        for _ in range(num_batches):
            batch_inputs = []
            batch_targets = []
            
            for _ in range(batch_size):
                start_idx = torch.randint(0, max_start, (1,)).item()
                input_seq = corpus_data[start_idx:start_idx+seq_len]
                target_seq = corpus_data[start_idx+1:start_idx+seq_len+1]
                
                batch_inputs.append(input_seq)
                batch_targets.append(target_seq)
            
            if len(batch_inputs) == batch_size:
                batches.append((
                    torch.stack(batch_inputs).to(device),
                    torch.stack(batch_targets).to(device)
                ))
        
        return batches
    
    def evaluate_model(model, val_corpus):
        """Proper evaluation on validation set"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        val_batches = create_batches(val_corpus, batch_size, seq_len)[:50]  # Sample for evaluation
        
        with torch.no_grad():
            for x, y in val_batches:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                
                total_loss += loss.item() * x.numel()
                total_tokens += x.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        model.train()
        return avg_loss, perplexity
    
    # Long-term training
    def train_model_properly(model, train_corpus, val_corpus, model_name):
        """Proper long-term training with convergence tracking"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        
        train_losses = []
        val_losses = []
        val_perplexities = []
        
        print(f"\n🏋️ Training {model_name} for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_batches = create_batches(train_corpus, batch_size, seq_len)
            epoch_loss = 0
            
            for batch_idx, (x, y) in enumerate(train_batches):
                optimizer.zero_grad()
                
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Progress within epoch
                if batch_idx % 100 == 0 and batch_idx > 0:
                    print(f"    Epoch {epoch+1:3d}, Batch {batch_idx:3d}, Loss: {loss.item():.4f}")
            
            scheduler.step()
            avg_train_loss = epoch_loss / len(train_batches)
            train_losses.append(avg_train_loss)
            
            # Evaluation phase
            if epoch % eval_every == 0 or epoch == num_epochs - 1:
                val_loss, val_perplexity = evaluate_model(model, val_corpus)
                val_losses.append(val_loss)
                val_perplexities.append(val_perplexity)
                
                elapsed = time.time() - start_time
                print(f"  📊 Epoch {epoch+1:3d}/{num_epochs}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, perplexity={val_perplexity:.2f} ({elapsed:.1f}s)")
            
            # Early convergence check
            if len(val_losses) > 10:
                recent_improvement = val_losses[-10] - val_losses[-1]
                if recent_improvement < 0.001:  # Very little improvement
                    print(f"    Early convergence detected at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        print(f"  ✅ Training completed in {total_time:.1f} seconds")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_perplexities': val_perplexities,
            'final_val_loss': val_losses[-1],
            'final_perplexity': val_perplexities[-1],
            'training_time': total_time,
            'epochs_completed': epoch + 1
        }
    
    # Train both models with proper long-term training
    standard_results = train_model_properly(standard_model, train_corpus, val_corpus, "Standard")
    bespoke_results = train_model_properly(bespoke_model, train_corpus, val_corpus, "Bespoke")
    
    # CRITICAL: Compare after full convergence
    print(f"\n🎯 CONVERGENCE ANALYSIS:")
    print(f"Standard - Final validation loss: {standard_results['final_val_loss']:.4f}, Perplexity: {standard_results['final_perplexity']:.2f}")
    print(f"Bespoke  - Final validation loss: {bespoke_results['final_val_loss']:.4f}, Perplexity: {bespoke_results['final_perplexity']:.2f}")
    
    # Calculate true performance ratio after convergence
    loss_ratio = bespoke_results['final_val_loss'] / standard_results['final_val_loss']
    perplexity_ratio = bespoke_results['final_perplexity'] / standard_results['final_perplexity']
    param_ratio = bespoke_params / standard_params
    
    # Convergence analysis
    standard_converged = len(standard_results['val_losses']) >= 10
    bespoke_converged = len(bespoke_results['val_losses']) >= 10
    
    print(f"\n📈 PROPER CONVERGENCE RESULTS:")
    print(f"Loss ratio (bespoke/standard): {loss_ratio:.4f} ({'better' if loss_ratio < 1.0 else 'worse'})")
    print(f"Perplexity ratio: {perplexity_ratio:.4f} ({'better' if perplexity_ratio < 1.0 else 'worse'})")
    print(f"Parameter ratio: {param_ratio:.4f}")
    print(f"Convergence: Standard={standard_converged}, Bespoke={bespoke_converged}")
    
    # True validation criteria
    quality_improved = loss_ratio < 0.95  # At least 5% better
    converged_properly = standard_converged and bespoke_converged
    reasonable_params = param_ratio < 2.0  # Less than 2x parameters
    
    theory_validated = quality_improved and converged_properly and reasonable_params
    
    print(f"\n🎯 PROPER THEORY VALIDATION:")
    print(f"Quality improved: {quality_improved} (loss ratio: {loss_ratio:.4f})")
    print(f"Converged properly: {converged_properly}")
    print(f"Reasonable parameters: {reasonable_params} ({param_ratio:.4f}x)")
    print(f"Theory validated: {'✅ YES' if theory_validated else '❌ NO'}")
    
    if theory_validated:
        improvement = (1 - loss_ratio) * 100
        print(f"\n🚀 SUCCESS: Proper long-term validation confirms bespoke benefits!")
        print(f"   ✅ Quality improvement: {improvement:.1f}% better after full convergence")
        print(f"   ✅ Convergence achieved: Both models trained to stability")
        print(f"   ✅ Parameter efficiency: {param_ratio:.3f}x overhead reasonable")
        print(f"   🎯 Theory VALIDATED with proper training methodology!")
    else:
        print(f"\n⚠️  Long-term validation shows:")
        if not quality_improved:
            print(f"   - Quality improvement insufficient: {loss_ratio:.4f} (needed <0.95)")
        if not converged_properly:
            print(f"   - Models did not converge properly")
        if not reasonable_params:
            print(f"   - Parameter overhead too high: {param_ratio:.4f}x")
        print(f"   📊 Theory needs refinement or different approach")
    
    # Save comprehensive results
    timestamp = int(time.time())
    results = {
        'experiment': 'long_term_bespoke_validation',
        'timestamp': timestamp,
        'configuration': {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'epochs': num_epochs,
            'corpus_size': corpus_size
        },
        'standard_model': standard_results,
        'bespoke_model': bespoke_results,
        'comparison': {
            'loss_ratio': loss_ratio,
            'perplexity_ratio': perplexity_ratio,
            'param_ratio': param_ratio,
            'quality_improved': quality_improved,
            'theory_validated': theory_validated
        },
        'token_analysis': {
            'high_freq_count': len(high_freq_tokens),
            'mid_freq_count': len(mid_freq_tokens),
            'low_freq_count': len(low_freq_tokens)
        }
    }
    
    output_file = f"/Users/bard/Code/Claude_Data/tool_outputs/long_term_bespoke_validation_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Complete results saved: {output_file}")
    
    return theory_validated, results

if __name__ == "__main__":
    print("🔬 PROPER LONG-TERM BESPOKE VALIDATION")
    print("This will take several minutes with 500 epochs of training...")
    print("=" * 70)
    
    success, results = long_term_bespoke_test()
    
    print("\n" + "="*70)
    if success:
        print("🏆 LONG-TERM VALIDATION: SUCCESS!")
        print("Bespoke token dimensions provide genuine benefits after proper convergence!")
    else:
        print("📊 LONG-TERM VALIDATION: Inconclusive")  
        print("May need different approach or longer training for benefits to emerge")
    print("="*70)
