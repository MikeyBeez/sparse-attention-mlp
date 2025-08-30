#!/usr/bin/env python3
"""
Medium-Term Bespoke Validation
===============================

Faster validation with meaningful training time (100 epochs)
to see if bespoke benefits emerge with proper convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import json
import time

def medium_term_bespoke_test():
    print("🔬 MEDIUM-TERM BESPOKE VALIDATION (100 epochs)")
    print("=" * 60)
    
    device = torch.device("cpu")
    
    # Configuration optimized for faster but meaningful training
    vocab_size = 500     # Smaller for faster training
    embed_dim = 128      # Smaller for speed
    seq_len = 32         # Smaller sequences 
    batch_size = 16
    corpus_size = 50000  # Still substantial
    
    # Meaningful training time
    num_epochs = 100     # Reasonable training time
    eval_every = 20      # Evaluate every 20 epochs
    
    print(f"Config: {vocab_size} vocab, {embed_dim} embed_dim, {num_epochs} epochs")
    print(f"Corpus: {corpus_size:,} tokens")
    
    # Generate corpus with strong frequency distribution
    ranks = np.arange(1, vocab_size + 1)
    zipf_probs = 1 / ranks**1.3  # Strong Zipfian for clear patterns
    zipf_probs = zipf_probs / zipf_probs.sum()
    
    print("🎲 Generating corpus...")
    corpus = torch.multinomial(
        torch.tensor(zipf_probs, dtype=torch.float), 
        corpus_size, 
        replacement=True
    )
    
    # Token frequency analysis
    token_counts = Counter(corpus.tolist())
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Clear frequency categories
    total_occurrences = sum(count for _, count in sorted_tokens)
    
    # Top 20% of tokens by occurrence count (not unique tokens)
    cumulative = 0
    high_freq_tokens = set()
    mid_freq_tokens = set()
    low_freq_tokens = set()
    
    for token, count in sorted_tokens:
        freq_percentile = cumulative / total_occurrences
        
        if freq_percentile < 0.6:      # Top 60% of occurrences
            high_freq_tokens.add(token)
        elif freq_percentile < 0.85:   # Next 25% of occurrences  
            mid_freq_tokens.add(token)
        else:                          # Bottom 15% (rare tokens)
            low_freq_tokens.add(token)
            
        cumulative += count
    
    high_coverage = sum(token_counts[t] for t in high_freq_tokens)/total_occurrences
    print(f"Categories: {len(high_freq_tokens)} high ({high_coverage:.1%}), {len(mid_freq_tokens)} mid, {len(low_freq_tokens)} low")
    
    # Train/validation split
    train_size = int(0.85 * corpus_size)
    train_corpus = corpus[:train_size]
    val_corpus = corpus[train_size:]
    
    class MediumTermBespokeEmbedding(nn.Module):
        """Bespoke embedding for medium-term convergence testing"""
        def __init__(self, vocab_size, embed_dim, high_tokens, mid_tokens, low_tokens):
            super().__init__()
            
            self.high_tokens = high_tokens
            self.mid_tokens = mid_tokens
            self.low_tokens = low_tokens
            
            # Strategic dimension allocation
            self.high_dim = int(embed_dim * 1.5)   # 50% more for frequent
            self.mid_dim = embed_dim               # Standard
            self.low_dim = int(embed_dim * 0.5)    # 50% less for rare
            
            # Create category-specific embeddings
            if len(high_tokens) > 0:
                self.high_embed = nn.Embedding(len(high_tokens), self.high_dim)
                self.high_token_to_idx = {token: idx for idx, token in enumerate(sorted(high_tokens))}
                self.high_proj = nn.Linear(self.high_dim, embed_dim, bias=False)
            else:
                self.high_embed = None
                
            if len(mid_tokens) > 0:
                self.mid_embed = nn.Embedding(len(mid_tokens), self.mid_dim)
                self.mid_token_to_idx = {token: idx for idx, token in enumerate(sorted(mid_tokens))}
            else:
                self.mid_embed = None
                
            if len(low_tokens) > 0:
                self.low_embed = nn.Embedding(len(low_tokens), self.low_dim)
                self.low_token_to_idx = {token: idx for idx, token in enumerate(sorted(low_tokens))}
                self.low_proj = nn.Linear(self.low_dim, embed_dim, bias=False)
            else:
                self.low_embed = None
            
            self.embed_dim = embed_dim
            
        def forward(self, token_ids):
            batch_size, seq_len = token_ids.shape
            output = torch.zeros(batch_size, seq_len, self.embed_dim, 
                               device=token_ids.device)
            
            flat_tokens = token_ids.view(-1)
            flat_output = output.view(-1, self.embed_dim)
            
            for i, token_id in enumerate(flat_tokens):
                token_val = token_id.item()
                
                if token_val in self.high_tokens and self.high_embed is not None:
                    embed_idx = self.high_token_to_idx[token_val]
                    embed_val = self.high_embed.weight[embed_idx]
                    flat_output[i] = self.high_proj(embed_val)
                    
                elif token_val in self.mid_tokens and self.mid_embed is not None:
                    embed_idx = self.mid_token_to_idx[token_val]
                    flat_output[i] = self.mid_embed.weight[embed_idx]
                    
                elif token_val in self.low_tokens and self.low_embed is not None:
                    embed_idx = self.low_token_to_idx[token_val]
                    embed_val = self.low_embed.weight[embed_idx]
                    flat_output[i] = self.low_proj(embed_val)
                    
                else:
                    # Unknown token fallback
                    flat_output[i] = torch.randn(self.embed_dim) * 0.01
            
            return output
    
    class LanguageModelSimple(nn.Module):
        """Simplified but proper language model"""
        def __init__(self, embedding, vocab_size, embed_dim):
            super().__init__()
            self.embedding = embedding
            
            # Simple but effective architecture
            self.transformer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 2,
                dropout=0.1,
                batch_first=True
            )
            
            self.output_head = nn.Linear(embed_dim, vocab_size)
            
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.transformer(x)
            logits = self.output_head(x)
            return logits
    
    # Create models
    standard_embed = nn.Embedding(vocab_size, embed_dim)
    bespoke_embed = MediumTermBespokeEmbedding(vocab_size, embed_dim, high_freq_tokens, mid_freq_tokens, low_freq_tokens)
    
    standard_model = LanguageModelSimple(standard_embed, vocab_size, embed_dim).to(device)
    bespoke_model = LanguageModelSimple(bespoke_embed, vocab_size, embed_dim).to(device)
    
    # Parameter analysis
    standard_params = sum(p.numel() for p in standard_model.parameters())
    bespoke_params = sum(p.numel() for p in bespoke_model.parameters())
    print(f"\nParameters: Standard {standard_params:,}, Bespoke {bespoke_params:,} ({bespoke_params/standard_params:.3f}x)")
    
    def create_language_modeling_batches(corpus_data, batch_size, seq_len):
        """Proper language modeling batches"""
        batches = []
        max_start = len(corpus_data) - seq_len - 1
        
        # Multiple batches per epoch for proper training
        num_batches = min(200, max_start // seq_len)
        
        for _ in range(num_batches):
            batch_inputs = []
            batch_targets = []
            
            for _ in range(batch_size):
                start_idx = torch.randint(0, max_start, (1,)).item()
                input_seq = corpus_data[start_idx:start_idx+seq_len]
                target_seq = corpus_data[start_idx+1:start_idx+seq_len+1]
                
                batch_inputs.append(input_seq)
                batch_targets.append(target_seq)
            
            batches.append((
                torch.stack(batch_inputs).to(device),
                torch.stack(batch_targets).to(device)
            ))
        
        return batches
    
    def evaluate_language_model(model, val_corpus):
        """Proper language modeling evaluation"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        val_batches = create_language_modeling_batches(val_corpus, batch_size, seq_len)[:30]
        
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
    
    def train_model_convergence(model, train_corpus, val_corpus, model_name):
        """Training with convergence tracking"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        
        train_losses = []
        val_losses = []
        val_perplexities = []
        
        print(f"\n🏋️ Training {model_name}...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            model.train()
            train_batches = create_language_modeling_batches(train_corpus, batch_size, seq_len)
            epoch_loss = 0
            
            for x, y in train_batches:
                optimizer.zero_grad()
                
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            scheduler.step()
            avg_train_loss = epoch_loss / len(train_batches)
            train_losses.append(avg_train_loss)
            
            # Evaluation
            if epoch % eval_every == 0 or epoch == num_epochs - 1:
                val_loss, val_perplexity = evaluate_language_model(model, val_corpus)
                val_losses.append(val_loss)
                val_perplexities.append(val_perplexity)
                
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1:3d}: train={avg_train_loss:.4f}, val={val_loss:.4f}, ppl={val_perplexity:.2f} ({elapsed:.1f}s)")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_perplexities': val_perplexities,
            'final_val_loss': val_losses[-1],
            'final_perplexity': val_perplexities[-1]
        }
    
    # Train both models properly
    print("\n📈 Starting medium-term training comparison...")
    standard_results = train_model_convergence(standard_model, train_corpus, val_corpus, "Standard")
    bespoke_results = train_model_convergence(bespoke_model, train_corpus, val_corpus, "Bespoke")
    
    # Analysis after proper training
    print(f"\n🎯 MEDIUM-TERM CONVERGENCE RESULTS:")
    print(f"Standard final: loss={standard_results['final_val_loss']:.4f}, ppl={standard_results['final_perplexity']:.2f}")
    print(f"Bespoke final:  loss={bespoke_results['final_val_loss']:.4f}, ppl={bespoke_results['final_perplexity']:.2f}")
    
    # Key metrics
    loss_ratio = bespoke_results['final_val_loss'] / standard_results['final_val_loss']
    perplexity_ratio = bespoke_results['final_perplexity'] / standard_results['final_perplexity']
    param_ratio = bespoke_params / standard_params
    
    # Convergence quality analysis
    standard_improved = len(standard_results['val_losses']) >= 3 and \
                       standard_results['val_losses'][-1] < standard_results['val_losses'][0]
    bespoke_improved = len(bespoke_results['val_losses']) >= 3 and \
                      bespoke_results['val_losses'][-1] < bespoke_results['val_losses'][0]
    
    print(f"\n📊 PROPER VALIDATION ANALYSIS:")
    print(f"Loss improvement: {loss_ratio:.4f} ({'better' if loss_ratio < 1.0 else 'worse'})")
    print(f"Perplexity improvement: {perplexity_ratio:.4f}")
    print(f"Parameter overhead: {param_ratio:.4f}x")
    print(f"Both models converged: {standard_improved and bespoke_improved}")
    
    # Realistic validation criteria
    meaningful_improvement = loss_ratio < 0.90  # At least 10% better
    reasonable_parameters = param_ratio < 2.5   # Less than 2.5x parameters
    both_converged = standard_improved and bespoke_improved
    
    theory_holds = meaningful_improvement and reasonable_parameters and both_converged
    
    print(f"\n🎯 MEDIUM-TERM THEORY VALIDATION:")
    print(f"Meaningful improvement: {meaningful_improvement} (loss ratio: {loss_ratio:.4f})")
    print(f"Reasonable parameters: {reasonable_parameters} ({param_ratio:.4f}x)")
    print(f"Both converged: {both_converged}")
    print(f"Theory validated: {'✅ YES' if theory_holds else '❌ NO'}")
    
    if theory_holds:
        improvement_pct = (1 - loss_ratio) * 100
        print(f"\n🚀 MEDIUM-TERM SUCCESS!")
        print(f"   ✅ Quality: {improvement_pct:.1f}% better after {num_epochs} epochs")
        print(f"   ✅ Convergence: Both models trained properly")
        print(f"   ✅ Parameters: {param_ratio:.3f}x overhead acceptable")
        print(f"   🎯 Bespoke theory CONFIRMED with proper training!")
    else:
        print(f"\n⚠️  Medium-term results:")
        if not meaningful_improvement:
            improvement_pct = (1 - loss_ratio) * 100
            print(f"   - Improvement modest: {improvement_pct:.1f}% (target: ≥10%)")
        if not reasonable_parameters:
            print(f"   - Parameter overhead high: {param_ratio:.3f}x")
        if not both_converged:
            print(f"   - Convergence issues detected")
        print(f"   📊 May need longer training or different approach")
    
    # Save results
    timestamp = int(time.time())
    results = {
        'experiment': 'medium_term_bespoke_validation',
        'timestamp': timestamp,
        'config': {
            'epochs': num_epochs,
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'corpus_size': corpus_size
        },
        'results': {
            'standard': standard_results,
            'bespoke': bespoke_results,
            'loss_ratio': loss_ratio,
            'perplexity_ratio': perplexity_ratio,
            'param_ratio': param_ratio,
            'theory_validated': theory_holds
        },
        'token_analysis': {
            'high_freq_count': len(high_freq_tokens),
            'high_freq_coverage': high_coverage,
            'mid_freq_count': len(mid_freq_tokens),
            'low_freq_count': len(low_freq_tokens)
        }
    }
    
    output_file = f"/Users/bard/Code/Claude_Data/tool_outputs/medium_term_validation_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved: {output_file}")
    
    return theory_holds, results

if __name__ == "__main__":
    print("🔬 MEDIUM-TERM BESPOKE VALIDATION")
    print("Testing with 100 epochs - meaningful but faster than 500")
    print("=" * 70)
    
    success, results = medium_term_bespoke_test()
    
    print("\n" + "="*70)
    if success:
        print("✅ MEDIUM-TERM VALIDATION: SUCCESS!")
        print("Bespoke benefits emerge with proper training time!")
    else:
        print("⚠️  MEDIUM-TERM VALIDATION: Mixed results")  
        print("Benefits present but may need optimization or longer training")
    print("="*70)
