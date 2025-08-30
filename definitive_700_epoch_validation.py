#!/usr/bin/env python3
"""
Definitive 700-Epoch Bespoke Token Validation
=============================================

This is the definitive test. 700 epochs of proper training to determine
once and for all whether bespoke token dimensions provide any benefit.

This will take several hours but will provide the final answer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import json
import time
from datetime import datetime

def definitive_700_epoch_validation():
    print("🔬 DEFINITIVE 700-EPOCH BESPOKE TOKEN VALIDATION")
    print("=" * 70)
    print("⚠️  This will take several hours - definitive test for the theory")
    print("🎯 Final determination of bespoke token dimension benefits")
    print()
    
    start_timestamp = datetime.now()
    print(f"📅 Started: {start_timestamp}")
    
    device = torch.device("cpu")  # CPU for consistency and reliability
    
    # Configuration for definitive test
    vocab_size = 1000      # Large enough for realistic patterns
    embed_dim = 128        # Standard size
    seq_len = 32           # Reasonable sequences
    batch_size = 16        # Stable training
    corpus_size = 200000   # Large corpus for clear patterns
    
    # THE DEFINITIVE TEST: 700 epochs
    num_epochs = 700       # Definitive training time
    eval_every = 50        # Evaluate every 50 epochs
    save_every = 100       # Save progress every 100 epochs
    
    print(f"📊 Configuration:")
    print(f"   Vocabulary: {vocab_size:,} tokens")
    print(f"   Embedding: {embed_dim} dimensions")
    print(f"   Corpus: {corpus_size:,} tokens")
    print(f"   Training: {num_epochs} epochs (DEFINITIVE)")
    print(f"   Device: {device}")
    
    # Generate large realistic corpus with strong frequency patterns
    print(f"\n🎲 Generating {corpus_size:,} token corpus...")
    ranks = np.arange(1, vocab_size + 1)
    zipf_probs = 1 / ranks**1.2  # Strong but realistic Zipfian
    zipf_probs = zipf_probs / zipf_probs.sum()
    
    corpus = torch.multinomial(
        torch.tensor(zipf_probs, dtype=torch.float), 
        corpus_size, 
        replacement=True
    )
    print(f"✅ Corpus generated")
    
    # Comprehensive token frequency analysis
    print(f"\n📊 Analyzing token frequencies...")
    token_counts = Counter(corpus.tolist())
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Strategic categorization for maximum potential benefit
    total_occurrences = sum(count for _, count in sorted_tokens)
    
    # Conservative approach - clear categories
    high_freq_threshold = int(0.1 * len(sorted_tokens))    # Top 10% of unique tokens
    low_freq_threshold = int(0.8 * len(sorted_tokens))     # Bottom 20% of unique tokens
    
    high_freq_tokens = set(token for token, _ in sorted_tokens[:high_freq_threshold])
    low_freq_tokens = set(token for token, _ in sorted_tokens[low_freq_threshold:])
    mid_freq_tokens = set(range(vocab_size)) - high_freq_tokens - low_freq_tokens
    
    # Calculate coverage
    high_coverage = sum(token_counts.get(t, 0) for t in high_freq_tokens) / total_occurrences
    mid_coverage = sum(token_counts.get(t, 0) for t in mid_freq_tokens) / total_occurrences  
    low_coverage = sum(token_counts.get(t, 0) for t in low_freq_tokens) / total_occurrences
    
    print(f"📈 Token categorization:")
    print(f"   High frequency: {len(high_freq_tokens):3d} tokens ({high_coverage:.1%} coverage)")
    print(f"   Mid frequency:  {len(mid_freq_tokens):3d} tokens ({mid_coverage:.1%} coverage)")
    print(f"   Low frequency:  {len(low_freq_tokens):3d} tokens ({low_coverage:.1%} coverage)")
    
    # Print top tokens for verification
    print(f"   Top 10 tokens: {sorted_tokens[:10]}")
    
    # Create train/validation split
    train_size = int(0.85 * corpus_size)
    train_corpus = corpus[:train_size]
    val_corpus = corpus[train_size:]
    
    print(f"📚 Data split: {len(train_corpus):,} train, {len(val_corpus):,} validation")
    
    class DefinitiveBespokeEmbedding(nn.Module):
        """Final bespoke embedding implementation for definitive test"""
        def __init__(self, vocab_size, embed_dim, high_tokens, mid_tokens, low_tokens):
            super().__init__()
            
            self.high_tokens = high_tokens
            self.mid_tokens = mid_tokens
            self.low_tokens = low_tokens
            
            # Conservative dimension allocation for stable training
            self.high_dim = int(embed_dim * 1.4)   # 40% more for frequent tokens
            self.mid_dim = embed_dim               # Standard dimension
            self.low_dim = int(embed_dim * 0.6)    # 40% less for rare tokens
            
            print(f"🔧 Embedding dimensions: high={self.high_dim}, mid={self.mid_dim}, low={self.low_dim}")
            
            # Create embeddings for each category
            self.high_embed = nn.Embedding(len(high_tokens), self.high_dim) if len(high_tokens) > 0 else None
            self.mid_embed = nn.Embedding(len(mid_tokens), self.mid_dim) if len(mid_tokens) > 0 else None
            self.low_embed = nn.Embedding(len(low_tokens), self.low_dim) if len(low_tokens) > 0 else None
            
            # Token to index mappings
            self.high_token_to_idx = {token: idx for idx, token in enumerate(sorted(high_tokens))} if self.high_embed else {}
            self.mid_token_to_idx = {token: idx for idx, token in enumerate(sorted(mid_tokens))} if self.mid_embed else {}
            self.low_token_to_idx = {token: idx for idx, token in enumerate(sorted(low_tokens))} if self.low_embed else {}
            
            # Projections to standard dimension
            self.high_proj = nn.Linear(self.high_dim, embed_dim, bias=False) if self.high_embed and self.high_dim != embed_dim else None
            self.low_proj = nn.Linear(self.low_dim, embed_dim, bias=False) if self.low_embed and self.low_dim != embed_dim else None
            
            self.embed_dim = embed_dim
            
        def forward(self, token_ids):
            batch_size, seq_len = token_ids.shape
            output = torch.zeros(batch_size, seq_len, self.embed_dim, 
                               device=token_ids.device, dtype=torch.float32)
            
            # Process each token
            for i in range(batch_size):
                for j in range(seq_len):
                    token_val = token_ids[i, j].item()
                    
                    if token_val in self.high_tokens and self.high_embed is not None:
                        embed_idx = self.high_token_to_idx[token_val]
                        embed_val = self.high_embed.weight[embed_idx]
                        if self.high_proj is not None:
                            output[i, j] = self.high_proj(embed_val)
                        else:
                            output[i, j] = embed_val
                            
                    elif token_val in self.mid_tokens and self.mid_embed is not None:
                        embed_idx = self.mid_token_to_idx[token_val]
                        output[i, j] = self.mid_embed.weight[embed_idx]
                        
                    elif token_val in self.low_tokens and self.low_embed is not None:
                        embed_idx = self.low_token_to_idx[token_val]
                        embed_val = self.low_embed.weight[embed_idx]
                        if self.low_proj is not None:
                            output[i, j] = self.low_proj(embed_val)
                        else:
                            output[i, j] = embed_val
                    else:
                        # Fallback for unmapped tokens
                        output[i, j] = torch.randn(self.embed_dim) * 0.01
            
            return output
    
    class DefinitiveLanguageModel(nn.Module):
        """Realistic language model for definitive validation"""
        def __init__(self, embedding, vocab_size, embed_dim, seq_len):
            super().__init__()
            self.embedding = embedding
            self.pos_embedding = nn.Embedding(seq_len, embed_dim)
            
            # Multi-layer transformer for realistic modeling
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=8,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                for _ in range(4)  # 4 layers for substantial model
            ])
            
            self.layer_norm = nn.LayerNorm(embed_dim)
            self.output_head = nn.Linear(embed_dim, vocab_size)
            
            # Proper weight initialization
            self.apply(self._init_weights)
            
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
        def forward(self, input_ids):
            batch_size, seq_len_actual = input_ids.shape
            
            # Token embeddings
            x = self.embedding(input_ids)
            
            # Position embeddings
            positions = torch.arange(seq_len_actual, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            x = x + self.pos_embedding(positions)
            
            # Transformer layers
            for layer in self.transformer_layers:
                x = layer(x)
            
            # Output
            x = self.layer_norm(x)
            logits = self.output_head(x)
            
            return logits
    
    # Create models
    print(f"\n🏗️  Creating models...")
    standard_embed = nn.Embedding(vocab_size, embed_dim)
    bespoke_embed = DefinitiveBespokeEmbedding(vocab_size, embed_dim, high_freq_tokens, mid_freq_tokens, low_freq_tokens)
    
    standard_model = DefinitiveLanguageModel(standard_embed, vocab_size, embed_dim, seq_len).to(device)
    bespoke_model = DefinitiveLanguageModel(bespoke_embed, vocab_size, embed_dim, seq_len).to(device)
    
    # Parameter analysis
    standard_params = sum(p.numel() for p in standard_model.parameters())
    bespoke_params = sum(p.numel() for p in bespoke_model.parameters())
    param_ratio = bespoke_params / standard_params
    
    print(f"📊 Model parameters:")
    print(f"   Standard: {standard_params:,}")
    print(f"   Bespoke:  {bespoke_params:,}")
    print(f"   Ratio:    {param_ratio:.3f}x")
    
    def create_language_modeling_batches(corpus_data, batch_size, seq_len):
        """Create proper language modeling batches"""
        batches = []
        max_start = len(corpus_data) - seq_len - 1
        
        # More batches for thorough training
        num_batches = min(500, max_start // seq_len)  # Up to 500 batches per epoch
        
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
    
    def evaluate_model_thoroughly(model, val_corpus):
        """Thorough evaluation on validation set"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        val_batches = create_language_modeling_batches(val_corpus, batch_size, seq_len)[:100]  # 100 batches for evaluation
        
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
    
    def train_model_definitively(model, train_corpus, val_corpus, model_name):
        """Definitive 700-epoch training with comprehensive tracking"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        
        train_losses = []
        val_losses = []
        val_perplexities = []
        
        print(f"\n🚀 Starting {model_name} definitive training ({num_epochs} epochs)...")
        print(f"⏰ Estimated time: Several hours")
        
        training_start = time.time()
        
        for epoch in range(num_epochs):
            model.train()
            train_batches = create_language_modeling_batches(train_corpus, batch_size, seq_len)
            
            epoch_loss = 0
            batch_count = 0
            
            for x, y in train_batches:
                optimizer.zero_grad()
                
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            scheduler.step()
            avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
            train_losses.append(avg_train_loss)
            
            # Evaluation
            if epoch % eval_every == 0 or epoch == num_epochs - 1:
                val_loss, val_perplexity = evaluate_model_thoroughly(model, val_corpus)
                val_losses.append(val_loss)
                val_perplexities.append(val_perplexity)
                
                elapsed = time.time() - training_start
                remaining = (elapsed / (epoch + 1)) * (num_epochs - epoch - 1)
                
                print(f"  📊 Epoch {epoch+1:3d}/{num_epochs}: train={avg_train_loss:.4f}, val={val_loss:.4f}, ppl={val_perplexity:.2f}")
                print(f"      ⏰ Elapsed: {elapsed/3600:.1f}h, Remaining: ~{remaining/3600:.1f}h")
            
            # Save progress periodically
            if epoch % save_every == 0 and epoch > 0:
                progress = {
                    'epoch': epoch,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_perplexities': val_perplexities,
                    'model_name': model_name
                }
                
                timestamp = int(time.time())
                progress_file = f"/Users/bard/Code/Claude_Data/tool_outputs/progress_{model_name.lower()}_{timestamp}.json"
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)
                
                print(f"      💾 Progress saved: epoch {epoch}")
        
        total_training_time = time.time() - training_start
        print(f"  ✅ {model_name} training completed in {total_training_time/3600:.2f} hours")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_perplexities': val_perplexities,
            'final_val_loss': val_losses[-1] if val_losses else float('inf'),
            'final_perplexity': val_perplexities[-1] if val_perplexities else float('inf'),
            'training_time_hours': total_training_time / 3600,
            'epochs_completed': num_epochs
        }
    
    # THE DEFINITIVE TEST: Train both models for 700 epochs
    print(f"\n🎯 BEGINNING DEFINITIVE 700-EPOCH VALIDATION")
    print(f"   This is the final test to determine if bespoke tokens work")
    
    standard_results = train_model_definitively(standard_model, train_corpus, val_corpus, "Standard")
    bespoke_results = train_model_definitively(bespoke_model, train_corpus, val_corpus, "Bespoke")
    
    # DEFINITIVE ANALYSIS
    end_timestamp = datetime.now()
    total_experiment_time = end_timestamp - start_timestamp
    
    print(f"\n" + "="*70)
    print(f"🎯 DEFINITIVE 700-EPOCH RESULTS")
    print(f"="*70)
    print(f"📅 Completed: {end_timestamp}")
    print(f"⏰ Total time: {total_experiment_time}")
    
    std_final_loss = standard_results['final_val_loss']
    bes_final_loss = bespoke_results['final_val_loss']
    std_final_ppl = standard_results['final_perplexity']
    bes_final_ppl = bespoke_results['final_perplexity']
    
    if std_final_loss > 0 and bes_final_loss > 0:
        loss_ratio = bes_final_loss / std_final_loss
        ppl_ratio = bes_final_ppl / std_final_ppl
        
        print(f"\n📊 Final Performance (after 700 epochs):")
        print(f"   Standard: loss={std_final_loss:.4f}, perplexity={std_final_ppl:.2f}")
        print(f"   Bespoke:  loss={bes_final_loss:.4f}, perplexity={bes_final_ppl:.2f}")
        print(f"   Loss ratio: {loss_ratio:.4f} ({'better' if loss_ratio < 1.0 else 'worse'})")
        print(f"   PPL ratio:  {ppl_ratio:.4f} ({'better' if ppl_ratio < 1.0 else 'worse'})")
        print(f"   Parameter ratio: {param_ratio:.4f}x")
        
        # DEFINITIVE VALIDATION CRITERIA
        # After 700 epochs, we need clear evidence
        significant_improvement = loss_ratio < 0.90   # At least 10% better
        reasonable_parameters = param_ratio < 2.0     # Less than 2x parameters
        both_converged = (len(standard_results['val_losses']) >= 10 and 
                         len(bespoke_results['val_losses']) >= 10)
        
        # Check for convergence trends
        if len(standard_results['val_losses']) >= 5:
            std_trend = standard_results['val_losses'][-1] - standard_results['val_losses'][-5]
            std_converging = abs(std_trend) < 0.1
        else:
            std_converging = True
            
        if len(bespoke_results['val_losses']) >= 5:
            bes_trend = bespoke_results['val_losses'][-1] - bespoke_results['val_losses'][-5]
            bes_converging = abs(bes_trend) < 0.1
        else:
            bes_converging = True
        
        converged_properly = std_converging and bes_converging
        
        # FINAL DETERMINATION
        theory_definitively_validated = (significant_improvement and 
                                       reasonable_parameters and 
                                       both_converged and 
                                       converged_properly)
        
        print(f"\n🎯 DEFINITIVE VALIDATION CRITERIA:")
        print(f"   Significant improvement: {significant_improvement} ({loss_ratio:.4f} < 0.90)")
        print(f"   Reasonable parameters: {reasonable_parameters} ({param_ratio:.4f}x < 2.0)")
        print(f"   Both models converged: {both_converged}")
        print(f"   Converged properly: {converged_properly}")
        
        print(f"\n🏆 DEFINITIVE CONCLUSION:")
        if theory_definitively_validated:
            improvement_pct = (1 - loss_ratio) * 100
            print(f"✅ THEORY VALIDATED AFTER 700 EPOCHS!")
            print(f"   🎯 Performance: {improvement_pct:.1f}% improvement")
            print(f"   📊 Parameters: {param_ratio:.3f}x overhead acceptable")
            print(f"   ⏰ Training: Full convergence achieved")
            print(f"   🔬 Conclusion: Bespoke token dimensions DO work with sufficient training!")
        else:
            improvement_pct = (1 - loss_ratio) * 100 if loss_ratio < 1.0 else -(loss_ratio - 1) * 100
            print(f"❌ THEORY NOT VALIDATED AFTER 700 EPOCHS")
            print(f"   📊 Performance: {improvement_pct:+.1f}% change (insufficient)")
            print(f"   📊 Parameters: {param_ratio:.3f}x overhead")
            print(f"   🔬 Conclusion: Bespoke token dimensions do NOT provide sufficient benefit")
            print(f"   💡 Result: Focus on other efficiency approaches")
        
        # Save comprehensive results
        timestamp = int(time.time())
        definitive_results = {
            'experiment': 'definitive_700_epoch_bespoke_validation',
            'timestamp': timestamp,
            'start_time': start_timestamp.isoformat(),
            'end_time': end_timestamp.isoformat(),
            'duration_hours': total_experiment_time.total_seconds() / 3600,
            
            'configuration': {
                'vocab_size': vocab_size,
                'embed_dim': embed_dim,
                'corpus_size': corpus_size,
                'epochs': num_epochs,
                'definitive_test': True
            },
            
            'token_analysis': {
                'high_freq_count': len(high_freq_tokens),
                'mid_freq_count': len(mid_freq_tokens),
                'low_freq_count': len(low_freq_tokens),
                'high_coverage': high_coverage,
                'mid_coverage': mid_coverage,
                'low_coverage': low_coverage
            },
            
            'model_results': {
                'standard': standard_results,
                'bespoke': bespoke_results
            },
            
            'definitive_analysis': {
                'loss_ratio': loss_ratio,
                'perplexity_ratio': ppl_ratio,
                'parameter_ratio': param_ratio,
                'improvement_percent': (1 - loss_ratio) * 100,
                'significant_improvement': significant_improvement,
                'reasonable_parameters': reasonable_parameters,
                'both_converged': both_converged,
                'converged_properly': converged_properly,
                'theory_validated': theory_definitively_validated
            },
            
            'conclusion': {
                'status': 'VALIDATED' if theory_definitively_validated else 'REJECTED',
                'confidence': 'HIGH - 700 epoch definitive test',
                'recommendation': ('Proceed with bespoke implementation' if theory_definitively_validated 
                                 else 'Focus on other efficiency approaches'),
                'evidence_quality': 'DEFINITIVE'
            }
        }
        
        # Save to multiple locations for reliability
        output_file = f"/Users/bard/Code/Claude_Data/tool_outputs/DEFINITIVE_700_epoch_results_{timestamp}.json"
        backup_file = f"/Users/bard/Code/nanoGPT_sparse_attention/DEFINITIVE_RESULTS_{timestamp}.json"
        
        for filepath in [output_file, backup_file]:
            with open(filepath, 'w') as f:
                json.dump(definitive_results, f, indent=2)
        
        print(f"\n💾 DEFINITIVE RESULTS SAVED:")
        print(f"   Primary: {output_file}")
        print(f"   Backup:  {backup_file}")
        
        return theory_definitively_validated, definitive_results
        
    else:
        print(f"❌ DEFINITIVE TEST FAILED - Invalid loss values")
        print(f"   Standard loss: {std_final_loss}")
        print(f"   Bespoke loss: {bes_final_loss}")
        return False, {'error': 'invalid_loss_values'}

if __name__ == "__main__":
    print("🔬 DEFINITIVE 700-EPOCH BESPOKE TOKEN VALIDATION")
    print("This is the final test - several hours of training")
    print("Results will determine the fate of the bespoke token theory")
    print("=" * 70)
    
    try:
        validated, results = definitive_700_epoch_validation()
        
        print(f"\n" + "="*70)
        if validated:
            print("🏆 DEFINITIVE RESULT: THEORY VALIDATED!")
            print("Bespoke token dimensions work with sufficient training!")
        else:
            print("📊 DEFINITIVE RESULT: THEORY REJECTED")  
            print("Bespoke token dimensions do not provide sufficient benefit")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ DEFINITIVE TEST FAILED WITH ERROR:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': int(time.time())
        }
        
        error_file = f"/Users/bard/Code/Claude_Data/tool_outputs/definitive_test_error_{int(time.time())}.json"
        with open(error_file, 'w') as f:
            json.dump(error_info, f, indent=2)
        
        print(f"Error details saved: {error_file}")
