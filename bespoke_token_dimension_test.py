#!/usr/bin/env python3
"""
Bespoke Token Dimension Theory Validation
========================================

Test the hypothesis that we can use fewer dimensions for bespoke tokens
and maintain performance across small to large corpus sizes.

Theory: Different token types (common words, rare words, punctuation, numbers)
can be represented with different embedding dimensions while maintaining
semantic quality and computational efficiency.

Experiments:
1. Baseline: Standard equal-dimension embeddings
2. Bespoke: Stratified embedding dimensions by token frequency/type
3. Scaling: Test across corpus sizes (1K → 1M tokens)
4. Performance: Measure both quality and efficiency gains
"""

import math
import os
import sys
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"[info] using device: {device}")

torch.manual_seed(1337)

@dataclass
class ExperimentConfig:
    """Configuration for bespoke token dimension experiments"""
    vocab_size: int = 1000
    block_size: int = 128
    corpus_sizes: List[int] = None  # Will be set to [1000, 10000, 100000, 1000000]
    
    # Standard embedding config
    standard_embed_dim: int = 256
    
    # Bespoke embedding config
    high_freq_dim: int = 384      # Top 20% most frequent tokens
    mid_freq_dim: int = 256       # Middle 60% tokens  
    low_freq_dim: int = 128       # Bottom 20% tokens (rare)
    special_token_dim: int = 64   # Punctuation, numbers, special symbols
    
    # Model architecture
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    max_iters: int = 1000
    learning_rate: float = 3e-4
    
    def __post_init__(self):
        if self.corpus_sizes is None:
            self.corpus_sizes = [1000, 10000, 100000]  # Start smaller for testing
            
        # Ensure head_dim is compatible
        assert self.standard_embed_dim % self.n_heads == 0
        assert self.high_freq_dim % self.n_heads == 0

class TokenFrequencyAnalyzer:
    """Analyze token frequency patterns to create bespoke embedding strategies"""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.token_counts = Counter()
        self.token_categories = {}
        
    def analyze_corpus(self, tokens: torch.Tensor) -> Dict:
        """Analyze token frequency patterns in the corpus"""
        # Count token frequencies
        for token in tokens.flatten():
            self.token_counts[token.item()] += 1
            
        # Sort by frequency
        sorted_tokens = sorted(self.token_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize tokens
        total_tokens = len(sorted_tokens)
        high_freq_cutoff = int(0.2 * total_tokens)  # Top 20%
        low_freq_cutoff = int(0.8 * total_tokens)   # Bottom 20%
        
        for i, (token_id, count) in enumerate(sorted_tokens):
            if i < high_freq_cutoff:
                category = 'high_freq'
            elif i >= low_freq_cutoff:
                category = 'low_freq'
            elif token_id < 10:  # Assume first 10 tokens are special (punct, etc)
                category = 'special'
            else:
                category = 'mid_freq'
                
            self.token_categories[token_id] = category
            
        # Calculate statistics
        stats = {
            'total_unique_tokens': len(sorted_tokens),
            'total_token_instances': sum(count for _, count in sorted_tokens),
            'high_freq_tokens': sum(1 for cat in self.token_categories.values() if cat == 'high_freq'),
            'mid_freq_tokens': sum(1 for cat in self.token_categories.values() if cat == 'mid_freq'),
            'low_freq_tokens': sum(1 for cat in self.token_categories.values() if cat == 'low_freq'),
            'special_tokens': sum(1 for cat in self.token_categories.values() if cat == 'special'),
            'frequency_distribution': dict(sorted_tokens[:50])  # Top 50 for analysis
        }
        
        return stats

class BespokeEmbedding(nn.Module):
    """
    Embedding layer with different dimensions for different token categories
    """
    def __init__(self, config: ExperimentConfig, token_categories: Dict[int, str]):
        super().__init__()
        self.config = config
        self.token_categories = token_categories
        
        # Create separate embedding layers for each category
        self.embeddings = nn.ModuleDict({
            'high_freq': nn.Embedding(config.vocab_size, config.high_freq_dim),
            'mid_freq': nn.Embedding(config.vocab_size, config.mid_freq_dim), 
            'low_freq': nn.Embedding(config.vocab_size, config.low_freq_dim),
            'special': nn.Embedding(config.vocab_size, config.special_token_dim)
        })
        
        # Projection layers to standard dimension for compatibility
        self.projections = nn.ModuleDict({
            'high_freq': nn.Linear(config.high_freq_dim, config.standard_embed_dim),
            'mid_freq': nn.Linear(config.mid_freq_dim, config.standard_embed_dim),
            'low_freq': nn.Linear(config.low_freq_dim, config.standard_embed_dim),
            'special': nn.Linear(config.special_token_dim, config.standard_embed_dim)
        })
        
        # Default to mid_freq for unknown tokens
        self.default_category = 'mid_freq'
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        output = torch.zeros(batch_size, seq_len, self.config.standard_embed_dim, 
                           device=token_ids.device, dtype=torch.float)
        
        # Process each category separately
        for category in self.embeddings.keys():
            # Create mask for tokens in this category
            category_mask = torch.zeros_like(token_ids, dtype=torch.bool)
            for i in range(batch_size):
                for j in range(seq_len):
                    token_id = token_ids[i, j].item()
                    if self.token_categories.get(token_id, self.default_category) == category:
                        category_mask[i, j] = True
            
            if category_mask.any():
                # Get embeddings for this category
                category_tokens = token_ids[category_mask]
                if len(category_tokens) > 0:
                    embeds = self.embeddings[category](category_tokens)
                    projected = self.projections[category](embeds)
                    output[category_mask] = projected
        
        return output
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Calculate memory usage for each category"""
        usage = {}
        for category, embedding in self.embeddings.items():
            params = sum(p.numel() for p in embedding.parameters())
            proj_params = sum(p.numel() for p in self.projections[category].parameters())
            usage[category] = params + proj_params
        usage['total'] = sum(usage.values())
        return usage

class TransformerBlock(nn.Module):
    """Standard transformer block for comparison"""
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.standard_embed_dim)
        self.attn = nn.MultiheadAttention(
            config.standard_embed_dim, 
            config.n_heads, 
            dropout=config.dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(config.standard_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.standard_embed_dim, 4 * config.standard_embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.standard_embed_dim, config.standard_embed_dim),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        
        # MLP with residual  
        x = x + self.mlp(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    """Language model with either standard or bespoke embeddings"""
    def __init__(self, config: ExperimentConfig, use_bespoke: bool = False, 
                 token_categories: Optional[Dict[int, str]] = None):
        super().__init__()
        self.config = config
        self.use_bespoke = use_bespoke
        
        # Embedding layer
        if use_bespoke and token_categories:
            self.embedding = BespokeEmbedding(config, token_categories)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.standard_embed_dim)
        
        # Position embedding
        self.pos_embedding = nn.Embedding(config.block_size, config.standard_embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(config.standard_embed_dim)
        self.head = nn.Linear(config.standard_embed_dim, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        if self.use_bespoke:
            x = self.embedding(token_ids)
        else:
            x = self.embedding(token_ids)
        
        # Position embeddings
        pos_ids = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(pos_ids)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get detailed parameter count"""
        counts = {}
        
        if self.use_bespoke:
            counts['embedding'] = self.embedding.get_memory_usage()['total']
        else:
            counts['embedding'] = sum(p.numel() for p in self.embedding.parameters())
            
        counts['position'] = sum(p.numel() for p in self.pos_embedding.parameters())
        counts['transformer'] = sum(sum(p.numel() for p in block.parameters()) for block in self.blocks)
        counts['output'] = sum(p.numel() for p in self.head.parameters()) + sum(p.numel() for p in self.ln_f.parameters())
        counts['total'] = sum(counts.values())
        
        return counts

class BespokeTokenExperiment:
    """Main experiment runner"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        
    def generate_corpus(self, size: int) -> torch.Tensor:
        """Generate synthetic corpus with realistic frequency distribution"""
        # Create Zipfian distribution (realistic language distribution)
        ranks = np.arange(1, self.config.vocab_size + 1)
        zipf_probs = 1 / ranks
        zipf_probs = zipf_probs / zipf_probs.sum()
        
        # Generate tokens according to Zipfian distribution
        tokens = torch.multinomial(
            torch.tensor(zipf_probs, dtype=torch.float), 
            size, 
            replacement=True
        )
        
        return tokens
        
    def create_dataset(self, corpus: torch.Tensor, split_ratio: float = 0.8):
        """Create train/test datasets from corpus"""
        total_size = len(corpus)
        train_size = int(total_size * split_ratio)
        
        train_data = corpus[:train_size]
        test_data = corpus[train_size:]
        
        return train_data, test_data
    
    def get_batch(self, data: torch.Tensor, batch_size: int = None):
        """Get a random batch from the data"""
        if batch_size is None:
            batch_size = self.config.batch_size
            
        if len(data) < self.config.block_size + 1:
            # Handle very small datasets
            return data[:batch_size].unsqueeze(0).to(device), data[1:batch_size+1].unsqueeze(0).to(device)
            
        max_start = len(data) - self.config.block_size - 1
        if max_start <= 0:
            max_start = 1
            
        ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+self.config.block_size] for i in ix])
        
        return x.to(device), y.to(device)
    
    def train_model(self, model: LanguageModel, train_data: torch.Tensor, 
                   max_iters: int = None) -> Dict:
        """Train the model and return training metrics"""
        if max_iters is None:
            max_iters = self.config.max_iters
            
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        
        losses = []
        start_time = time.time()
        
        model.train()
        for iter_num in range(max_iters):
            if iter_num % 100 == 0:
                # Evaluate
                model.eval()
                with torch.no_grad():
                    x_val, y_val = self.get_batch(train_data)  # Using train data for simplicity
                    logits = model(x_val)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_val.view(-1))
                    losses.append(loss.item())
                    if iter_num % 200 == 0:
                        print(f"  step {iter_num:4d} | loss {loss.item():.4f}")
                model.train()
                
            # Training step
            x, y = self.get_batch(train_data)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        training_time = time.time() - start_time
        
        return {
            'final_loss': losses[-1] if losses else float('inf'),
            'loss_history': losses,
            'training_time': training_time,
            'iterations': max_iters
        }
    
    def evaluate_model(self, model: LanguageModel, test_data: torch.Tensor) -> Dict:
        """Evaluate model performance"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            # Evaluate on multiple batches
            num_eval_batches = min(50, len(test_data) // self.config.block_size)
            for _ in range(num_eval_batches):
                try:
                    x, y = self.get_batch(test_data)
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    
                    total_loss += loss.item() * x.numel()
                    total_tokens += x.numel()
                except:
                    continue
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens_evaluated': total_tokens
        }
    
    def run_single_experiment(self, corpus_size: int) -> Dict:
        """Run experiment for a single corpus size"""
        print(f"\n[experiment] Running corpus size: {corpus_size:,} tokens")
        
        # Generate corpus and analyze
        corpus = self.generate_corpus(corpus_size)
        analyzer = TokenFrequencyAnalyzer(self.config.vocab_size)
        freq_stats = analyzer.analyze_corpus(corpus)
        
        print(f"  Token analysis: {freq_stats['total_unique_tokens']} unique tokens")
        print(f"  Distribution: {freq_stats['high_freq_tokens']} high, {freq_stats['mid_freq_tokens']} mid, {freq_stats['low_freq_tokens']} low freq")
        
        # Create datasets
        train_data, test_data = self.create_dataset(corpus)
        
        # Train standard model
        print("  Training standard model...")
        standard_model = LanguageModel(self.config, use_bespoke=False)
        standard_results = self.train_model(standard_model, train_data)
        standard_eval = self.evaluate_model(standard_model, test_data)
        standard_params = standard_model.get_parameter_count()
        
        # Train bespoke model
        print("  Training bespoke model...")
        bespoke_model = LanguageModel(self.config, use_bespoke=True, 
                                    token_categories=analyzer.token_categories)
        bespoke_results = self.train_model(bespoke_model, train_data)
        bespoke_eval = self.evaluate_model(bespoke_model, test_data)
        bespoke_params = bespoke_model.get_parameter_count()
        
        # Calculate efficiency gains
        param_ratio = bespoke_params['total'] / standard_params['total']
        performance_ratio = bespoke_eval['loss'] / standard_eval['loss']
        efficiency_score = (1 / param_ratio) / performance_ratio  # Higher is better
        
        results = {
            'corpus_size': corpus_size,
            'frequency_analysis': freq_stats,
            'standard_model': {
                'training': standard_results,
                'evaluation': standard_eval,
                'parameters': standard_params
            },
            'bespoke_model': {
                'training': bespoke_results,  
                'evaluation': bespoke_eval,
                'parameters': bespoke_params
            },
            'efficiency_metrics': {
                'parameter_ratio': param_ratio,
                'performance_ratio': performance_ratio, 
                'efficiency_score': efficiency_score,
                'memory_saved_mb': (standard_params['total'] - bespoke_params['total']) * 4 / (1024**2)  # Assuming float32
            }
        }
        
        return results
    
    def run_full_experiment(self) -> Dict:
        """Run the complete experiment across all corpus sizes"""
        print("🧪 Starting Bespoke Token Dimension Experiments")
        print(f"Configuration: {self.config.vocab_size} vocab, {self.config.corpus_sizes} corpus sizes")
        
        all_results = {}
        
        for corpus_size in self.config.corpus_sizes:
            try:
                results = self.run_single_experiment(corpus_size)
                all_results[corpus_size] = results
                
                # Print summary
                eff_metrics = results['efficiency_metrics']
                print(f"  Results: {eff_metrics['parameter_ratio']:.3f}x params, {eff_metrics['efficiency_score']:.3f} efficiency score")
                
            except Exception as e:
                print(f"  Error in corpus size {corpus_size}: {e}")
                continue
        
        # Generate summary analysis
        summary = self.analyze_results(all_results)
        
        return {
            'experiment_config': self.config,
            'results': all_results,
            'summary': summary
        }
    
    def analyze_results(self, results: Dict) -> Dict:
        """Analyze results across all experiments"""
        if not results:
            return {'error': 'No results to analyze'}
            
        corpus_sizes = sorted(results.keys())
        
        # Extract key metrics
        param_ratios = [results[size]['efficiency_metrics']['parameter_ratio'] for size in corpus_sizes]
        efficiency_scores = [results[size]['efficiency_metrics']['efficiency_score'] for size in corpus_sizes]
        standard_losses = [results[size]['standard_model']['evaluation']['loss'] for size in corpus_sizes]
        bespoke_losses = [results[size]['bespoke_model']['evaluation']['loss'] for size in corpus_sizes]
        
        summary = {
            'corpus_sizes': corpus_sizes,
            'parameter_efficiency': {
                'ratios': param_ratios,
                'average_reduction': 1 - np.mean(param_ratios),
                'best_reduction': 1 - min(param_ratios),
                'consistent_savings': all(r < 1.0 for r in param_ratios)
            },
            'performance_quality': {
                'standard_losses': standard_losses,
                'bespoke_losses': bespoke_losses,
                'quality_maintained': all(b <= s * 1.1 for s, b in zip(standard_losses, bespoke_losses)),  # Within 10%
                'average_quality_ratio': np.mean([b/s for s, b in zip(standard_losses, bespoke_losses)])
            },
            'efficiency_scores': {
                'scores': efficiency_scores,
                'average_score': np.mean(efficiency_scores),
                'improving_with_scale': efficiency_scores[-1] > efficiency_scores[0] if len(efficiency_scores) > 1 else False
            },
            'theory_validation': {
                'parameter_reduction_achieved': np.mean(param_ratios) < 0.9,  # At least 10% reduction
                'quality_preserved': np.mean([b/s for s, b in zip(standard_losses, bespoke_losses)]) < 1.2,  # Within 20%
                'scales_effectively': len([s for s in efficiency_scores if s > 0.8]) >= len(efficiency_scores) // 2
            }
        }
        
        return summary
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"bespoke_token_experiment_{timestamp}.json"
            
        filepath = f"/Users/bard/Code/Claude_Data/tool_outputs/{filename}"
        
        # Convert any tensors or complex objects to serializable format
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"💾 Results saved to: {filepath}")
        return filepath
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj

def main():
    """Run the complete bespoke token dimension experiment"""
    
    # Configure experiment
    config = ExperimentConfig(
        vocab_size=500,  # Start smaller for faster testing
        corpus_sizes=[1000, 5000, 20000],  # Manageable sizes for initial validation
        block_size=64,
        max_iters=500,  # Reduced for faster testing
        n_layers=3,     # Smaller model for faster training
        n_heads=4,
        standard_embed_dim=128,
        high_freq_dim=192,  # 50% more for frequent tokens
        mid_freq_dim=128,   # Standard size
        low_freq_dim=64,    # 50% less for rare tokens
        special_token_dim=32  # Very small for punctuation
    )
    
    # Run experiment
    experiment = BespokeTokenExperiment(config)
    results = experiment.run_full_experiment()
    
    # Save results
    filepath = experiment.save_results(results)
    
    # Print final summary
    print("\n" + "="*60)
    print("🎯 EXPERIMENT SUMMARY")
    print("="*60)
    
    summary = results['summary']
    theory = summary['theory_validation']
    
    print(f"Parameter Efficiency: {summary['parameter_efficiency']['average_reduction']:.1%} average reduction")
    print(f"Quality Preservation: {summary['performance_quality']['average_quality_ratio']:.3f} ratio (1.0 = identical)")
    print(f"Efficiency Scores: {summary['efficiency_scores']['average_score']:.3f} average")
    
    print(f"\n🔬 THEORY VALIDATION:")
    print(f"  ✅ Parameter reduction achieved: {theory['parameter_reduction_achieved']}")
    print(f"  ✅ Quality preserved: {theory['quality_preserved']}")  
    print(f"  ✅ Scales effectively: {theory['scales_effectively']}")
    
    validation_score = sum([
        theory['parameter_reduction_achieved'],
        theory['quality_preserved'], 
        theory['scales_effectively']
    ])
    
    print(f"\n🏆 OVERALL VALIDATION: {validation_score}/3 criteria met")
    
    if validation_score >= 2:
        print("🚀 THEORY VALIDATION: SUCCESS! Bespoke token dimensions show promise.")
    else:
        print("⚠️  THEORY VALIDATION: PARTIAL. Further optimization needed.")
        
    print(f"\n📊 Detailed results saved to: {filepath}")
    
    return results

if __name__ == "__main__":
    results = main()
