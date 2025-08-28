# run_mps_topk_mlp.py
# End-to-end: tiny GPT on random tokens -> collect attention teacher -> train TopK selector + MLP approximator
# Works on M1/M2 via PyTorch MPS. No nanoGPT edits required.

import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Device (MPS first)
# ----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"[info] using device: {device}")

# Repro
torch.manual_seed(1337)

# ----------------------------
# Tiny dataset (random tokens)
# ----------------------------
# NOTE: We use a synthetic dataset to keep this fast on a Mac mini.
# You can later swap this with a real corpus/tokenizer.
@dataclass
class DataCfg:
    vocab_size: int = 64
    block_size: int = 64
    train_tokens: int = 50_000  # total tokens
    batch_size: int = 32

cfg_data = DataCfg()
data = torch.randint(0, cfg_data.vocab_size, (cfg_data.train_tokens,), dtype=torch.long)
def get_batch(split="train"):
    ix = torch.randint(0, len(data) - cfg_data.block_size - 1, (cfg_data.batch_size,))
    x = torch.stack([data[i:i+cfg_data.block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+cfg_data.block_size] for i in ix])
    return x.to(device), y.to(device)

# ----------------------------
# Model config (tiny GPT-ish)
# ----------------------------
@dataclass
class GPTCfg:
    vocab_size: int = cfg_data.vocab_size
    block_size: int = cfg_data.block_size
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    dropout: float = 0.0

cfg = GPTCfg()

# ----------------------------
# Tiny GPT building blocks
# (compatible shapes with nanoGPT style)
# ----------------------------
class CausalSelfAttention(nn.Module):
    """Full attention (teacher). We will log Q,K,V & outputs from the *first* layer."""
    def __init__(self, n_embd, n_head, block_size, dropout=0.0, log_layer=False, top_k=3, thresh=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.block_size = block_size
        self.log_layer = log_layer  # only True for layer 0 when collecting teacher signals
        self.top_k = top_k
        self.thresh = thresh

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # project to Q,K,V
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

        # Buffers for logging (filled during forward if log_layer=True)
        self.logged = {
            "Q": [], "K": [], "V": [],
            "attn_probs": [], "topk_idx": [],
            "out": []
        }

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)                               # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)            # each (B, T, C)
        # shape into heads
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, Hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # attention scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)    # (B, H, T, T)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                                                   # (B, H, T, Hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)              # (B, T, C)
        y = self.resid_drop(self.c_proj(y))                           # (B, T, C)

        # Optional logging for teacher collection (layer 0)
        if self.log_layer and self.training is False:
            with torch.no_grad():
                # Threshold then top-k on attention for *salient keys* (per query)
                att_use = att.clone()
                if self.thresh > 0:
                    att_use = att_use * (att_use > self.thresh)
                    att_use = att_use / (att_use.sum(dim=-1, keepdim=True) + 1e-8)
                top_vals, top_idx = torch.topk(att_use, k=min(self.top_k, att_use.size(-1)), dim=-1)  # (B,H,T,K)

                # Log teacher signals (detach to free graph)
                self.logged["Q"].append(q.detach().cpu())
                self.logged["K"].append(k.detach().cpu())
                self.logged["V"].append(v.detach().cpu())
                self.logged["attn_probs"].append(att_use.detach().cpu())
                self.logged["topk_idx"].append(top_idx.detach().cpu())
                self.logged["out"].append(y.detach().cpu())  # attention output pre-residual next block

        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg: GPTCfg, layer_idx: int = 0, log_first_layer: bool = False, top_k=3, thresh=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg.n_embd, cfg.n_head, cfg.block_size,
                                        cfg.dropout, log_layer=(log_first_layer and layer_idx==0),
                                        top_k=top_k, thresh=thresh)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg.n_embd, cfg.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTCfg, log_first_layer=False, top_k=3, thresh=0.0):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.n_embd))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i, log_first_layer, top_k, thresh) for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok = self.tok_emb(idx)                        # (B,T,C)
        x = self.drop(tok + self.pos_emb[:, :T, :])   # (B,T,C)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ----------------------------
# Train baseline (full attention)
# ----------------------------
lr = 3e-3
max_steps = 300   # keep small for Mac mini
eval_every = 100

model = TinyGPT(cfg, log_first_layer=False).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)

print("[info] training baseline (full attention)...")
model.train()
for step in range(1, max_steps+1):
    x, y = get_batch()
    logits, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % eval_every == 0:
        print(f"  step {step:4d} | loss {loss.item():.4f}")

# ----------------------------
# Collect teacher signals from first layer (eval mode)
# ----------------------------
print("[info] collecting teacher signals (Q,K,V, top-k indices, attention outputs) from layer 0...")
# re-create model with logging on first layer; copy weights from trained model
collector = TinyGPT(cfg, log_first_layer=True, top_k=3, thresh=0.0).to(device)
collector.load_state_dict(model.state_dict())  # same weights
collector.eval()

with torch.no_grad():
    for _ in range(32):  # ~32 batches is enough for small selector/MLP
        xb, yb = get_batch()
        _ = collector(xb)  # triggers logging inside CausalSelfAttention of layer 0

# stitch logged tensors
log = collector.blocks[0].attn.logged
def cat_list(key):
    return torch.cat(log[key], dim=0) if len(log[key])>0 else None

Q = cat_list("Q")          # (N, H, T, Hs)
K = cat_list("K")
V = cat_list("V")
ATTN = cat_list("attn_probs")  # (N, H, T, T)
TOPK = cat_list("topk_idx")    # (N, H, T, K)
Y_out = cat_list("out")        # (N, T, C)

print({k: (cat_list(k).shape if cat_list(k) is not None else None) for k in ["Q","K","V","attn_probs","topk_idx","out"]})

# We'll train selector & approximator on the *first head* of *first layer* for clarity
# You can extend to all heads later.
head = 0
Qh = Q[:, head]            # (N, T, Hs)
Kh = K[:, head]            # (N, T, Hs)
Vh = V[:, head]            # (N, T, Hs)
ATTNh = ATTN[:, head]      # (N, T, T)
TOPKh = TOPK[:, head]      # (N, T, K)

N, T, Hs = Qh.shape
C = cfg.n_embd
Ksel = TOPKh.size(-1)

# ----------------------------
# Train Top-K KeySelector
# Input: query vector (Hs)
# Output: logits over keys [T], trained to match teacher top-1 (can extend to multi-label)
# ----------------------------
class KeySelector(nn.Module):
    def __init__(self, d_in, seq_len, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, seq_len)
        )
    def forward(self, q_flat):
        return self.net(q_flat)  # (N*T, T)

selector = KeySelector(Hs, T, hidden=64).to(device)
opt_sel = torch.optim.AdamW(selector.parameters(), lr=3e-3)

# Flatten queries and targets
Q_flat = Qh.reshape(-1, Hs).to(device)         # (N*T, Hs)
# For simplicity, train to predict the first teacher key (strongest)
y_top1 = TOPKh[..., 0].reshape(-1).to(device)  # (N*T,)

print("[info] training KeySelector...")
selector.train()
for it in range(400):
    logits = selector(Q_flat)              # (N*T, T)
    loss = F.cross_entropy(logits, y_top1) # predict top-1 index
    opt_sel.zero_grad(set_to_none=True)
    loss.backward()
    opt_sel.step()
    if (it+1) % 100 == 0:
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == y_top1).float().mean().item()
        print(f"  selector it {it+1:4d} | loss {loss.item():.4f} | acc {acc:.3f}")

# ----------------------------
# Build sparse selector (no Q·Kᵀ at inference)
# Given Q, predict top-k indices
# ----------------------------
def predict_topk_indices(q_batch, k=Ksel):
    # q_batch: (B, T, Hs)
    B, Tq, Hs = q_batch.shape
    logits = selector(q_batch.reshape(-1, Hs))       # (B*Tq, T)
    # choose top-k per query
    top_vals, top_idx = torch.topk(logits, k=min(k, Tq), dim=-1)
    return top_idx.view(B, Tq, -1)  # (B, Tq, k)

# ----------------------------
# Train MLP Approximator (small; smaller than attention head width)
# Input: concatenated top-k V vectors for each query -> output: head output (Hs) per query
# ----------------------------
class HeadApproximator(nn.Module):
    def __init__(self, d_in, d_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out)
        )
    def forward(self, x):
        return self.net(x)

# Prepare teacher pairs for approximator
# For each (batch, query_pos), build input by concatenating teacher-selected V vectors (Hs*Ksel)
with torch.no_grad():
    # teacher input (concat of V at teacher TOPKh)
    TOP_flat = TOPKh.reshape(-1, Ksel)     # (N*T, K)
    Vh_expanded = Vh.unsqueeze(2).expand(N, T, T, Hs)  # (N,T,T,Hs)
    gather_idx = TOP_flat.unsqueeze(-1).expand(-1, -1, Hs)  # (N*T, K, Hs)
    V_concat = torch.gather(Vh_expanded.reshape(-1, T, Hs), 1, gather_idx).reshape(-1, Ksel*Hs)  # (N*T, K*Hs)

    # teacher output per head: from Y_out, we need the portion that corresponds to this head before c_proj.
    # Simpler proxy: reconstruct per-head output via (ATTNh @ Vh). That matches head output pre-concat.
    head_out_teacher = (ATTNh @ Vh).reshape(-1, Hs)  # (N*T, Hs)

V_concat = V_concat.to(device)
head_out_teacher = head_out_teacher.to(device)

approx = HeadApproximator(d_in=Ksel*Hs, d_out=Hs, hidden=64).to(device)
opt_apx = torch.optim.AdamW(approx.parameters(), lr=3e-3)

print("[info] training HeadApproximator (small MLP over top-k V)...")
approx.train()
for it in range(600):
    pred = approx(V_concat)
    loss = F.mse_loss(pred, head_out_teacher)
    opt_apx.zero_grad(set_to_none=True)
    loss.backward()
    opt_apx.step()
    if (it+1) % 150 == 0:
        with torch.no_grad():
            rel = (pred - head_out_teacher).pow(2).sum().sqrt() / (head_out_teacher.norm() + 1e-9)
        print(f"  approx it {it+1:4d} | mse {loss.item():.6f} | rel_err {rel.item():.4f}")

# ----------------------------
# Build a Drop-in Sparse+Approx Attention for layer 0, head 0
# It:
#  - uses KeySelector to pick indices (no Q·Kᵀ)
#  - gathers top-k V
#  - feeds concat(V_k) to approximator to produce head output
#  - keeps other heads exact (optional: you can extend to all heads)
# ----------------------------
class HybridCausalSelfAttention(nn.Module):
    def __init__(self, teacher_attn: CausalSelfAttention, use_heads="first_only", selector=None, approx=None, ksel=Ksel):
        super().__init__()
        # copy teacher projections for K,V (we still need V vectors)
        self.c_attn = teacher_attn.c_attn
        self.c_proj = teacher_attn.c_proj
        self.n_embd = teacher_attn.n_embd
        self.n_head = teacher_attn.n_head
        self.head_size = teacher_attn.head_size
        self.block_size = teacher_attn.block_size
        self.resid_drop = teacher_attn.resid_drop
        self.ln_dummy = nn.Identity()
        self.selector = selector
        self.approx = approx
        self.ksel = ksel
        # causal mask
        self.register_buffer("mask", teacher_attn.mask)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B,H,T,Hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Exact attention for all heads except head 0 (to keep demo simple)
        # Head 0: use learned selector + approximator
        # Heads 1..H-1: compute normally
        # Build output tensor per head
        out_heads = torch.empty(B, self.n_head, T, self.head_size, device=x.device)

        # ---- head 0 via selector+approx (no Q·Kᵀ) ----
        q0 = q[:, 0]        # (B, T, Hs)
        v0 = v[:, 0]        # (B, T, Hs)
        # predict indices (B,T,k)
        idx0 = predict_topk_indices(q0)  # uses trained selector on device
        # gather values -> (B,T,k,Hs)
        gather_idx = idx0.unsqueeze(-1).expand(-1, -1, -1, self.head_size)
        v0_exp = v0.unsqueeze(1).expand(B, T, T, self.head_size)
        v_sel = torch.gather(v0_exp, 2, gather_idx)  # (B,T,k,Hs)
        v_cat = v_sel.reshape(B*T, self.ksel*self.head_size)  # (B*T, k*Hs)
        # MLP approximates head output per query (B*T, Hs) -> reshape back
        h0 = self.approx(v_cat).reshape(B, T, self.head_size)
        out_heads[:, 0] = h0

        # ---- remaining heads exact ----
        if self.n_head > 1:
            q_rest = q[:, 1:]                 # (B,H-1,T,Hs)
            k_rest = k[:, 1:]
            v_rest = v[:, 1:]
            att = (q_rest @ k_rest.transpose(-2, -1)) / math.sqrt(self.head_size)
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y_rest = att @ v_rest             # (B,H-1,T,Hs)
            out_heads[:, 1:] = y_rest

        y = out_heads.transpose(1, 2).contiguous().view(B, T, C)  # concat heads
        y = self.resid_drop(self.c_proj(y))
        return y

# Swap layer-0 attention
model.eval()
hybrid = TinyGPT(cfg, log_first_layer=False).to(device)
hybrid.load_state_dict(model.state_dict())
# replace block 0 attn with hybrid module
hybrid.blocks[0].attn = HybridCausalSelfAttention(
    teacher_attn=hybrid.blocks[0].attn,
    selector=selector,
    approx=approx,
    ksel=Ksel
).to(device)
hybrid.eval()

# ----------------------------
# Evaluate approximation error
# ----------------------------
@torch.no_grad()
def eval_rel_error(num_batches=10):
    model.eval(); hybrid.eval()
    rels = []
    for _ in range(num_batches):
        xb, _ = get_batch()
        y1, _ = model(xb)     # logits
        y2, _ = hybrid(xb)    # logits with head-0 approximated
        # compare hidden (pre-logits) or logits. We'll compare final logits.
        r = (y1 - y2).pow(2).sum().sqrt() / (y1.norm() + 1e-9)
        rels.append(r.item())
    return sum(rels)/len(rels)

err = eval_rel_error()
print(f"[result] relative error (final logits, hybrid vs teacher): {err:.4f}")

# Also show top-1 selector accuracy on a fresh batch of queries
with torch.no_grad():
    xb, _ = get_batch()
    # recompute Q for first layer to check selector quality
    probe = TinyGPT(cfg, log_first_layer=True, top_k=3, thresh=0.0).to(device)
    probe.load_state_dict(model.state_dict())
    probe.eval()
    _ = probe(xb)  # logs for one batch
    Qp = probe.blocks[0].attn.logged["Q"][-1][:, 0]      # (B,T,Hs) head 0
    top1_true = probe.blocks[0].attn.logged["topk_idx"][-1][:, 0, :, 0]  # (B,T)
    logits_sel = selector(Qp.reshape(-1, Qp.size(-1)))
    top1_pred = logits_sel.argmax(dim=-1).view_as(top1_true)
    acc = (top1_pred.to(top1_true.device) == top1_true).float().mean().item()
    print(f"[result] key selector top-1 accuracy (head 0, layer 0): {acc:.3f}")

