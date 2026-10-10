"""
train_currulao_gru.py
=====================
Trains a GRU on real currulao transcriptions
(Baudilio Cuama and Marino Beltran, book "Arrullos y Currulaos").

The model learns:
    P(next_phrase | last 4 phrases and their harmonic states)

The GRU keeps a hidden state that accumulates the contour of previous
phrases -- something neither the HMM nor the MLP (CurrulaoBrain) can do.

Key difference vs the MLP:
  - The MLP only sees [current_state, previous_phrase]
  - The GRU processes a 4-step sequence and "remembers" the arc
    G->G->G->G vs B->F->G->H, even when the last step is the same

Exports: currulao_gru.onnx  ->  loadable with Chonnx in ChucK

ONNX input:  (1, 8) flat = 4 timesteps x [phrase/9.0, state]
ONNX output: (1, 10)     = probabilities of the next phrase
"""

import torch
import torch.nn as nn
import numpy as np
import onnxruntime as rt

# =============================================================================
# 1. VOCABULARY
# =============================================================================
N_PHRASES = 10
WINDOW    = 4   # how many context phrases the GRU uses

# 0=A  1=B  2=C  3=D  4=E  5=F  6=G  7=H  8=I  9=J
NAMES = ["A","B","C","D","E","F","G","H","I","J"]

# =============================================================================
# 2. CORPUS
#    Each tuple: (harmonic_state, phrase)   state: 0 = tonic, 1 = dominant
# =============================================================================

# Baudilio Cuama (starts on dominant = 1)
seq_baudilio = [
    (1, 1),   # C1:  B
    (0, 5),   # C2:  F
    (1, 6),   # C3:  G
    (0, 4),   # C4:  E
    (1, 9),   # C5:  J  (climax)
    (0, 4),   # C6:  E
    (1, 8),   # C7:  I
    (0, 5),   # C8:  F
    (1, 5),   # C9:  F
    (0, 6),   # C10: G
    (1, 6),   # C11: G
    (0, 6),   # C12: G
    (1, 6),   # C13: G
    (0, 6),   # C14: G
    (1, 6),   # C15: G
    (0, 6),   # C16: G
    (1, 6),   # C17: G
]

# Marino Beltran (starts on tonic = 0)
seq_marino = [
    (0, 0),   # C1:  A
    (1, 1),   # C2:  B
    (0, 6),   # C3:  G
    (1, 7),   # C4:  H
    (0, 6),   # C5:  G
    (1, 7),   # C6:  H
    (0, 6),   # C7:  G
    (1, 3),   # C8:  D
    (0, 6),   # C9:  G
    (1, 3),   # C10: D
    (0, 7),   # C11: H
    (1, 7),   # C12: H
    (0, 6),   # C13: G
    (1, 2),   # C14: C
    (0, 6),   # C15: G
    (1, 6),   # C16: G
    (0, 6),   # C17: G
    (1, 6),   # C18: G
    (0, 6),   # C19: G
]

# =============================================================================
# 3. BUILD WINDOWS
#    Each example: context of WINDOW steps -> next phrase
# =============================================================================
def sequence_to_windows(seq, window):
    examples = []
    for i in range(window, len(seq)):
        context = seq[i - window : i]   # list of (state, phrase), oldest first
        target_phrase = seq[i][1]
        examples.append((context, target_phrase))
    return examples

data = (sequence_to_windows(seq_baudilio, WINDOW) +
        sequence_to_windows(seq_marino,   WINDOW))

print(f"Training examples: {len(data)}")
print(f"  Baudilio: {len(seq_baudilio)-WINDOW} | Marino: {len(seq_marino)-WINDOW}")

# Tensors: flat input (N, WINDOW*2), target (N,)
# Per-timestep order (oldest first): [phrase/9.0, state, phrase/9.0, state, ...]
X_list = []
y_list = []
for context, target_phrase in data:
    feats = []
    for state, phrase in context:
        feats.append(phrase / (N_PHRASES - 1.0))
        feats.append(float(state))
    X_list.append(feats)
    y_list.append(target_phrase)

X = torch.tensor(X_list, dtype=torch.float32)   # (N, 8)
y = torch.tensor(y_list, dtype=torch.long)       # (N,)

print(f"X shape: {X.shape}  y shape: {y.shape}")

# =============================================================================
# 4. MODEL -- GRU: 2 -> hidden=32 -> 10
# =============================================================================
HIDDEN = 32

class CurrulaoGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=HIDDEN,
                          num_layers=1, batch_first=True)
        self.fc  = nn.Linear(HIDDEN, N_PHRASES)

    def forward(self, x_flat):           # x_flat: (batch, WINDOW*2)
        x = x_flat.view(-1, WINDOW, 2)  # (batch, 4, 2)
        out, _ = self.gru(x)            # out: (batch, 4, HIDDEN)
        return self.fc(out[:, -1, :])   # logits of the last step: (batch, 10)

model = CurrulaoGRU()
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params}")

# =============================================================================
# 5. TRAINING
# =============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)

print("\nTraining CurrulaoGRU...")
model.train()
for epoch in range(2000):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 400 == 0:
        print(f"  Epoch {epoch+1:5d}  Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "currulao_gru_weights.pth")
print("Weights saved: currulao_gru_weights.pth")

# =============================================================================
# 6. QUICK VALIDATION
# =============================================================================
model.eval()
print("\nValidation -- model predictions (4-phrase window):")
print(f"{'Context':<35} {'Top-3 predictions'}")
print("-" * 65)

test_cases = [
    # (context: list of (state, phrase), description)   T = tonic, D = dominant
    ([(1,6),(0,6),(1,6),(0,6)], "D-G T-G D-G T-G  (steady G)"),
    ([(0,6),(1,7),(0,6),(1,7)], "T-G D-H T-G D-H  (G/H alternation)"),
    ([(1,1),(0,5),(1,6),(0,4)], "D-B T-F D-G T-E  (Baudilio opening)"),
    ([(0,0),(1,1),(0,6),(1,7)], "T-A D-B T-G D-H  (Marino opening)"),
    ([(0,4),(1,9),(0,4),(1,8)], "T-E D-J T-E D-I  (Baudilio climax)"),
]

with torch.no_grad():
    for context, desc in test_cases:
        feats = []
        for state, phrase in context:
            feats.append(phrase / (N_PHRASES - 1.0))
            feats.append(float(state))
        x = torch.tensor([feats], dtype=torch.float32)
        probs = torch.softmax(model(x), dim=-1)[0]
        top3 = torch.topk(probs, 3)
        top3_str = "  ".join(
            f"{NAMES[i.item()]}({p.item():.2f})"
            for i, p in zip(top3.indices, top3.values)
        )
        print(f"  {desc:<35} -> {top3_str}")

# =============================================================================
# 7. EXPORT TO ONNX
# =============================================================================
class GRUWrapper(nn.Module):
    """Export wrapper: returns probabilities (not logits)."""
    def __init__(self, base):
        super().__init__()
        self.model = base

    def forward(self, x):   # x: (1, 8)
        return torch.softmax(self.model(x), dim=-1)

wrapped = GRUWrapper(model)
wrapped.eval()

dummy = torch.zeros(1, WINDOW * 2, dtype=torch.float32)
output_path = "currulao_gru.onnx"

torch.onnx.export(
    wrapped, dummy, output_path,
    input_names=["context"],
    output_names=["phrase_probs"],
    opset_version=17,
    external_data=False,
)
print(f"\nExported: {output_path}")

# Validate with onnxruntime
sess = rt.InferenceSession(output_path)
dummy_np = np.zeros((1, WINDOW * 2), dtype=np.float32)
result = sess.run(None, {"context": dummy_np})
print(f"Output shape: {result[0].shape}  (expected: (1, {N_PHRASES}))")
print(f"Probs sum:    {result[0][0].sum():.4f}  (expected: 1.0)")

# =============================================================================
# 8. CHUCK REFERENCE
# =============================================================================
print("""
=== USAGE IN CHUCK ===

Chonnx brain;
brain.load_model(me.dir() + "../models/currulao_gru.onnx");

// Input: 8 values = 4 timesteps x [phrase/9.0, state]
// Order: oldest first
//   set_input(0, history[0] / 9.0)   phrase t-4, normalized
//   set_input(1, state_t4 * 1.0)     state at t-4
//   set_input(2, history[1] / 9.0)   phrase t-3, normalized
//   set_input(3, state_t3 * 1.0)     state at t-3
//   set_input(4, history[2] / 9.0)   phrase t-2, normalized
//   set_input(5, state_t2 * 1.0)     state at t-2
//   set_input(6, history[3] / 9.0)   phrase t-1, normalized
//   set_input(7, state_t1 * 1.0)     state at t-1
//   brain.run();
//   get_output(0..9) -> probabilities of the next phrase

// Since the state always alternates, if the CURRENT state is E:
//   t-4 -> E   (4 steps back, even -> same state)
//   t-3 -> E^1
//   t-2 -> E
//   t-1 -> E^1
""")
