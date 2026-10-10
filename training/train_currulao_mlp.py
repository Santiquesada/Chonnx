"""
train_currulao_mlp.py
=====================
Trains an MLP (CurrulaoBrain) on real currulao transcriptions
(Baudilio Cuama and Marino Beltran, book "Arrullos y Currulaos").

The model learns:
    P(next_phrase | harmonic_state, previous_phrase)

This is something ChucK's HMM cannot do, since it only knows:
    P(phrase | harmonic_state)

Exports: currulao.onnx  ->  loadable with Chonnx in ChucK
"""

import torch
import torch.nn as nn
import numpy as np
import onnxruntime as rt

# =============================================================================
# 1. PHRASE VOCABULARY
#    (reconstructed from Baudilio Cuama and Marino Beltran)
# =============================================================================
# Index -> name -> rhythmic pattern (own notation: 0=rest, 1=hit, -1=sixteenth-note toggle)
PHRASES = {
    0: ("A", [1,0,0,1,0,0]),
    1: ("B", [1,0,0,1,0,1]),
    2: ("C", [1,0,1,1,0,1]),
    3: ("D", [1,1,1,1,1,1]),
    4: ("E", [-1,0,1,-1,0,1,1,0,1]),
    5: ("F", [-1,0,1,-1,0,1,1,1,1]),
    6: ("G", [-1,0,1,-1,1,1,1,0,1]),   # dominant phrase of the corpus
    7: ("H", [-1,0,1,-1,1,1,1,1,1]),
    8: ("I", [-1,0,1,-1,0,1,-1,1,1,-1,0,1]),
    9: ("J", [-1,0,1,-1,0,1,-1,0,1,-1,0,1]),  # climax, maximum density
}
N_PHRASES = len(PHRASES)

# Harmonic state: 0 = tonic, 1 = dominant

# =============================================================================
# 2. TRAINING DATA
#    Each tuple: (current_state, previous_phrase, next_phrase)
#    Source: real sequences from the two transcriptions
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

def sequence_to_examples(seq):
    """Turns a sequence of (state, phrase) into training triples."""
    examples = []
    for i in range(1, len(seq)):
        current_state   = seq[i][0]
        current_phrase  = seq[i][1]
        previous_phrase = seq[i-1][1]
        examples.append((current_state, previous_phrase, current_phrase))
    return examples

data = sequence_to_examples(seq_baudilio) + sequence_to_examples(seq_marino)

print(f"Training examples: {len(data)}")
print(f"  Baudilio: {len(seq_baudilio)-1} | Marino: {len(seq_marino)-1}")

# Tensors
X = torch.tensor(
    [[state, prev_phrase / (N_PHRASES - 1.0)] for state, prev_phrase, _ in data],
    dtype=torch.float32
)
y = torch.tensor([next_phrase for _, _, next_phrase in data], dtype=torch.long)

# =============================================================================
# 3. MODEL -- tiny MLP: 2 -> 16 -> 10
# =============================================================================
class CurrulaoBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, N_PHRASES),
        )

    def forward(self, x):
        return self.net(x)   # raw logits

model = CurrulaoBrain()

# =============================================================================
# 4. TRAINING
# =============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

print("\nTraining CurrulaoBrain...")
model.train()
for epoch in range(500):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1:4d}  Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "currulao_weights.pth")
print("Weights saved: currulao_weights.pth")

# =============================================================================
# 5. QUICK VALIDATION -- what does the model predict?
# =============================================================================
model.eval()
print("\nValidation -- model predictions:")
print(f"{'State':<10} {'Previous phrase':<18} {'Top-3 predictions'}")
print("-" * 60)
test_cases = [
    (0, 6, "T, prev=G"),
    (1, 6, "D, prev=G"),
    (1, 4, "D, prev=E  (predicts J?)"),
    (0, 9, "T, prev=J  (predicts E?)"),
    (1, 7, "D, prev=H  (predicts H?)"),
]
with torch.no_grad():
    for state, prev, desc in test_cases:
        x = torch.tensor([[state, prev / (N_PHRASES - 1.0)]])
        probs = torch.softmax(model(x), dim=-1)[0]
        top3 = torch.topk(probs, 3)
        top3_str = "  ".join(
            f"{PHRASES[i.item()][0]}({p.item():.2f})"
            for i, p in zip(top3.indices, top3.values)
        )
        print(f"  {desc:<30} -> {top3_str}")

# =============================================================================
# 6. EXPORT TO ONNX
# =============================================================================
class CurrulaoWrapper(nn.Module):
    """Export wrapper: takes [state, prev_norm], returns probabilities."""
    def __init__(self, base):
        super().__init__()
        self.model = base

    def forward(self, x):   # x: [1, 2]
        return torch.softmax(self.model(x), dim=-1)

wrapped = CurrulaoWrapper(model)
wrapped.eval()

dummy = torch.zeros(1, 2, dtype=torch.float32)
output_path = "currulao.onnx"

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
dummy_np = np.zeros((1, 2), dtype=np.float32)
result = sess.run(None, {"context": dummy_np})
print(f"Output shape: {result[0].shape}  (expected: (1, {N_PHRASES}))")
print(f"Probs sum: {result[0][0].sum():.4f}  (expected: 1.0)")

# =============================================================================
# 7. CHUCK REFERENCE
# =============================================================================
print("""
=== USAGE IN CHUCK ===
Chonnx brain;
brain.load_model(me.dir() + "../models/currulao.onnx");

// set_input(0, state)            0.0 = tonic   1.0 = dominant
// set_input(1, prev_phrase/9.0)  previous phrase, normalized
// brain.run();
// get_output(0..9)               probability of each phrase

// Phrase indices:
// 0=A  1=B  2=C  3=D  4=E  5=F  6=G  7=H  8=I  9=J
""")
