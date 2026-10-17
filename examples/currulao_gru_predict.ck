// currulao_gru_predict.ck
// Minimal Chonnx example: generates a sequence of currulao marimba phrases
// with CurrulaoGRU and prints it to the console (no audio).
//
// The GRU predicts the next phrase from the last 4 phrases and their
// harmonic states (0 = tonic, 1 = dominant), which alternate every phrase.
// For the full musical version (marimba + bombo + cununo) see
// ai/CurrulaoGRU.ck in the Live-Coding-Colombian-Music repository.

@import "../Chonnx.chug"

// === PARAMETERS ===
16  => int   N_PHRASES_TO_GENERATE;
0   => int   state;   // initial harmonic state: 0 = tonic, 1 = dominant
1.0 => float TEMP;    // temperature: 1.0 = model as trained, >1 = more variety
// ==================

Chonnx brain;
brain.load_model(me.dir() + "../models/currulao_gru.onnx");

["A","B","C","D","E","F","G","H","I","J"] @=> string names[];

// History of the last 4 phrases (oldest first), initialized with G
[6, 6, 6, 6] @=> int history[];

// Builds the 8 inputs (4 timesteps x [phrase/9.0, state]), runs the model
// and samples the next phrase from the 10 output probabilities.
fun int predictNext(int currentState) {
    // States alternate: t-4 and t-2 share the current state, t-3 and t-1 the opposite
    [currentState, 1 - currentState, currentState, 1 - currentState] @=> int st[];

    for (0 => int i; i < 4; i++) {
        brain.set_input(i * 2,     history[i] / 9.0);
        brain.set_input(i * 2 + 1, st[i] * 1.0);
    }
    brain.run();

    // Temperature scaling p^(1/T), then proportional sampling
    float acc[10];
    0.0 => float total;
    for (0 => int i; i < 10; i++) {
        Math.pow(brain.get_output(i), 1.0 / TEMP) +=> total;
        total => acc[i];
    }
    Math.random2f(0.0, total) => float r;
    for (0 => int i; i < 10; i++) {
        if (r <= acc[i]) return i;
    }
    return 6;  // fallback: G
}

fun void pushHistory(int phrase) {
    for (0 => int i; i < 3; i++) history[i + 1] => history[i];
    phrase => history[3];
}

<<< "=== CurrulaoGRU via Chonnx ===" >>>;
for (0 => int n; n < N_PHRASES_TO_GENERATE; n++) {
    predictNext(state) => int next;
    <<< (state == 0 ? "T" : "D"), "[",
        names[history[0]], names[history[1]], names[history[2]], names[history[3]],
        "] ->", names[next] >>>;
    pushHistory(next);
    1 - state => state;
}
