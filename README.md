# Chonnx

**Generic ONNX inference engine for [ChucK](https://chuck.stanford.edu/).**

Chonnx is a chugin (a ChucK plugin written in C++) that loads any float32 `.onnx` model
and runs it in real time from ChucK code, using [ONNX Runtime](https://onnxruntime.ai/).
It knows nothing about the model it loads: it inspects the model's inputs and outputs
when loading, receives numbers from ChucK, runs the network and returns numbers.
Musical logic (sampling, sequencing, synthesis) stays in ChucK.

The goal is infrastructure: a bridge between models trained offline in any framework
(PyTorch, TensorFlow, etc., exported to ONNX) and live performance in ChucK, without
writing a new chugin for each architecture or routing through OSC to another program.

## API

```chuck
Chonnx brain;
brain.load_model(me.dir() + "model.onnx");   // inspects inputs/outputs and allocates buffers

// Single input / single output models
brain.set_input(index, value);               // write into input tensor 0
brain.run();                                 // returns 1 on success, 0 on failure
brain.get_output(index) => float y;          // read from output tensor 0

// Models with several input/output tensors
brain.set_input(input_idx, sample_idx, value);
brain.get_output(output_idx, sample_idx) => float y;
```

- Inputs are float32. Outputs may be float32 or int64 (int64 is returned as float).
- Dynamic dimensions (`-1`) are treated as 1.
- Input buffers keep their values between calls to `run()`.

## Example: CurrulaoGRU

`models/currulao_gru.onnx` is a small GRU that predicts the next marimba phrase in
**currulao** (Colombian Pacific), given the last four phrases and their harmonic states
(tonic/dominant). It was trained on transcriptions of Baudilio Cuama and Marino Beltrán
from the book *Arrullos y Currulaos*, adapted to a ten-phrase vocabulary (A–J).

```
ChucK                       Chonnx                      ONNX model
history of 4 phrases  ->    set_input(0..7)       ->    GRU (2 -> 32 -> 10)
+ harmonic states           run()                       softmax
sample next phrase    <-    get_output(0..9)      <-    10 probabilities
```

Run `examples/currulao_gru_predict.ck` to print a generated phrase sequence.
The full musical version (marimba, bombo and cununo) is `ai/CurrulaoGRU.ck` in
[Live-Coding-Colombian-Music](https://github.com/Santiquesada/Live-Coding-Colombian-Music).

`models/currulao.onnx` is an earlier MLP (CurrulaoBrain) that only sees the current
harmonic state and the previous phrase; it is kept for comparison.

## Repository layout

| Path | Contents |
|---|---|
| `Chonnx.cpp` | Chugin source |
| `Chonnx.vcxproj`, `packages.config`, `makefile.win` | Visual Studio build |
| `examples/` | ChucK examples |
| `models/` | Exported ONNX models |
| `training/` | PyTorch scripts that train and export the currulao models |
| `docs/` | Reference documents (Spanish) |

## Building (Windows)

Chonnx currently builds on **Windows only** (it uses the Win32 API to pass UTF-8 paths to
ONNX Runtime).

Requirements: Visual Studio 2022 (C++ desktop workload), the ChucK chugin headers
(`chugin.h`, from the [chugins](https://github.com/ccrma/chugins) repository) and
ONNX Runtime 1.26 (NuGet package `Microsoft.ML.OnnxRuntime`).

1. Restore the NuGet package into `..\packages`:
   `nuget restore packages.config -PackagesDirectory ..\packages`
2. Tell the project where the ChucK headers are, either by placing them in
   `chuck\include\` next to the project, or by creating a `Chonnx.local.props` file:
   ```xml
   <Project>
     <PropertyGroup>
       <ChuckIncludeDir>C:\path\to\chugins\chuck\include</ChuckIncludeDir>
     </PropertyGroup>
   </Project>
   ```
3. Build `Release|x64` in Visual Studio, or run `msbuild Chonnx.vcxproj /p:Configuration=Release /p:Platform=x64`.

## Installing

Copy `Chonnx.chug` to your ChucK chugins folder and `onnxruntime.dll` next to it
(or anywhere on the `PATH`). Then `Chonnx` is available as a class in any ChucK program.

## Training the currulao models

```
pip install torch onnx onnxscript onnxruntime
python training/train_currulao_gru.py    # -> currulao_gru.onnx
python training/train_currulao_mlp.py    # -> currulao.onnx
```

## Author

Santiago Quesada — musician, live coder and electronic engineer.

## License

MIT — see [LICENSE](LICENSE).
