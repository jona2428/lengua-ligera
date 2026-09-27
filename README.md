# lengua-ligera

**Modelos de lenguaje que infieren sólo con enteros.** Pesos ternarios {-1, 0, +1} con escalas por bloque, activaciones int8, LayerNorm con raíz cuadrada entera, softmax y GELU por tabla. La coma flotante se usa sólo para entrenar, como *molde de fundición*: el entregable es un archivo `.lli` que corre en CPUs baratas o silicio chico sin unidad de coma flotante.

*Integer-only language models: ternary weights, int8 activations, integer LayerNorm/softmax. Floating point is only the training "mold"; the deliverable runs with integers. English summary below.*

Todo el núcleo está escrito desde cero en Rust, sin dependencias.

## Resultados (Wikipedia en español, nivel carácter, vocabulario de 128)

| | bits/peso | valid (bpc) |
|---|---|---|
| float FP32, 3M parámetros (CPU) | 32 | 1.8216 |
| ternario QAT, 3M | 2.1 | 1.9441 |
| **motor entero** con el mismo archivo | 2.1 | **1.9441** (99.99% misma predicción top-1) |
| + activaciones int8 | 2.1 | +0.0009 |
| float FP32, 25M parámetros (TPU v5e-1 de Colab, 1 h) | 32 | 1.3457 |

Generación en 1 hilo con caché KV (3M parámetros): float32 441 car/s, 28.5 MB de RAM pico; **entero int8 1689 car/s, 7.7 MB**.

**Honestidad** (¿sabe cuándo no sabe?): la confianza se calcula en enteros y el modelo se abstiene bajo un umbral. Con umbral 90%, responde el 26.7% de Wikipedia con 97.4% de acierto, pero sólo el 0.8% de los caracteres de nombres propios inventados y el 2-3% de ruido o texto invertido. Cuantizar casi no daña la calibración (ECE 0.0040 float vs 0.0057 ternario+int8).

**Formato propio `trit-exp`**: trit × 2^e con exponente compartido por bloque (1.7-1.9 bits/peso). La matmul no necesita multiplicaciones, sólo sumas y shifts.

## Estructura

| | |
|---|---|
| `src/model.rs`, `src/ops.rs` | transformer pre-LN con backprop a mano (gradcheck en `tests/`) |
| `src/bin/train.rs` | entrenamiento FP32 y QAT (straight-through) multihilo |
| `src/quant.rs` | formatos: minifloats, intN, NF4, k-means, ternario, binario, trit-exp |
| `src/export.rs` | funde el molde float a enteros |
| `src/intinfer.rs` | **motor 100% entero** (sin `f32`/`f64`) y formato `.lli` |
| `src/bin/int_check.rs` | verifica motor entero vs float |
| `src/bin/bench_gen.rs` | velocidad y memoria de generación |
| `src/bin/honesty.rs`, `src/bin/ood.rs` | calibración, abstención, fuera de distribución |
| `kaggle/molde.py` | molde JAX para TPU (paridad con Rust a 6 decimales), escribe `.llg` |
| `tools/prep_corpus.py` | corpus desde `wikimedia/wikipedia` 20231101.es |

## Uso

```bash
python tools/prep_corpus.py                      # data/train.bin, valid.bin, vocab.txt
cargo run --release --bin train                  # molde FP32 -> runs/base.llg
INIT=runs/base.llg QAT=ternario-g32 STEPS=1500 OUT=runs/tern.llg cargo run --release --bin train
MODEL=runs/tern.llg FMT=ternario-g32 OUT=runs/tern.lli cargo run --release --bin int_check
ACT_BITS=8 LLI=runs/tern.lli cargo run --release --bin int_sample -- "La ciudad de"
```

Entrenar en TPU: `kaggle/lengua_ligera_colab.ipynb` (Colab) o `kaggle/lengua_ligera_tpu.ipynb` (Kaggle).

## English summary

A from-scratch Rust stack (no dependencies) that trains a character-level transformer in float, quantizes it with QAT to ternary weights, and fuses it into an integer-only inference engine. The integer engine reproduces the float model's bits-per-character exactly; int8 activations cost ~0.001 bpc and run 3.8x faster than float32 on one core. We also measure honesty: an integer confidence threshold lets the model abstain on out-of-distribution input (invented names, noise) while staying accurate on what it answers. A JAX "mold" (`kaggle/molde.py`) trains larger models on TPU and writes checkpoints the Rust engine consumes.

## Autor

Neuroteo ([jona2428](https://github.com/jona2428)), con ayuda de Claude.

## Licencia

GPL-3.0-or-later (ver `LICENSE`). Si necesitas usarlo en un producto cerrado, abre un issue para conversar otra licencia.
