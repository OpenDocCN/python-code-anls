# SK²Decompile — Evaluation on BringUpBench

This directory contains the evaluation pipeline for SK²Decompile on the [BringUpBench](https://github.com/toddmaustin/bringup-bench) benchmark, as described in **Section A.6** of our paper:

> **SK²Decompile: LLM-based Two-Phase Binary Decompilation from Skeleton to Skin**
> [[arXiv:2509.22114]](https://arxiv.org/abs/2509.22114)

## Overview

[BringUpBench](https://github.com/toddmaustin/bringup-bench) (Austin, 2024) is a benchmark suite of **90 self-contained C programs** designed for bringing up newly designed CPUs, accelerators, compilers, and operating systems. It has **zero library dependencies** — all programs rely solely on a built-in `libmin` library and only 4 system calls — making it an ideal, standardized test bed for decompilation evaluation on complex, real-world binaries.

We compiled, decompiled, and executed all projects across optimization levels O0–O3, yielding **505 functions** in total. We compared SK²Decompile against the industry-standard rule-based decompiler, **IDA Pro** (Hex-Rays).

## Results

### SK²Decompile vs IDA Pro

| Opt Level | Functions | SK²Decompile Compilable | SK²Decompile Executable | IDA Compilable | IDA Executable |
|:---------:|:---------:|:-----------------------:|:-----------------------:|:--------------:|:--------------:|
| O0 | 382 | **50.26%** | **49.48%** | — | — |
| O1 | 379 | **40.90%** | **39.05%** | — | — |
| O2 | 368 | **37.77%** | **34.24%** | — | — |
| O3 | 359 | **31.75%** | **29.53%** | — | — |
| **Avg** | **1488** | **42.3%** | **27.0%** | **23.6%** | **21.7%** |

> The average row reports the paper's aggregate numbers (Table 8 in Section A.6). Per-opt-level IDA baselines are not separately reported in the paper. Detailed per-benchmark breakdowns are available in `reports/`.

## Directory Structure

```
bringupbench/
├── README.md                              # This file
├── config.env                             # Environment configuration (paths)
├── scripts/
│   ├── build-host-opt-levels.sh           # Step 1: Compile benchmarks at O0-O3
│   ├── decompile-all-pseudo.sh            # Step 2: IDA Pro batch decompilation
│   ├── dump_pseudo.py                     # IDA headless decompilation helper
│   ├── disasm-all-objdump.sh              # Step 3: objdump batch disassembly
│   ├── build-func-maps.py                # Step 4: Build function-level mappings
│   ├── clean-all-benchmarks.sh            # Utility: clean all build artifacts
│   └── eval_infer_out.py                 # Step 5: Automated evaluation
├── data/
│   ├── func_maps/                         # Pre-built function mappings (JSONL)
│   │   ├── merged.O0.func_map.jsonl       # O0: 493 functions
│   │   ├── merged.O1.func_map.jsonl       # O1: 449 functions
│   │   ├── merged.O2.func_map.jsonl       # O2: 441 functions
│   │   └── merged.O3.func_map.jsonl       # O3: 439 functions
│   └── infer_results/                     # SK²Decompile inference results
│       ├── merged.O0.func_map.infer.jsonl # O0: 382 evaluated functions
│       ├── merged.O1.func_map.infer.jsonl # O1: 379 evaluated functions
│       ├── merged.O2.func_map.infer.jsonl # O2: 368 evaluated functions
│       └── merged.O3.func_map.infer.jsonl # O3: 359 evaluated functions
└── reports/                               # Evaluation result summaries
    ├── O0_results.md
    ├── O1_results.md
    ├── O2_results.md
    └── O3_results.md
```

## Reproduction Pipeline

Our evaluation pipeline consists of five steps, as described in the paper:

```
Source (.c)
  │
  ▼  Step 1: Compilation
Binary (.host.O0 ~ .host.O3)
  │
  ├──▶ Step 2: Baseline Extraction (IDA Pro) ──▶ Pseudocode (.pseudo)
  │
  ├──▶ Step 3: Ground Truth Mapping           ──▶ Function Maps (.func_map.jsonl)
  │
  ▼  Step 4: Decompilation (SK²Decompile)
Inferred C code (.func_map.infer.jsonl)
  │
  ▼  Step 5: Validation
Evaluation Reports (reports/)
```

### Prerequisites

| Dependency | Purpose | Installation |
|------------|---------|-------------|
| [Bringup-Bench](https://github.com/toddmaustin/bringup-bench) | Upstream benchmark suite (90 C programs) | `git clone https://github.com/toddmaustin/bringup-bench.git` |
| GCC | Compile benchmarks | `apt install gcc` |
| IDA Pro + Hex-Rays | Decompile binaries to pseudocode | Commercial software |
| objdump (binutils) | Disassemble binaries | `apt install binutils` |
| clang-format | Pseudocode normalization | `apt install clang-format` |
| Python >= 3.10 | Run evaluation scripts | `apt install python3` |

### Quick Start (Evaluation Only)

If you only want to reproduce the evaluation step (Step 5), the pre-built data is included in `data/`. You only need the Bringup-Bench source repository:

```bash
# 1. Clone Bringup-Bench
git clone https://github.com/toddmaustin/bringup-bench.git

# 2. Configure paths
cd bringupbench
vim config.env  # Set BENCH_REPO_ROOT to your bringup-bench path

# 3. Run evaluation (e.g., O0)
python3 scripts/eval_infer_out.py data/infer_results/merged.O0.func_map.infer.jsonl

# 4. Check results
cat reports/O0_results.md
```

### Full Pipeline (From Scratch)

To reproduce the entire pipeline from compilation to evaluation:

```bash
cd bringupbench
vim config.env  # Set BENCH_REPO_ROOT and IDA_BIN
```

**Step 1: Compile benchmarks at O0–O3**

Build all 90 Bringup-Bench programs at four optimization levels, producing `<name>.host.O{0,1,2,3}` binaries.

```bash
scripts/build-host-opt-levels.sh
```

**Step 2: Baseline Extraction (IDA Pro)**

Use IDA Pro in headless mode to decompile all binaries, producing `.pseudo` files with Hex-Rays pseudocode.

```bash
scripts/decompile-all-pseudo.sh
```

Each function is delimited by `/* function_name @ 0xADDRESS */` in the output.

**Step 3: Ground Truth Mapping**

Parse source code, pseudocode, and assembly; match functions by name across all three representations; normalize pseudocode (remove IDA-specific types, hex-to-decimal conversion, clang-format).

```bash
# Disassemble (optional, for assembly mapping)
scripts/disasm-all-objdump.sh

# Build function-level mappings
python3 scripts/build-func-maps.py
```

Output: per-binary `.func_map.jsonl` files. Merge them per optimization level:

```bash
cat $BENCH_REPO_ROOT/*/*.host.O0.func_map.jsonl > data/func_maps/merged.O0.func_map.jsonl
cat $BENCH_REPO_ROOT/*/*.host.O1.func_map.jsonl > data/func_maps/merged.O1.func_map.jsonl
cat $BENCH_REPO_ROOT/*/*.host.O2.func_map.jsonl > data/func_maps/merged.O2.func_map.jsonl
cat $BENCH_REPO_ROOT/*/*.host.O3.func_map.jsonl > data/func_maps/merged.O3.func_map.jsonl
```

**Step 4: Decompilation (SK²Decompile Inference)**

Feed the `pseudo_normalize` field from the function maps to SK²Decompile. The two-phase inference pipeline (see `../sk2decompile_inf.py`) produces C code for each function. Results should be written into the JSONL with the `pseudo.content-fix` field containing the final decompiled function body.

```bash
# Example: use the main SK²Decompile inference pipeline
cd ../  # back to sk2decompile/evaluation/
python3 sk2decompile_inf.py \
    --dataset_path bringupbench/data/func_maps/merged.O0.func_map.jsonl \
    --model_path LLM4Binary/sk2decompile-struct-6.7b \
    --recover_model_path LLM4Binary/sk2decompile-ident-6.7b
```

**Step 5: Validation**

For each function, replace the original source with the decompiled output, rebuild in an isolated workspace, and run the project's test suite.

```bash
python3 scripts/eval_infer_out.py data/infer_results/merged.O0.func_map.infer.jsonl \
    --jobs 16 \
    --command-timeout 20
```

Common options:

```bash
--jobs N              # Parallel workers (default: 96)
--command-timeout S   # Timeout per make command in seconds (default: 20)
--limit N             # Process only first N cases (for debugging)
--keep-workspaces     # Keep temporary build directories
```

## Data Format

### func_map.jsonl (Function Mappings)

Each line is a JSON object containing the source, pseudocode, and assembly for one function:

```jsonc
{
  "source": {
    "path": "ackermann/ackermann.c",          // Source file (relative to BENCH_REPO_ROOT)
    "function_name": "ackermann",              // Function name
    "content": "int ackermann(int m, ...) { ... }\n"  // Complete function body
  },
  "pseudo": {
    "path": "ackermann/ackermann.host.O0.pseudo",
    "function_name": "ackermann",
    "address": "0x11e9",                       // Function address in binary
    "label": "ackermann",
    "content": "__int64 __fastcall ackermann(...) { ... }\n"  // Raw IDA pseudocode
  },
  "pseudo_normalize": "int ackermann(...) { ... }",  // Normalized pseudocode
  "binary": "ackermann/ackermann.host.O0",     // Binary file path
  "assembly": "<ackermann>:\npush %rbp\n..."   // Cleaned objdump output
}
```

### func_map.infer.jsonl (Inference Results)

Extends `func_map.jsonl` with SK²Decompile inference outputs:

```jsonc
{
  // ... all fields from func_map.jsonl ...
  "pseudo": {
    // ... all fields above, plus:
    "content-fix": "..."           // Final decompiled function (used for source replacement)
  },
  "infer-out-model1": "...",       // Phase 1 (Structure Recovery) raw output
  "infer-out-model2": "...",       // Phase 2 (Identifier Naming) raw output
  "pseudo_normalize-fix": "..."    // Corrected normalized pseudocode
}
```

## Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **Replacement Rate** | Fraction of functions where the decompiled output can be located and substituted into the original source file |
| **Compilable Rate** | Fraction of functions where the modified source compiles successfully (`make build`) |
| **Executable Rate** | Fraction of functions where the compiled program passes its test suite (`make test`, output matches reference) |

The evaluation uses BringUpBench's own build infrastructure (`Makefile`, `libmin`, `libtarg`) to compile and validate. Each function is tested in an isolated workspace to prevent cross-contamination.

## Notes

- BringUpBench programs are self-contained with zero external dependencies, making them ideal for evaluating decompilation without the confounding factor of missing headers or libraries.
- The `func_maps/` data contains more functions than `infer_results/` because some functions are filtered during inference (e.g., exceeding token limits).
- All scripts load paths from `config.env`. You can also override via environment variables or CLI arguments (priority: CLI > env > config.env).
- For the complete SK²Decompile methodology and other benchmark results (HumanEval, MBPP, ExeBench, GitHub2025), see the [main README](../../README.md).
