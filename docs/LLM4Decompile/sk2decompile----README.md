# SK²Decompile

**SK²Decompile: LLM-based Two-Phase Binary Decompilation from Skeleton to Skin**

<p align="left">
    🚀&nbsp;<a href="#quick-start">Quick Start</a>
    | 🤖&nbsp;<a href="#training-pipeline">Training Pipeline</a>
    | 📊&nbsp;<a href="#evaluation">Evaluation</a>
    | 📝&nbsp;<a href="#citation">Citation</a>
</p>

## Overview

SK²Decompile is a novel two-phase framework for binary decompilation using Large Language Models (LLMs). Our approach decomposes the complex decompilation task into two manageable phases:

- **Phase 1 Structure Recovery (Skeleton)**: Transform binary/pseudo-code into obfuscated intermediate representations 🤗 [HF Link](https://huggingface.co/LLM4Binary/sk2decompile-struct-6.7b)
- **Phase 2 Identifier Naming (Skin)**: Generate human-readable source code with meaningful identifiers 🤗 [HF Link](https://huggingface.co/LLM4Binary/sk2decompile-ident-6.7)

This repository contains the complete implementation of our paper, including data preprocessing tools, training scripts, and evaluation benchmarks.

## Architecture

Our two-phase approach is inspired by the skeleton-to-skin metaphor:

```
Binary/Pseudo-code → [Phase 1: Skeleton] → Normalized IR → [Phase 2: Skin] → Source Code
                          ↓                                        ↓
                  (Structure Extraction)                   (Identifier Recovery)
```

## Repository Structure

```
SK2Decompile/
├── Preprocess/        # Data preprocessing and normalization tools
├── LLaMA-Factory/     # Supervised Fine-Tuning (SFT) implementation
├── verl/              # Reinforcement Learning (RL) with VERL/GRPO
│   └── SK2DECOMPILE/
│       ├── data/              # Example RL training data
│       ├── reward_functions/  # Custom reward functions (4 variants)
│       ├── scripts/           # Training launch scripts
│       └── README.md          # Detailed RL documentation
├── evaluation/        # Comprehensive evaluation suite
│   ├── bringupbench/          # BringUpBench evaluation (Section A.6)
│   │   ├── scripts/           # Pipeline scripts (compile, decompile, evaluate)
│   │   ├── data/              # Pre-built function maps and inference results
│   │   ├── reports/           # Evaluation result summaries
│   │   └── README.md          # Detailed BringUpBench documentation
│   └── ...                    # HumanEval, MBPP evaluation scripts
└── README.md          # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.0+
- PyTorch 2.0+
- 40GB+ GPU memory (recommended)
- [Psychec](https://github.com/ltcmelo/psychec.git) (for data preprocessing)

### Installation

```bash
git clone https://github.com/yourusername/SK2Decompile.git
cd SK2Decompile
```

## Training Pipeline

### Phase 0: Data Preprocessing

Transform raw pseudo-code into normalized representations suitable for training:

```bash
cd Preprocess

# Requirements
pip install tree-sitter==0.24.0 tree-sitter-c==0.23.4 tqdm

# Step 1: Normalize pseudo-code according to R2I standard
python3 normalize_pseudo.py --input_json exebench_c.json --output_json exebench_pseudonorm.json --key_name pseudo

# Step 2: Obfuscate source code to generate IR
python3 normalize_src_basedonpseudo.py --input_json exebench_pseudonorm.json --output_json exebench_norm_top0.json --top 0 --pseudo pseudo_norm

# Step 3: Format codes with clang-format
python3 format.py --input exebench_norm_top0.json --output exebench_format_top0.json

# Step 4: Infer types for obfuscated IR (used for compiler-based rewards)
python3 inf_type.py --input_json train_format_top0.json --output_name train_format_top0_type \
    --generator ../psychec/psychecgen --solver ../psychec/psychecsolver-exe --split 2 --idx 0
```

### Phase 1: Supervised Fine-Tuning (SFT)

Our two-phase SFT approach trains specialized models for each transformation:

#### Setup LLaMA-Factory
```bash
cd ../LLaMA-Factory
# Follow installation instructions in LLaMA-Factory/README.md
```

#### Train Models
```bash
# Train Skeleton Model (pseudo2norm)
llamafactory-cli train LLaMA-Factory/SK2DECOMPILE/train/pseudo2norm-example.yaml

# Train Skin Model (norm2code)
llamafactory-cli train LLaMA-Factory/SK2DECOMPILE/train/norm2code-example.yaml
```

**Sample Training Data:**
- Pseudo2Norm: `LLaMA-Factory/SK2DECOMPILE/data/pseudo2norm-examples.jsonl`
- Norm2Code: `LLaMA-Factory/SK2DECOMPILE/data/norm2code-examples.jsonl`

### Phase 2: Reinforcement Learning (RL)

After SFT, we apply GRPO (Group Relative Policy Optimization) to further align each model with task-specific objectives (Section 3.5 of the paper):

- **Structure Recovery**
- **Identifier Naming**

Our RL training is based on [VERL](https://github.com/volcengine/verl) v0.4.1 (Sheng et al., 2024).

#### Setup VERL
```bash
git clone https://github.com/volcengine/verl.git
cd verl && git checkout v0.4.1 && pip install -e .
pip install tree-sitter==0.24.0 tree-sitter-c==0.23.4 openai
```

#### Run RL Training
```bash
# Structure Recovery RL
bash verl/SK2DECOMPILE/scripts/run_struct_rl.sh

# Identifier Naming RL (requires embedding server)
bash verl/SK2DECOMPILE/scripts/run_ident_rl.sh
```

See [`verl/SK2DECOMPILE/README.md`](verl/SK2DECOMPILE/README.md) for the full reproduction guide, including how to integrate reward functions into VERL and prepare training data.

**RL Training Data:** `verl/SK2DECOMPILE/data/sk2decompile-rl-examples.jsonl`

## Evaluation
```
cd ../evaluation
```

**Inference**
```
pip install vllm
apt install clang-format
#translate the data into reverse_sample.json format
python normalize_pseudo.py --input_json reverse_sample.json --output_json reverse_sample.json
python sk2decompile.py --dataset_path reverse_sample.json --model_path LLM4Binary/sk2decompile-struct-6.7b --recover_model_path LLM4Binary/sk2decompile-ident-6.7b
```

**Detailed version**
0. Install `vllm` and `transformers` via `pip`; install `clang-format` via `apt`.
1. Prepare a Linux-x64 executable file (ELF).
2. Use IDA to decompile it (you can also simply use this website: [https://dogbolt.org/](https://dogbolt.org/)).
3. Convert the data into the corresponding format
   ([https://huggingface.co/LLM4Binary/sk2decompile-struct-6.7b/blob/main/reverse_sample.json](https://huggingface.co/LLM4Binary/sk2decompile-struct-6.7b/blob/main/reverse_sample.json))
4. Run inference:

   ```bash
   python normalize_pseudo.py --input_json reverse_sample.json --output_json reverse_sample.json
   python sk2decompile.py --dataset_path reverse_sample.json \
       --model_path LLM4Binary/sk2decompile-struct-6.7b \
       --recover_model_path LLM4Binary/sk2decompile-ident-6.7b
   ```

**Model page:**
[https://huggingface.co/LLM4Binary/sk2decompile-struct-6.7b](https://huggingface.co/LLM4Binary/sk2decompile-struct-6.7b)

**Project overview:**
[https://github.com/albertan017/LLM4Decompile/tree/main/sk2decompile](https://github.com/albertan017/LLM4Decompile/tree/main/sk2decompile)

**Notes:**

* IDA decompilation results should be preprocessed before inference.
* Use `vllm` to recover function structure (`sk2decompile-struct`) and variable names (`sk2decompile-ident`) step by step.
* Training was done on C language Linux-x64 code with IDA pseudocode; performance may degrade for other languages or architectures.


Comprehensive evaluation on standard benchmarks:

```bash
# evaluate exe_rate
python evaluate_exe.py --json_file your_json_file_path
                       --dcompilers decompiler1,decompiler2,...,decompilerN
# evaluate r2i
python evaluate_r2i.py --json_file your_json_file_path
                       --dcompilers decompiler1,decompiler2,...,decompilerN
                       --output_path your_output_path
# evaluate gpt-judge
python gpt_judge.py --json_file your_json_file_path
                    --decompilers decompiler1,decompiler2,...,decompilerN
                    --opt OPT
                    --api_key your_openai_api_key
```

**BringUpBench Evaluation** (Section A.6 of the paper)

We also evaluate on [BringUpBench](https://github.com/toddmaustin/bringup-bench) — 90 self-contained C programs with 505 functions across O0–O3. SK²Decompile achieves **42.3% compilation rate** and **27.0% re-executability rate**, compared to IDA Pro's 23.6% / 21.7%.

See [`evaluation/bringupbench/README.md`](evaluation/bringupbench/README.md) for the full reproduction pipeline, pre-built data, and detailed results.

## 📊 Results

Our approach achieves state-of-the-art performance:

| Metric | Dataset | Improvement |
|--------|---------|-------------|
| **Re-executability** | HumanEval | **+21.6%** over GPT-5-mini |
| **R2I Score** | GitHub2025 | **+29.4%** over Idioms |

## 🔬 Key Innovations

1. **Two-Phase Decomposition**: Separating structure recovery from identifier prediction
2. **Compiler-Based RL**: Using compiler feedback as reward signal
3. **Generic Placeholders**: Language-agnostic intermediate representation
4. **Independent Optimization**: Separate RL objectives for each phase

## Citation

If you use SK²Decompile in your research, please cite our paper:

```bibtex
@article{tan2025sk2decompile,
  title={SK2Decompile: LLM-based Two-Phase Binary Decompilation from Skeleton to Skin},
  author={Tan, Hanzhuo and Li, Weihao and Tian, Xiaolong and Wang, Siyi and Liu, Jiaming and Li, Jing and Zhang, Yuqun},
  journal={arXiv preprint arXiv:2509.22114},
  year={2025}
}
```

## Contributing

We welcome contributions! Areas of interest:
- Support for additional architectures (ARM, RISC-V)
- Integration with more decompilation tools
- Improved intermediate representations
- Multi-language support

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

We thank the developers of:
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the SFT framework
- [VERL](https://github.com/volcengine/verl) for the RL implementation
- [Psychec](https://github.com/ltcmelo/psychec.git) for C type inference

---

For detailed documentation on each component, please refer to the individual README files in each directory.
