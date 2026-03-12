# SK²Decompile — Reinforcement Learning with VERL

This directory contains the RL (Reinforcement Learning) training pipeline for SK²Decompile, built on top of the [VERL](https://github.com/volcengine/verl) framework (Sheng et al., 2024).

For the full methodology and experimental details, please refer to our paper:
> **SK²Decompile: LLM-based Two-Phase Binary Decompilation from Skeleton to Skin**
> [[arXiv:2509.22114]](https://arxiv.org/abs/2509.22114)

## Overview

After supervised fine-tuning (SFT), SK²Decompile applies reinforcement learning to further align each phase's model with task-specific objectives. We adopt the **GRPO** (Group Relative Policy Optimization) algorithm (DeepSeek-AI et al., 2025) to train both models with their respective reward signals:

- **Structure Recovery** (Skeleton): The reward is based on compiler feedback — a positive reward is granted only if the generated IR successfully compiles, with an additional component reflecting the correctness of placeholder recovery (Equation 3 in the paper).
- **Identifier Naming** (Skin): The reward is the cosine similarity between the embeddings of the generated code and the reference source code, encouraging semantically aligned identifier predictions rather than exact lexical matches (Equation 4 in the paper).

The reward functions and training scripts provided here are **reference implementations** for reproducing the RL training pipeline. For the precise reward formulations and design rationale, please refer to Section 3.5 of the paper.

## Directory Structure

```
SK2DECOMPILE/
├── README.md                          # This file
├── data/
│   └── sk2decompile-rl-examples.jsonl # Example RL training data
├── reward_functions/                  # Reference reward implementations
│   ├── __init__.py
│   ├── exe_type.py                    # Example: compilability + placeholder Jaccard
│   ├── sim_exe.py                     # Example: compilability + word-level similarity
│   ├── embedding_gte.py               # Example: embedding-based identifier similarity (GTE)
│   └── embedding_qwen3.py            # Example: embedding-based identifier similarity (Qwen3)
└── scripts/
    ├── run_struct_rl.sh               # Reference script: Structure Recovery RL
    └── run_ident_rl.sh                # Reference script: Identifier Naming RL
```

## Reward Formulations (from the Paper)

### Structure Recovery Reward (Eq. 3)

The Structure Recovery reward consists of two components:

1. **Compilability**: The generated IR is compiled using the ground-truth header. A reward of 1.0 is granted only upon successful compilation (verified via [Psyche-C](https://github.com/ltcmelo/psychec.git) for header generation).
2. **Placeholder Recovery**: The Jaccard similarity between the generated placeholder set (I_gen) and the ground-truth set (I_IR).

```
r_placeholder = |I_gen ∩ I_IR| / |I_gen ∪ I_IR|

r_structure = { 0.0,                        if IR cannot be compiled
              { 1.0 + r_placeholder,         if IR can be compiled
```

### Identifier Naming Reward (Eq. 4)

The Identifier Naming reward measures the semantic similarity between the generated code and the reference source code using embedding cosine similarity:

```
r_identifier = cos(e_gen, e_src) = (e_gen · e_src) / (||e_gen|| · ||e_src||)
```

where `e_gen` and `e_src` are the embeddings of the generated and reference code respectively. In our experiments, we use qwen-embedding-0.6B (Zhang et al., 2025) as the embedding model.

> **Note**: The reward functions in `reward_functions/` are reference implementations that demonstrate the reward design. Please refer to Section 3.5 of the paper for the complete formulation and design rationale.

## Reproduction Guide

### Step 1: Install VERL

Our RL training is based on **VERL v0.4.1** ([HybridFlow](https://github.com/volcengine/verl), Sheng et al., 2024). We recommend using the same version for reproducibility.

```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout v0.4.1  # or the commit closest to v0.4.1
pip install -e .
```

### Step 2: Integrate Reward Functions

Copy the reward functions into VERL's reward module and register them in the routing dispatcher:

```bash
# Copy reward functions
cp reward_functions/exe_type.py       <VERL_DIR>/verl/utils/reward_score/sk2d_exe_type.py
cp reward_functions/sim_exe.py        <VERL_DIR>/verl/utils/reward_score/sk2d_sim_exe.py
cp reward_functions/embedding_gte.py  <VERL_DIR>/verl/utils/reward_score/sk2d_embedding_gte.py
cp reward_functions/embedding_qwen3.py <VERL_DIR>/verl/utils/reward_score/sk2d_embedding_qwen3.py
```

Then add routing branches to `<VERL_DIR>/verl/utils/reward_score/__init__.py` in the `default_compute_score()` function:

```python
# Structure Recovery reward (example)
elif data_source == "sk2decompile_structure":
    from . import sk2d_exe_type
    res = sk2d_exe_type.compute_score(solution_str, ground_truth, extra_info)

# Identifier Naming reward (example)
elif data_source == "sk2decompile_identifier":
    from . import sk2d_embedding_qwen3
    res = sk2d_embedding_qwen3.compute_score(solution_str, ground_truth, extra_info)
```

The `data_source` field in your training Parquet files determines which reward function is dispatched for each sample.

### Step 3: Prepare Training Data

Training data should be in Parquet format. Each row contains:

| Field | Description |
|-------|-------------|
| `prompt` | Chat-format messages, e.g., `[{"role": "user", "content": "<pseudocode>... What is the source code?"}]` |
| `data_source` | Reward function routing key (must match the branch registered in Step 2) |
| `reward_model.ground_truth` | Expected output (IR for Structure Recovery, source code for Identifier Naming) |
| `reward_model.style` | `"rule"` (rule-based reward) |
| `extra_info.header` | C header declarations for compilability checking (Structure Recovery only) |

See `data/sk2decompile-rl-examples.jsonl` for example data format. Convert JSONL to Parquet before training.

### Step 4: Launch Training

The reference training scripts are in `scripts/`. Edit the configuration variables at the top of each script before launching.

**Structure Recovery RL:**
```bash
# Edit scripts/run_struct_rl.sh to set:
#   VERL_DIR, VENV_PATH, MODEL_PATH, TRAIN_DATA, VAL_DATA, WANDB_*
bash scripts/run_struct_rl.sh
```

**Identifier Naming RL** (requires a running embedding server):
```bash
# 1. Start the embedding server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen3-Embedding-0.6B --port 8000 --dtype float16

# 2. Edit scripts/run_ident_rl.sh to set:
#   VERL_DIR, VENV_PATH, MODEL_PATH, TRAIN_DATA, VAL_DATA, WANDB_*
bash scripts/run_ident_rl.sh
```

### Step 5: Install Additional Dependencies

```bash
# For compiler-based rewards (Structure Recovery)
apt install gcc
pip install psychec  # or build from https://github.com/ltcmelo/psychec.git

# For embedding-based rewards (Identifier Naming)
pip install tree-sitter==0.24.0 tree-sitter-c==0.23.4 openai
```

## Configurations

Reference hyperparameters used in the training scripts:

| Parameter | Structure Recovery | Identifier Naming |
|-----------|:-:|:-:|
| `train_batch_size` | 128 | 128 |
| `max_prompt_length` | 1024 | 1024 |
| `max_response_length` | 2048 | 2048 |
| `lr` | 1e-6 | 1e-6 |
| `kl_loss_coef` | 0.01 | 0.02 |
| `kl_loss_type` | low_var_kl | low_var_kl |
| `rollout.n` (GRPO samples) | 16 | 16 |
| `total_epochs` | 2 | 2 |

## Troubleshooting

**OOM (Out of Memory)**:
- Reduce `ppo_micro_batch_size_per_gpu` (default: 4)
- Enable `actor.fsdp_config.param_offload=True`
- Reduce `rollout.gpu_memory_utilization` (default: 0.80)

**Embedding server connection error** (Identifier Naming only):
- Ensure the vLLM embedding server is running on port 8000
- Check environment variables: `QWEN3_EMBEDDING_API_BASE` (default: `http://127.0.0.1:8000/v1`)

**Compilation timeout in reward** (Structure Recovery only):
- The `gcc -c` call has a 5-second timeout per sample
- If many samples timeout, check if the generated code contains infinite loops
