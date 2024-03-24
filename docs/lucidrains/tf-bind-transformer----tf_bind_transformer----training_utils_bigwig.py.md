# `.\lucidrains\tf-bind-transformer\tf_bind_transformer\training_utils_bigwig.py`

```py
import torch
from torch import nn
from tf_bind_transformer.optimizer import get_optimizer
from tf_bind_transformer.data_bigwig import BigWigDataset, BigWigTracksOnlyDataset, get_bigwig_dataloader, get_bigwig_tracks_dataloader
from enformer_pytorch.modeling_enformer import poisson_loss, pearson_corr_coef

def exists(val):
    # 检查值是否存在
    return val is not None

def default(val, d):
    # 如果值存在则返回该值，否则返回默认值
    return val if exists(val) else d

# helpers for logging and accumulating values across gradient steps

def accum_log(log, new_logs):
    # 累积日志中的值
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# simple Trainer class

class BigWigTrainer(nn.Module):
    def __init__(
        self,
        model,
        *,
        human_factor_fasta_folder,
        annot_file_path,
        human_loci_path,
        mouse_loci_path,
        human_fasta_file,
        mouse_fasta_file,
        batch_size,
        bigwig_tracks_only_folder_path = None,
        bigwig_folder_path = None,
        train_chromosome_ids = None,
        valid_chromosome_ids = None,
        mouse_factor_fasta_folder = None,
        downsample_factor = 128,
        target_length = 896,
        lr = 3e-4,
        wd = 0.1,
        validate_every = 250,
        grad_clip_norm = None,
        grad_accum_every = 1,
        held_out_targets_human = [],
        held_out_targets_mouse = [],
        held_out_cell_types_human = [],
        held_out_cell_types_mouse = [],
        context_length = 4096,
        shuffle = False,
        shift_aug_range = (-2, 2),
        rc_aug = False,
        checkpoint_filename = './checkpoint.pt',
        include_biotypes_metadata_in_context = False,
        biotypes_metadata_path = None,
        include_biotypes_metadata_columns = ['germ_layer', 'cellline_cat'],
        biotypes_metadata_delimiter = ' | ',
        bigwig_reduction_type = 'sum',
        enformer_train_valid_split = True
    def forward(
        self,
        finetune_enformer_ln_only = True,
        **kwargs
```