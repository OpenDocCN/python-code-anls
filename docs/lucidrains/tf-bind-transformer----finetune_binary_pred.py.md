# `.\lucidrains\tf-bind-transformer\finetune_binary_pred.py`

```py
# 导入 load_dotenv 函数，用于加载 .env 文件中的环境变量
from dotenv import load_dotenv

# 设置缓存路径在 .env 文件中，并取消下一行的注释
# load_dotenv()

# 导入 Enformer 类
from enformer_pytorch import Enformer
# 导入 AdapterModel、Trainer 类
from tf_bind_transformer import AdapterModel, Trainer

# 实例化 Enformer 对象或加载预训练模型
enformer = Enformer.from_hparams(
    dim = 768,
    depth = 4,
    heads = 8,
    target_length = -1,
    use_convnext = True,
    num_downsamples = 6   # 分辨率为 2 ^ 6 == 64bp
)

# 实例化模型包装器，接受 Enformer 对象作为输入
model = AdapterModel(
    enformer = enformer,
    use_aa_embeds = True,
    use_free_text_context = True,
    free_text_embed_method = 'mean_pool',
    binary_target = True,
    target_mse_loss = False,
    use_squeeze_excite = True,
    aa_embed_encoder = 'protalbert'
).cuda()

# 训练常量
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
# 有效批量大小为 BATCH_SIZE * GRAD_ACCUM_STEPS = 16
VALIDATE_EVERY = 250
GRAD_CLIP_MAX_NORM = 1.5

REMAP_FILE_PATH = './remap2022_all.bed'
TFACTOR_FOLDER = './tfactor.fastas'
FASTA_FILE_PATH = './hg38.ml.fa'
NON_PEAK_PATH = './generated-non-peaks.bed'

CONTEXT_LENGTH = 4096

SCOPED_NEGS_REMAP_PATH = './neg-npy/remap2022.bed'
SCOPED_NEGS_PATH = './neg-npy'

TRAIN_CHROMOSOMES = [*range(1, 24, 2), 'X'] # 在奇数染色体上训练
VALID_CHROMOSOMES = [*range(2, 24, 2)]      # 在偶数染色体上验证

HELD_OUT_TARGET = ['AFF4']

# 实例化 Trainer 类用于微调
trainer = Trainer(
    model,
    context_length = CONTEXT_LENGTH,
    batch_size = BATCH_SIZE,
    validate_every = VALIDATE_EVERY,
    grad_clip_norm = GRAD_CLIP_MAX_NORM,
    grad_accum_every = GRAD_ACCUM_STEPS,
    remap_bed_file = REMAP_FILE_PATH,
    negative_bed_file = NON_PEAK_PATH,
    factor_fasta_folder = TFACTOR_FOLDER,
    fasta_file = FASTA_FILE_PATH,
    train_chromosome_ids = TRAIN_CHROMOSOMES,
    valid_chromosome_ids = VALID_CHROMOSOMES,
    held_out_targets = HELD_OUT_TARGET,
    include_scoped_negs = True,
    scoped_negs_remap_bed_path = SCOPED_NEGS_REMAP_PATH,
    scoped_negs_path = SCOPED_NEGS_PATH,
)

# 在 while 循环中执行梯度步骤
while True:
    _ = trainer(finetune_enformer_ln_only = False)
```