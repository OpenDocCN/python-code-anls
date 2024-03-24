# `.\lucidrains\tf-bind-transformer\finetune_track.py`

```
# 导入 load_dotenv 函数，用于加载 .env 文件中的环境变量
from dotenv import load_dotenv

# 设置缓存路径在 .env 文件中，并取消下一行的注释
# load_dotenv()

# 导入 Enformer 类和 AdapterModel、BigWigTrainer 类
from enformer_pytorch import Enformer
from tf_bind_transformer import AdapterModel, BigWigTrainer

# 训练常量

# 批量大小
BATCH_SIZE = 1
# 梯度累积步数
GRAD_ACCUM_STEPS = 8
# 学习率
LEARNING_RATE = 1e-4   # Deepmind 在 Enformer 微调中使用了 1e-4

# 有效批量大小为 BATCH_SIZE * GRAD_ACCUM_STEPS = 16

# 每隔多少步进行验证
VALIDATE_EVERY = 250
# 梯度裁剪最大范数
GRAD_CLIP_MAX_NORM = 1.5

# TFactor 文件夹路径
TFACTOR_FOLDER = './tfactor.fastas'
# 人类基因组 FASTA 文件路径
HUMAN_FASTA_FILE_PATH = './hg38.ml.fa'
# 小鼠基因组 FASTA 文件路径
MOUSE_FASTA_FILE_PATH = './mm10.ml.fa'

# 人类基因组区域路径
HUMAN_LOCI_PATH = './chip_atlas/human_sequences.bed'
# 小鼠基因组区域路径
MOUSE_LOCI_PATH = './chip_atlas/mouse_sequences.bed'
# BigWig 文件夹路径
BIGWIG_PATH = './chip_atlas/bigwig'
# 仅包含 BigWig 轨道的文件夹路径
BIGWIG_TRACKS_ONLY_PATH = './chip_atlas/bigwig_tracks_only'
# 注释文件路径
ANNOT_FILE_PATH =  './chip_atlas/annot.tab'

# 目标长度
TARGET_LENGTH = 896

# 保留的目标
HELD_OUT_TARGET = ['GATA2']

# 实例化 Enformer 或加载预训练模型

enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough', target_length = TARGET_LENGTH)

# 实例化模型包装器，接受 Enformer 模型

model = AdapterModel(
    enformer = enformer,
    use_aa_embeds = True,
    use_free_text_context = True,
    free_text_embed_method = 'mean_pool',
    aa_embed_encoder = 'esm',
    finetune_output_heads = dict(
        human = 12,
        mouse = 24
    )
).cuda()

# 用于微调的训练器类

trainer = BigWigTrainer(
    model,
    human_loci_path = HUMAN_LOCI_PATH,
    mouse_loci_path = MOUSE_LOCI_PATH,
    human_fasta_file = HUMAN_FASTA_FILE_PATH,
    mouse_fasta_file = MOUSE_FASTA_FILE_PATH,
    bigwig_folder_path = BIGWIG_PATH,
    bigwig_tracks_only_folder_path = BIGWIG_TRACKS_ONLY_PATH,
    annot_file_path = ANNOT_FILE_PATH,
    target_length = TARGET_LENGTH,
    lr = LEARNING_RATE,
    batch_size = BATCH_SIZE,
    shuffle = True,
    validate_every = VALIDATE_EVERY,
    grad_clip_norm = GRAD_CLIP_MAX_NORM,
    grad_accum_every = GRAD_ACCUM_STEPS,
    human_factor_fasta_folder = TFACTOR_FOLDER,
    mouse_factor_fasta_folder = TFACTOR_FOLDER,
    held_out_targets = HELD_OUT_TARGET
)

# 在 while 循环中执行梯度步骤

while True:
    _ = trainer()
```