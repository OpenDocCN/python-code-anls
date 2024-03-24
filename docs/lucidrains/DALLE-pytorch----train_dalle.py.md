# `.\lucidrains\DALLE-pytorch\train_dalle.py`

```
# 导入必要的库
import argparse
from pathlib import Path
import time
from glob import glob
import os
import shutil

import torch
import wandb  # 如果用户没有安装 wandb，则提前退出
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# 导入 DALL-E 相关模块
from dalle_pytorch import __version__
from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE, DiscreteVAE, DALLE
from dalle_pytorch import distributed_utils
from dalle_pytorch.loader import TextImageDataset
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, ChineseTokenizer, YttmTokenizer

# 导入用于支持 webdataset 的库
import webdataset as wds
from torchvision import transforms as T
from PIL import Image
from io import BytesIO

# 参数解析
parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=False)

# 添加参数：离散 VAE 的路径
group.add_argument('--vae_path', type=str,
                   help='path to your trained discrete VAE')

# 添加参数：部分训练的 DALL-E 的路径
group.add_argument('--dalle_path', type=str,
                   help='path to your partially trained DALL-E')

# 添加参数：训练好的 VQGAN 权重路径
parser.add_argument('--vqgan_model_path', type=str, default=None,
                   help='path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)')

# 添加参数：训练好的 VQGAN 配置路径
parser.add_argument('--vqgan_config_path', type=str, default=None,
                   help='path to your trained VQGAN config. This should be a .yaml file. (only valid when taming option is enabled)')

# 添加参数：包含图像和文本用于学习 DALL-E 的文件夹路径
parser.add_argument('--image_text_folder', type=str, required=True,
                    help='path to your folder of images and text for learning the DALL-E')

# 添加参数：WebDataset 的列名，用于图像和文本
parser.add_argument('--wds', type=str, default='',
                    help='Comma separated list of WebDataset (1) image and (2) text column names. Must contain 2 values, e.g. img,cap.')

# 添加参数：是否截断超过最大标记长度的标题
parser.add_argument('--truncate_captions', dest='truncate_captions', action='store_true',
                    help='Captions passed in which exceed the max token length will be truncated if this is set.')

# 添加参数：随机调整裁剪的较低比率
parser.add_argument('--random_resize_crop_lower_ratio', dest='resize_ratio', type=float, default=0.75,
                    help='Random resized crop lower ratio')

# 添加参数：是否使用中文
parser.add_argument('--chinese', dest='chinese', action='store_true')

# 添加参数：是否启用 taming 模式
parser.add_argument('--taming', dest='taming', action='store_true')

# 添加参数：是否使用 Hugging Face Tokenizer
parser.add_argument('--hug', dest='hug', action='store_true')

# 添加参数：BPE json 文件路径
parser.add_argument('--bpe_path', type=str,
                    help='path to your BPE json file')

# 添加参数：DALL-E 输出文件名
parser.add_argument('--dalle_output_file_name', type=str, default="dalle",
                    help='output_file_name')

# 添加参数：启用 DeepSpeed 16 位精度
parser.add_argument('--fp16', action='store_true',
                    help='(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.')

# 添加参数：启用 Apex "O1" 自动混合精度
parser.add_argument('--amp', action='store_true',
                   help='Apex "O1" automatic mixed precision. More stable than 16 bit precision. Can\'t be used in conjunction with deepspeed zero stages 1-3.')

# 添加参数：W&B 保存结果时使用的名称
parser.add_argument('--wandb_name', default='dalle_train_transformer',
                    help='Name W&B will use when saving results.\ne.g. `--wandb_name "coco2017-full-sparse"`')

# 添加参数：W&B 日志记录的团队/实体名称
parser.add_argument('--wandb_entity', default=None,
                    help='(optional) Name of W&B team/entity to log to.')

# 添加参数：稳定 softmax，防止在 softmax 过程中值变得过大
parser.add_argument('--stable_softmax', dest='stable_softmax', action='store_true',
                    help='Prevent values from becoming too large during softmax. Helps with stability in fp16 and Mixture of Quantization training.')

# 分布式训练参数
parser = distributed_utils.wrap_arg_parser(parser)

# 训练设置参数
train_group = parser.add_argument_group('Training settings')

# 添加参数：是否启用 FLOPS 分析
train_group.add_argument('--flops_profiler', dest='flops_profiler', action='store_true', help='Exits after printing detailed flops/runtime analysis of forward/backward')

# 添加参数：训练轮数
train_group.add_argument('--epochs', default=20, type=int, help='Number of epochs')
# 添加一个参数到训练组，保存每n步一个检查点
train_group.add_argument('--save_every_n_steps', default=1000, type=int, help='Save a checkpoint every n steps')

# 添加一个参数到训练组，保留n个检查点，如果检查点数量超过n则删除旧的deepspeed检查点（谨慎操作）
train_group.add_argument('--keep_n_checkpoints', default=None, type=int, help='(Careful) Deletes old deepspeed checkpoints if there are more than n')

# 添加一个参数到训练组，批量大小
train_group.add_argument('--batch_size', default=4, type=int, help='Batch size')

# 添加一个参数到训练组，GA步数，每次迭代中跨步累积梯度的步数。仅适用于DeepSpeed。
train_group.add_argument('--ga_steps', default=1, type=int, help='Number of steps to accumulate gradients across per each iteration. DeepSpeed only.')

# 添加一个参数到训练组，学习率
train_group.add_argument('--learning_rate', default=3e-4, type=float, help='Learning rate')

# 添加一个参数到训练组，梯度规范化裁剪
train_group.add_argument('--clip_grad_norm', default=0.5, type=float, help='Clip gradient norm')

# 添加一个参数到训练组，学习率衰减
train_group.add_argument('--lr_decay', dest='lr_decay', action='store_true')

# 创建模型设置参数组
model_group = parser.add_argument_group('Model settings')

# 添加一个参数到模型设置组，模型维度
model_group.add_argument('--dim', default=512, type=int, help='Model dimension')

# 添加一个参数到模型设置组，文本序列长度
model_group.add_argument('--text_seq_len', default=256, type=int, help='Text sequence length')

# 添加一个参数到模型设置组，模型深度
model_group.add_argument('--depth', default=2, type=int, help='Model depth')

# 添加一个参数到模型设置组，模型头数
model_group.add_argument('--heads', default=8, type=int, help='Model number of heads')

# 添加一个参数到模型设置组，模型头维度
model_group.add_argument('--dim_head', default=64, type=int, help='Model head dimension')

# 添加一个参数到训练组，前馈层dropout
train_group.add_argument('--ff_dropout', default=0.0, type=float, help='Feed forward dropout.')

# 添加一个参数到训练组，注意力dropout
train_group.add_argument('--attn_dropout', default=0.0, type=float, help='Feed forward dropout.')

# 添加一个参数到模型设置组，可逆性
model_group.add_argument('--reversible', dest='reversible', action='store_true')

# 添加一个参数到模型设置组，图像损失权重
model_group.add_argument('--loss_img_weight', default=7, type=int, help='Image loss weight')

# 添加一个参数到模型设置组，注意力类型
model_group.add_argument('--attn_types', default='full', type=str, help='comma separated list of attention types. attention type can be: full or sparse or axial_row or axial_col or conv_like.')

# 添加一个参数到模型设置组，使用移位标记特性
model_group.add_argument('--shift_tokens', help='Use the shift tokens feature', action='store_true')

# 添加一个参数到模型设置组，使用旋转嵌入
model_group.add_argument('--rotary_emb', help='Use rotary embeddings', action='store_true')

# 添加一个参数到模型设置组，共享注意力层ID
model_group.add_argument('--shared_attn_ids', default=None, type=str, help='Comma separated list of shared attention layer ids. Default: sharing is disabled')

# 添加一个参数到模型设置组，共享前馈层ID
model_group.add_argument('--shared_ff_ids', default=None, type=str, help='Comma separated list of shared feed forward layer ids. Default: sharing is disabled')

# 添加一个参数到模型设置组，共享输入和输出嵌入
model_group.add_argument('--share_input_output_emb', help='Share input and output embeddings', action='store_true')

# 解析命令行参数
args = parser.parse_args()

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 获取可训练参数
def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

# 将检查点路径转换为带有插入标签的目录
def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, Path):
        cp_path = Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir

# 常量

# 图像文本列
WEBDATASET_IMAGE_TEXT_COLUMNS = tuple(args.wds.split(','))
ENABLE_WEBDATASET = True if len(WEBDATASET_IMAGE_TEXT_COLUMNS) == 2 else False

# DALLE输出文件名
DALLE_OUTPUT_FILE_NAME = args.dalle_output_file_name + ".pt"

# VAE路径
VAE_PATH = args.vae_path
VQGAN_MODEL_PATH = args.vqgan_model_path
VQGAN_CONFIG_PATH = args.vqgan_config_path
DALLE_PATH = args.dalle_path
RESUME = exists(DALLE_PATH)

# 训练周期
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

# 学习率
LEARNING_RATE = args.learning_rate
GRAD_CLIP_NORM = args.clip_grad_norm
LR_DECAY = args.lr_decay
SAVE_EVERY_N_STEPS = args.save_every_n_steps
KEEP_N_CHECKPOINTS = args.keep_n_checkpoints

# 模型维度
MODEL_DIM = args.dim
TEXT_SEQ_LEN = args.text_seq_len
DEPTH = args.depth
HEADS = args.heads
DIM_HEAD = args.dim_head
REVERSIBLE = args.reversible
# 从参数中获取损失图像权重
LOSS_IMG_WEIGHT = args.loss_img_weight
# 从参数中获取前馈神经网络的丢弃率
FF_DROPOUT = args.ff_dropout
# 从参数中获取注意力机制的丢弃率
ATTN_DROPOUT = args.attn_dropout
# 从参数中获取是否使用稳定的 softmax 函数
STABLE = args.stable_softmax
# 从参数中获取是否移动标记
SHIFT_TOKENS = args.shift_tokens
# 从参数中获取是否使用旋转嵌入
ROTARY_EMB = args.rotary_emb

# 从参数中获取注意力类型并转换为元组
ATTN_TYPES = tuple(args.attn_types.split(','))
# 如果存在共享的注意力 ID，则从参数中获取并转换为元组，否则为 None
SHARED_ATTN_IDS = tuple(args.shared_attn_ids.split(',')) if exists(args.shared_attn_ids) else None
# 如果存在共享的前馈神经网络 ID，则从参数中获取并转换为元组，否则为 None
SHARED_FF_IDS = tuple(args.shared_ff_ids.split(',')) if exists(args.shared_ff_ids) else None
# 从参数中获取是否共享输入输出嵌入
SHARE_INPUT_OUTPUT_EMB = args.share_input_output_emb

# 定义 DeepSpeed 检查点辅助文件名
DEEPSPEED_CP_AUX_FILENAME = 'auxiliary.pt'

# 如果未启用 WebDataset
if not ENABLE_WEBDATASET:
    # 如果指定的图像文本文件夹不存在，则抛出异常
    assert Path(args.image_text_folder).exists(), f'The path {args.image_text_folder} was not found.'
# 如果启用了 WebDataset
else:
    # 如果图像文本文件夹是一个目录
    if Path(args.image_text_folder).is_dir():
        # 获取目录下所有的 .tar 文件路径
        DATASET = [str(p) for p in Path(args.image_text_folder).glob("**/*") if ".tar" in str(p).lower()] # .name
        # 如果找到的 .tar 文件数量为 0，则抛出异常
        assert len(DATASET) > 0, 'The directory ({}) does not contain any WebDataset/.tar files.'.format(args.image_text_folder)
        print('Found {} WebDataset .tar(.gz) file(s) under given path {}!'.format(len(DATASET), args.image_text_folder))
    # 如果图像文本文件夹是一个 http(s) 链接
    elif ('http://' in args.image_text_folder.lower()) | ('https://' in args.image_text_folder.lower()):
        # 设置 DATASET 为 http(s) 链接
        DATASET = f"pipe:curl -L -s {args.image_text_folder} || true"
        print('Found {} http(s) link under given path!'.format(len(DATASET), args.image_text_folder))
    # 如果图像文本文件夹是一个 Google Cloud Storage (GCS) 链接
    elif 'gs://' in args.image_text_folder.lower():
        # 设置 DATASET 为 GCS 链接
        DATASET = f"pipe:gsutil cat {args.image_text_folder} || true"
        print('Found {} GCS link under given path!'.format(len(DATASET), args.image_text_folder))
    # 如果图像文本文件夹包含 .tar 文件
    elif '.tar' in args.image_text_folder:
        # 设置 DATASET 为图像文本文件夹路径
        DATASET = args.image_text_folder
        print('Found WebDataset .tar(.gz) file under given path {}!'.format(args.image_text_folder))
    else:
        # 抛出异常，未提供文件夹、.tar(.gz) 文件或指向 .tar 文件的 URL
        raise Exception('No folder, no .tar(.gz) and no url pointing to tar files provided under {}.'.format(args.image_text_folder))

# 初始化分布式后端
distr_backend = distributed_utils.set_backend_from_args(args)
distr_backend.initialize()

# 检查是否使用 DeepSpeed
using_deepspeed = distributed_utils.using_backend(distributed_utils.DeepSpeedBackend)
# 检查当前进程是否为根进程
is_root = distr_backend.is_root_worker()

# 分词器
if exists(args.bpe_path):
    # 根据 BPE 路径选择分词器类
    klass = HugTokenizer if args.hug else YttmTokenizer
    tokenizer = klass(args.bpe_path)
elif args.chinese:
    # 如果是中文文本，则使用中文分词器
    tokenizer = ChineseTokenizer()

# 重建 VAE
if RESUME:
    # 获取 DALL-E 模型路径
    dalle_path = Path(DALLE_PATH)
    # 如果使用 DeepSpeed，则获取 DeepSpeed 检查点目录
    if using_deepspeed:
        cp_dir = cp_path_to_dir(dalle_path, 'ds')
        # 检查 DeepSpeed 检查点目录是否存在
        assert cp_dir.is_dir(), f'DeepSpeed checkpoint directory {cp_dir} not found'
        dalle_path = cp_dir / DEEPSPEED_CP_AUX_FILENAME
    else:
        # 检查 DALL-E 模型文件是否存在
        assert dalle_path.exists(), 'DALL-E model file does not exist'
    # 加载模型参数、VAE 参数、权重等信息
    loaded_obj = torch.load(str(dalle_path), map_location='cpu')

    dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']
    opt_state = loaded_obj.get('opt_state')
    scheduler_state = loaded_obj.get('scheduler_state')

    # 根据 VAE 参数初始化 VAE 模型
    if vae_params is not None:
        vae = DiscreteVAE(**vae_params)
    elif args.taming:
        vae = VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)
    else:
        vae = OpenAIDiscreteVAE()

    # 获取恢复的训练轮数
    resume_epoch = loaded_obj.get('epoch', 0)
else:
    # 如果存在 VAE 模型路径
    if exists(VAE_PATH):
        # 获取 VAE 模型路径
        vae_path = Path(VAE_PATH)
        # 检查 VAE 模型文件是否存在
        assert vae_path.exists(), 'VAE model file does not exist'
        assert not vae_path.is_dir(), \
            ('Cannot load VAE model from directory; please use a '
             'standard *.pt checkpoint. '
             'Currently, merging a DeepSpeed-partitioned VAE into a DALLE '
             'model is not supported.')

        # 加载 VAE 模型参数和权重
        loaded_obj = torch.load(str(vae_path))

        vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']

        # 根据 VAE 参数初始化 VAE 模型，并加载权重
        vae = DiscreteVAE(**vae_params)
        vae.load_state_dict(weights)
    else:
        # 如果不是预训练模型，则打印提示信息
        if is_root:
            print('using pretrained VAE for encoding images to tokens')
        # 初始化 VAE 参数为 None
        vae_params = None

        # 如果使用 Taming 模型
        if args.taming:
            # 使用 VQGanVAE 模型
            vae = VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)
        else:
            # 使用 OpenAIDiscreteVAE 模型
            vae = OpenAIDiscreteVAE()

    # 初始化 DALL-E 参数字典
    dalle_params = dict(
        num_text_tokens=tokenizer.vocab_size,
        text_seq_len=TEXT_SEQ_LEN,
        dim=MODEL_DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        reversible=REVERSIBLE,
        loss_img_weight=LOSS_IMG_WEIGHT,
        attn_types=ATTN_TYPES,
        ff_dropout=FF_DROPOUT,
        attn_dropout=ATTN_DROPOUT,
        stable=STABLE,
        shift_tokens=SHIFT_TOKENS,
        rotary_emb=ROTARY_EMB,
        shared_attn_ids=SHARED_ATTN_IDS,
        shared_ff_ids=SHARED_FF_IDS,
        share_input_output_emb=SHARE_INPUT_OUTPUT_EMB,
    )
    # 初始化恢复训练的轮次为 0
    resume_epoch = 0
# 设置图像大小为VAE的图像大小
IMAGE_SIZE = vae.image_size
# 设置通道数为VAE的通道数
CHANNELS = vae.channels
# 判断是否为透明通道
TRANSPARENT = CHANNELS == 4
# 设置图像模式为RGBA或RGB
IMAGE_MODE = 'RGBA' if CHANNELS == 4 else 'RGB'

# 配置OpenAI VAE为float16s
if isinstance(vae, OpenAIDiscreteVAE) and args.fp16:
    # 如果是OpenAI离散VAE并且启用了fp16，设置编码器的输出卷积为float16
    vae.enc.blocks.output.conv.use_float16 = True

# 辅助函数

# 对模型的参数进行分组
def group_weight(model):
    group_decay, group_no_decay = [], []
    for params in model.named_parameters():
        if 'transformer' in params[0]:
            if 'bias' in params[0] or 'norm' in params[0]:
                group_no_decay.append(params[1])
                continue
        group_decay.append(params[1])

    assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

# 创建数据集和数据加载器

# 是否打乱数据集
is_shuffle = not distributed_utils.using_backend(distributed_utils.HorovodBackend)

# 图像预处理
imagepreproc = T.Compose([
    T.Lambda(lambda img: img.convert(IMAGE_MODE) if img.mode != IMAGE_MODE else img),
    T.RandomResizedCrop(IMAGE_SIZE, scale=(args.resize_ratio, 1.), ratio=(1., 1.)),
    T.ToTensor(),
])

# 图像转换函数
def imagetransform(b):
    return Image.open(BytesIO(b))

# 分词函数
def tokenize(s):
    return tokenizer.tokenize(s.decode('utf-8'), TEXT_SEQ_LEN, truncate_text=args.truncate_captions).squeeze(0)

if ENABLE_WEBDATASET:
    # 设置数据集大小
    DATASET_SIZE = int(1e9) # You need to set a nominal length for the Dataset in order to avoid warnings from DataLoader

    myimg, mycap = WEBDATASET_IMAGE_TEXT_COLUMNS
    # 图像文本映射
    image_text_mapping = {
        myimg: imagetransform,
        mycap: tokenize
    }
    # 图像映射
    image_mapping = {
        myimg: imagepreproc
    }

    # 数据集过滤函数
    def filter_dataset(item):
        if mycap not in item:
            return False
        if myimg not in item:
            return False
        return True

    # 创建WebDataset
    w_dataset = wds.WebDataset(DATASET, handler=wds.warn_and_continue)
    filtered_dataset = w_dataset.select(filter_dataset)
    ds = filtered_dataset.map_dict(**image_text_mapping).map_dict(**image_mapping).to_tuple(mycap, myimg).batched(BATCH_SIZE / distr_backend.get_world_size(), partial=True)
else:
    # 创建TextImageDataset
    ds = TextImageDataset(
        args.image_text_folder,
        text_len=TEXT_SEQ_LEN,
        image_size=IMAGE_SIZE,
        transparent=TRANSPARENT,
        resize_ratio=args.resize_ratio,
        truncate_captions=args.truncate_captions,
        tokenizer=tokenizer,
        shuffle=is_shuffle,
    )
    assert len(ds) > 0, 'dataset is empty'

if is_root:
    if not ENABLE_WEBDATASET:
        print(f'{len(ds)} image-text pairs found for training')

# 数据采样器

data_sampler = None

if not is_shuffle:
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        ds,
        num_replicas=distr_backend.get_world_size(),
        rank=distr_backend.get_rank()
    )

# WebLoader用于WebDataset和DeepSpeed兼容性

if ENABLE_WEBDATASET:
    dl = wds.WebLoader(ds, batch_size=None, shuffle=False, num_workers=4) # optionally add num_workers=2 (n) argument
    number_of_batches = DATASET_SIZE // (BATCH_SIZE * distr_backend.get_world_size())
    dl = dl.slice(number_of_batches)
    dl.length = number_of_batches
else:
    # 用于图像文本文件夹数据集的常规DataLoader
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=is_shuffle, drop_last=True, sampler=data_sampler)

# 初始化DALL-E

dalle = DALLE(vae=vae, **dalle_params)

if not using_deepspeed:
    if args.fp16:
        # 如果启用fp16，将DALL-E设置为半精度
        dalle = dalle.half()
    # 将DALL-E移动到GPU
    dalle = dalle.cuda()

if RESUME and not using_deepspeed:
    # 如果恢复训练并且不使用DeepSpeed，加载权重
    dalle.load_state_dict(weights)

# 优化器

# 创建Adam优化器
opt = Adam(get_trainable_params(dalle), lr=LEARNING_RATE)

if RESUME and opt_state:
    # 如果恢复训练并且有优化器状态，加载优化器状态
    opt.load_state_dict(opt_state)

# 调度器

scheduler = None

if LR_DECAY:
    # 创建一个学习率调度器 ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        opt,  # 传入优化器
        mode="min",  # 设置模式为最小化
        factor=0.5,  # 学习率调整因子
        patience=10,  # 忍耐次数
        cooldown=10,  # 冷却时间
        min_lr=1e-6,  # 最小学习率
        verbose=True,  # 是否打印信息
    )
    # 如果 RESUME 为真且存在学习率调度器状态
    if RESUME and scheduler_state:
        # 加载学习率调度器状态
        scheduler.load_state_dict(scheduler_state)
# 实验跟踪器

# 如果是根节点
if is_root:

    # 定义模型配置字典
    model_config = dict(
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD
    )

    # 初始化 wandb 实验
    run = wandb.init(
        project=args.wandb_name,
        entity=args.wandb_entity,
        resume=False,
        config=model_config,
    )

# 分发

# 检查批量大小是否符合要求
distr_backend.check_batch_size(BATCH_SIZE)
# 配置 DeepSpeed
deepspeed_config = {
    'train_batch_size': BATCH_SIZE,
    'gradient_accumulation_steps': args.ga_steps,
    'gradient_clipping': GRAD_CLIP_NORM,
    'fp16': {
        'enabled': args.fp16,
    },
    'amp': {
        'enabled': args.amp,
        'opt_level': 'O1',
    },
    "flops_profiler": {
        "enabled": args.flops_profiler,
        "profile_step": 200,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": True,
        "output_file": None # TODO 无法使其工作。
    },
}

# 如果 DeepSpeed 配置中的零优化阶段大于等于 2
if deepspeed_config.get('zero_optimization', {}).get('stage', 0) >= 2:
    print(f"Checkpoints made with DeepSpeed ZeRO Stages 2 and 3 will be stored in deepspeed checkpoint folder")
    print(f"As such, they will require DeepSpeed as a dependency in order to resume from or generate with.")
    print("See the deespeed conversion script for details on how to convert your ZeRO stage 2/3 checkpoint to a single file.")
    print("If using a single GPU, consider running with apex automatic mixed precision instead for a similar speedup to ZeRO.")
    time.sleep(2)

# 分发模型、优化器、数据加载器和调度器
(distr_dalle, distr_opt, distr_dl, distr_scheduler) = distr_backend.distribute(
    args=args,
    model=dalle,
    optimizer=opt,
    model_parameters=get_trainable_params(dalle),
    training_data=(
        (None if ENABLE_WEBDATASET else ds)
        if using_deepspeed
        else dl
    ),
    # 不将 LR 调度器传递给 DeepSpeed，以便手动推进
    lr_scheduler=scheduler if LR_DECAY and not using_deepspeed else None,
    config_params=deepspeed_config,
)
# 优先使用 `deepspeed_config` 中的调度器。

# 如果启用了 LR 衰减且分发调度器为 None，则使用全局调度器
if LR_DECAY and distr_scheduler is None:
    distr_scheduler = scheduler

# 如果正在使用 DeepSpeed 并且启用了 fp16
avoid_model_calls = using_deepspeed and args.fp16

# 如果恢复训练并且正在使用 DeepSpeed
if RESUME and using_deepspeed:
    distr_dalle.load_checkpoint(str(cp_dir))

# 保存模型
def save_model(path, epoch=0):
    save_obj = {
        'hparams': dalle_params,
        'vae_params': vae_params,
        'epoch': epoch,
        'version': __version__,
        'vae_class_name': vae.__class__.__name__
    }

    # 如果使用 DeepSpeed
    if using_deepspeed:
        cp_dir = cp_path_to_dir(path, 'ds')

        # 如果保留的检查点数量不为 None 且为根节点
        if KEEP_N_CHECKPOINTS is not None and is_root:
            checkpoints = sorted(glob(str(cp_dir / "global*")), key=os.path.getmtime, reverse=True)
            for checkpoint in checkpoints[KEEP_N_CHECKPOINTS:]:
                shutil.rmtree(checkpoint)

        # 保存 DeepSpeed 检查点
        distr_dalle.save_checkpoint(cp_dir, client_state=save_obj)

        if not is_root:
            return

        # 保存辅助值以便重用标准加载程序
        save_obj = {
            **save_obj,
            # 保存一个无意义的值，指导用户进一步帮助
            'weights': (
                'To get a working standard checkpoint, '
                'look into consolidating DeepSpeed checkpoints.'
            ),
        }
        torch.save(save_obj, str(cp_dir / DEEPSPEED_CP_AUX_FILENAME))
        if deepspeed_config.get('zero_optimization', {}).get('stage', 0) >= 2: # 参见 https://github.com/lucidrains/DALLE-pytorch/wiki/DeepSpeed-Checkpoints
            return

    if not is_root:
        return

    save_obj = {
        **save_obj,
        'weights': dalle.state_dict(),
        'opt_state': opt.state_dict(),
        'scheduler_state': (scheduler.state_dict() if scheduler else None)
    }

    torch.save(save_obj, path)

# 保存模型配置和路径为 artifact
def save_artifact(model_config, model_path, name = 'trained-dalle'):
    model_artifact = wandb.Artifact(name, type='model', metadata=dict(model_config))
    model_artifact.add_file(model_path)
    run.log_artifact(model_artifact)

# 训练
# 在训练开始之前保存一个检查点，以便在配置错误时提前失败
# 参考 https://github.com/lucidrains/DALLE-pytorch/wiki/DeepSpeed-Checkpoints

# 保存模型
save_model(DALLE_OUTPUT_FILE_NAME, epoch=resume_epoch)

# 循环每个 epoch
for epoch in range(resume_epoch, EPOCHS):
    # 如果有数据采样器，则设置当前 epoch
    if data_sampler:
        data_sampler.set_epoch(epoch)

    # 遍历数据加载器
    for i, (text, images) in enumerate((dl if ENABLE_WEBDATASET else distr_dl)):
        # 每隔 10 步打印时间
        if i % 10 == 0 and is_root:
            t = time.time()

        # 如果启用了 fp16，将图像转换为半精度
        if args.fp16:
            images = images.half()

        # 将文本和图像移动到 GPU
        text, images = map(lambda t: t.cuda(), (text, images))

        # 计算损失
        loss = distr_dalle(text, images, return_loss=True)

        # 如果使用了 DeepSpeed
        if using_deepspeed:
            distr_dalle.backward(loss)
            distr_dalle.step()
            # 梯度在步骤后会自动清零
        else:
            loss.backward()
            clip_grad_norm_(distr_dalle.parameters(), GRAD_CLIP_NORM)
            distr_opt.step()
            distr_opt.zero_grad()

        # 计算集体损失，取平均值
        avg_loss = distr_backend.average_all(loss)

        log = {}

        # 每隔 10 步打印损失
        if i % 10 == 0 and is_root:
            print(epoch, i, f'loss - {avg_loss.item()}')

            log = {
                **log,
                'epoch': epoch,
                'iter': i,
                'loss': avg_loss.item()
            }

        # 每隔 SAVE_EVERY_N_STEPS 步保存模型
        if i % SAVE_EVERY_N_STEPS == 0:
            save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)

        # 每隔 100 步处理图像和日志
        if i % 100 == 0 and is_root:
            sample_text = text[:1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)

            if not avoid_model_calls:
                # 避免 CUDA 索引错误
                image = dalle.generate_images(text[:1], filter_thres=0.9)  # 使用 0.9 的 topk 抽样

            if not avoid_model_calls:
                log['image'] = wandb.Image(image, caption=decoded_text)

        # 每隔 10 步打印每秒样本数
        if i % 10 == 9 and is_root:
            sample_per_sec = BATCH_SIZE * 10 / (time.time() - t)
            log["sample_per_sec"] = sample_per_sec
            print(epoch, i, f'sample_per_sec - {sample_per_sec}')

        # 如果达到指定步数并启用了 FLOPS ���析器，则停止训练
        if i == 201 and args.flops_profiler:
            raise StopIteration("Profiler has finished running. Stopping training early.")

        # 如果是根节点，记录日志
        if is_root:
            wandb.log(log)

    # 如果启用了学习率衰减，根据平均损失调整学习率
    if LR_DECAY:
        distr_scheduler.step(avg_loss)

    # 每个 epoch 结束时保存模型
    save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)

    if is_root:
        # 每个 epoch 结束时将训练好的模型保存到 wandb 作为 artifact
        save_artifact(model_config, DALLE_OUTPUT_FILE_NAME)

# 最后保存模型
save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)

if is_root:
    # 保存训练好的模型到 wandb，并完成 wandb 日志
    wandb.save(DALLE_OUTPUT_FILE_NAME)
    save_artifact(model_config, DALLE_OUTPUT_FILE_NAME)
    wandb.finish()
```