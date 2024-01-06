# `transformer_vq\scripts\launch.py`

```
# 导入必要的库
import argparse  # 用于解析命令行参数
import sys  # 用于与 Python 解释器交互
import time  # 用于时间相关操作

import jax  # JAX 数值计算库
import jax.numpy as jnp  # JAX 数值计算库的 NumPy 接口
import numpy as np  # NumPy 数学库
import optax  # 用于优化器的库
import wandb  # 用于记录实验结果的库
from flax import jax_utils  # Flax 深度学习库的 JAX 工具
from flax.training import common_utils  # Flax 深度学习库的训练工具
from flax.training.train_state import TrainState  # Flax 深度学习库的训练状态

from transformer_vq.nn.model import Transformer  # 导入 Transformer 模型
from transformer_vq.nn.types import TransformerConfig  # 导入 Transformer 配置类型
from transformer_vq.nn.vq import VQSpec  # 导入 VQ 规范
from transformer_vq.ops.evaluate import eval_op  # 导入评估操作
from transformer_vq.ops.sample import sample_op  # 导入采样操作
from transformer_vq.ops.train import train_op  # 导入训练操作
from transformer_vq.utils.datasets import Dataset  # 导入数据集
# 从transformer_vq.utils.io模块中导入load_checkpoint函数
from transformer_vq.utils.io import load_checkpoint
# 从transformer_vq.utils.io模块中导入save_checkpoint函数
from transformer_vq.utils.io import save_checkpoint
# 从transformer_vq.utils.io模块中导入save_pixels函数
from transformer_vq.utils.io import save_pixels
# 从transformer_vq.utils.io模块中导入save_text函数
from transformer_vq.utils.io import save_text
# 从transformer_vq.utils.tree模块中导入flattened_traversal函数
from transformer_vq.utils.tree import flattened_traversal

# 定义常量DTYPES，包含字符串"bfloat16"和"float32"
DTYPES = ["bfloat16", "float32"]
# 定义常量COMMANDS，包含字符串"train_vocab", "train", "validation", "test", "sample", "bench"
COMMANDS = ["train_vocab", "train", "validation", "test", "sample", "bench"]
# 定义常量DATASETS，包含字符串"enwik8", "pg19", "imagenet64"
DATASETS = ["enwik8", "pg19", "imagenet64"]
# 定义常量OPTIMIZERS，包含字符串"adamw", "lion", "adafactor"
OPTIMIZERS = ["adamw", "lion", "adafactor"]

# 创建ArgumentParser对象，用于解析命令行参数
parser = argparse.ArgumentParser("Launch script for Transformer VQ experiments.")
# 添加命令行参数--multihost，类型为整数，帮助信息为"Multihost mode?"，默认值为0
parser.add_argument("--multihost", type=int, help="Multihost mode?", default=0)
# 添加命令行参数--command，选项为COMMANDS中的字符串
parser.add_argument("--command", choices=COMMANDS)
# 添加命令行参数--dataset，选项为DATASETS中的字符串
parser.add_argument("--dataset", choices=DATASETS)
# 添加命令行参数--data_dir，类型为字符串，帮助信息为"Download path"，默认值为None
parser.add_argument("--data_dir", type=str, help="Download path", default=None)
# 添加命令行参数--vocab_path，类型为字符串，帮助信息为"Sentencepiece path"，默认值为None
parser.add_argument("--vocab_path", type=str, help="Sentencepiece path", default=None)
# 添加命令行参数--prng_seed，类型为整数，帮助信息为"PRNG seed"
parser.add_argument("--prng_seed", type=int, help="PRNG seed")
# 添加命令行参数--global_batch_size，类型为整数，帮助信息为"Global batch size"
parser.add_argument("--global_batch_size", type=int, help="Global batch size")
# 添加一个名为“sequence_len”的命令行参数，类型为整数，用于指定序列长度T
parser.add_argument("--sequence_len", type=int, help="Sequence length T")

# 添加一个名为“update_len”的命令行参数，类型为整数，用于指定更新长度LK
parser.add_argument("--update_len", type=int, help="Update length LK")

# 添加一个名为“block_len”的命令行参数，类型为整数，用于指定块长度L
parser.add_argument("--block_len", type=int, help="Block length L")

# 添加一个名为“mem_len”的命令行参数，类型为整数，用于指定带长度M
parser.add_argument("--mem_len", type=int, help="Band length M")

# 添加一个名为“grad_thru_cache”的命令行参数，类型为整数，用于指定是否通过缓存进行反向传播（0/1）
parser.add_argument("--grad_thru_cache", type=int, help="Backprop thru cache (0/1)")

# 添加一个名为“agg_cache”的命令行参数，类型为整数，用于指定是否包括聚合缓存（0/1）
parser.add_argument("--agg_cache", type=int, help="Include aggregated cache (0/1)")

# 添加一个名为“param_dtype”的命令行参数，选项为预定义的数据类型，用于指定参数的数据类型
parser.add_argument("--param_dtype", choices=DTYPES, help="Dtype for parameters")

# 添加一个名为“dtype”的命令行参数，选项为预定义的数据类型，用于指定计算的数据类型
parser.add_argument("--dtype", choices=DTYPES, help="Dtype for computation")

# 添加一个名为“d_model”的命令行参数，类型为整数，用于指定模型宽度
parser.add_argument("--d_model", type=int, help="Model width")

# 添加一个名为“d_k”的命令行参数，类型为整数，用于指定键的宽度
parser.add_argument("--d_k", type=int, help="Key width")

# 添加一个名为“d_v”的命令行参数，类型为整数，用于指定值的宽度
parser.add_argument("--d_v", type=int, help="Value width")

# 添加一个名为“d_ff”的命令行参数，类型为整数，用于指定MLP的扩展宽度（如果使用MLP），默认值为0
parser.add_argument("--d_ff", type=int, help="Fan-out width, if using MLPs", default=0)

# 添加一个名为“n_head”的命令行参数，类型为整数，用于指定注意力头的数量
parser.add_argument("--n_head", type=int, help="Num attention heads")

# 添加一个名为“n_code”的命令行参数，类型为整数，用于指定每个头的编码数量
parser.add_argument("--n_code", type=int, help="Num codes per head")

# 添加一个名为“n_layer”的命令行参数，类型为整数，用于指定Transformer层的数量（每个GAU两个）
parser.add_argument("--n_layer", type=int, help="Num transformer layers (two GAU each)")

# 添加一个名为“pe_abs”的命令行参数，类型为整数，用于指定是否包括绝对位置嵌入（0/1）
parser.add_argument("--pe_abs", type=int, help="Include abs pos embs (0/1)")

# 添加一个名为“pe_lam”的命令行参数，类型为整数，用于指定最大角波长，默认值为100000
parser.add_argument("--pe_lam", type=int, help="Max angular wavelength", default=100000)

# 添加一个名为“p_dropemb”的命令行参数，类型为浮点数，用于指定嵌入的丢失率
parser.add_argument("--p_dropemb", type=float, help="Embedding dropout rate")

# 添加一个名为“p_dropsin”的命令行参数，类型为浮点数，用于指定相对位置编码正弦的丢失率
parser.add_argument("--p_dropsin", type=float, help="Rel PE sinusoid dropout rate")
# 添加一个参数，用于指定残差丢弃率
parser.add_argument("--p_dropres", type=float, help="Residual dropout rate")
# 添加一个参数，用于指定层丢弃率
parser.add_argument("--p_droplyr", type=float, help="LayerDrop rate")
# 添加一个参数，用于指定码书提交系数
parser.add_argument("--c_beta", type=float, help="Codebook commit coefficient")
# 添加一个参数，用于指定码书EMA率
parser.add_argument("--c_gamma", type=float, help="Codebook EMA rate")
# 添加一个参数，用于指定输出嵌入是否与输入嵌入绑定（0/1）
parser.add_argument("--e_tie", type=int, help="Output embs tied w input embs (0/1)")
# 添加一个参数，用于指定输出嵌入是否在LN之后应用（0/1）
parser.add_argument("--e_preln", type=int, help="Output embs applied after LN (0/1)")
# 添加一个参数，用于指定输出嵌入的比例因子
parser.add_argument("--e_scale", type=float, help="Output embs scale factor")

# 添加一个参数，用于指定梯度裁剪范数，默认为None
parser.add_argument("--grad_clip", type=float, help="Gradient clip norm", default=None)
# 添加一个参数，用于指定优化器名称
parser.add_argument("--optimizer", choices=OPTIMIZERS, help="Optimizer name")
# 添加一个参数，用于指定峰值学习率
parser.add_argument("--lr_max", type=float, help="Peak learning rate")
# 添加一个参数，用于指定学习率调度名称
parser.add_argument("--lr_schedule", type=str, help="Learning rate schedule name")
# 添加一个参数，用于指定解耦权重衰减
parser.add_argument("--wd_lam", type=float, help="Decoupled weight decay")
# 添加一个参数，用于指定采样过程中的核心截断
parser.add_argument("--p_nucleus", type=float, help="Nucleus cutoff during sampling")
# 添加一个参数，用于指定线性预热步数
parser.add_argument("--n_warmup_step", type=int, help="Linear warmup steps")
# 添加一个参数，用于指定最大步数
parser.add_argument("--n_max_step", type=int, help="Maximum step number")
# 添加一个参数，用于指定额外步数，在微调中使用 > 0
parser.add_argument("--n_extra_step", type=int, help="Extra steps, use > 0 in finetune")
# 添加一个参数，用于指定每次打印的步数，默认为200
parser.add_argument("--n_print_step", type=int, help="Steps per print", default=200)
# 添加一个参数，用于指定训练步数之间的评估阶段
parser.add_argument("--n_save_step", type=int, help="Train steps between eval phases")
# 添加一个参数，用于指定评估阶段的批次数
parser.add_argument("--n_eval_step", type=int, help="Batches per eval phase")
# 添加命令行参数，指定要保留的检查点数量，默认为5
parser.add_argument("--n_save_keep", type=int, help="Checkpoints to keep", default=5)
# 添加命令行参数，指定要加载检查点的目录
parser.add_argument("--in_checkpoint_dir", type=str, help="Checkpoint dir to load from")
# 添加命令行参数，指定要保存检查点的目录
parser.add_argument("--out_checkpoint_dir", type=str, help="Checkpoint dir to save to")
# 添加命令行参数，指定模型名称
parser.add_argument("--model_name", type=str, help="Model name")
# 添加命令行参数，用于记录日志的连续性，默认为None
parser.add_argument("--run_id", type=str, help="For logging continuity", default=None)
# 解析命令行参数
args = parser.parse_args()

# 如果设置了多主机模式，则初始化分布式计算环境
if args.multihost:
    jax.distributed.initialize()

# 定义打印内存信息的函数
def print_mem_info():
    # 获取当前计算后端
    backend = jax.lib.xla_bridge.get_backend()
    # 获取当前计算后端的活跃缓冲区数量
    n_bufs = len(backend.live_buffers())

    # 定义将缓冲区转换为字节的函数
    def tobytes(b):
        return np.prod(b.shape) * int(str(b.dtype)[-2:]) // 8

    # 计算当前计算后端所有活跃缓冲区的总字节数
    n_bytes = sum([tobytes(b) for b in backend.live_buffers()])
    # 打印活跃缓冲区的数量
    print(f"num_live_buffers: {n_bufs}")
    # 打印活跃字节数
    print(f"num_live_bytes: {n_bytes}")
    # 遍历活跃缓冲区，打印缓冲区的形状
    for i, buf in enumerate(backend.live_buffers()):
        # 根据优化器和是否绑定 embs，打印正确数量的项目
        if args.n_vocab in list(buf.shape):
            print(f"buffer_{i}.shape: {buf.shape}")


# 获取参数标签函数
def get_param_label_fn():
    # 返回扁平化遍历的结果，根据路径判断是 "main" 还是 "codebook"
    return flattened_traversal(
        lambda path, _: "main" if not path[-1].startswith("c_") else "codebook"
    )


# 获取调度函数
def get_schedule_fn():
    # 设置预热步数的线性调度
    warmup = optax.linear_schedule(0.0, 1.0, transition_steps=args.n_warmup_step)
    # 排除额外的、固定的微调学习率，计算出减去预热步数后的步数
    dec_steps = args.n_max_step - args.n_warmup_step  
    # 如果学习率调度为余弦衰减
    if args.lr_schedule == "cosine":
        # 设置主要调度为余弦衰减调度
        main_schedule = optax.cosine_decay_schedule(
            init_value=1.0,
# 根据参数设置不同的学习率衰减策略
if args.lr_schedule == "constant":
    # 使用常数学习率
    main_schedule = optax.constant_schedule(value=args.lr)
elif args.lr_schedule == "exponential":
    # 使用指数衰减学习率
    main_schedule = optax.exponential_schedule(
        init_value=args.lr,
        decay_steps=dec_steps,
        alpha=0.1,  # final schedule value
    )
elif args.lr_schedule == "inv_sqrt":
    # 使用倒数平方根衰减学习率
    main_schedule = optax.polynomial_schedule(
        init_value=1.0,
        end_value=0.1,
        power=-0.5,
        transition_steps=dec_steps,
    )
elif args.lr_schedule == "linear":
    # 使用线性衰减学习率
    main_schedule = optax.linear_schedule(
        init_value=1.0,
        end_value=0.1,
        transition_steps=dec_steps,
    )
else:
    # 抛出未实现的学习率衰减策略异常
    raise NotImplementedError("schedule name not supported")
# 返回联合的学习率衰减策略
return optax.join_schedules(
    schedules=[warmup, main_schedule], boundaries=[args.n_warmup_step]
# 获取优化器的函数
def get_optimizer():
    # 如果参数中指定优化器为"adamw"，则返回adamw优化器对象
    if args.optimizer == "adamw":
        return optax.adamw(
            learning_rate=args.lr_max,  # 学习率设定为args中的lr_max
            b1=0.9,  # adamw优化器的参数
            b2=0.98,  # adamw优化器的参数
            eps=10**-9,  # adamw优化器的参数
            mu_dtype=jnp.float32,  # 全精度，根据Rae等人建议
            weight_decay=0.0,  # 权重衰减为0，因为optax中的优化器会按照学习率自动缩放权重衰减
        )
    # 如果参数中指定优化器为"lion"，则返回lion优化器对象
    if args.optimizer == "lion":
        return optax.lion(
            learning_rate=args.lr_max,  # 学习率设定为args中的lr_max
            b1=0.95,  # lion优化器的参数
            b2=0.98,  # lion优化器的参数
            mu_dtype=jnp.bfloat16,  # bfloat16，根据Chen等人建议
            weight_decay=0.0,  # 权重衰减为0，因为optax中的优化器会按照学习率自动缩放权重衰减
    )
    # 如果优化器选择为"adafactor"，则返回adafactor优化器
    if args.optimizer == "adafactor":
        return optax.adafactor(
            learning_rate=args.lr_max,
            multiply_by_parameter_scale=True,
            clipping_threshold=1.0,  # 必须 >= 1.0，根据optax文档要求
            weight_decay_rate=0.0,  # optax中的优化器通过学习率缩放权重衰减，所以自定义为0
        )

# 获取优化器
def get_tx():
    maybe_gradclip = []
    # 如果梯度裁剪不为None，则添加梯度裁剪函数
    if args.grad_clip is not None:
        maybe_gradclip.append(optax.clip_by_global_norm(args.grad_clip))
    # 获取优化器
    optimizer = get_optimizer()
    # 获取调度器
    schedule = optax.scale_by_schedule(get_schedule_fn())
    maybe_decay = []
    # 如果权重衰减参数大于0.0，则添加权重衰减函数
    if args.wd_lam > 0.0:
        wd = optax.add_decayed_weights(
            weight_decay=-args.wd_lam,
# 定义一个函数，用于生成一个掩码，判断参数中的元素是否为一维数组
mask=lambda p: jax.tree_util.tree_map(lambda x: jnp.ndim(x) != 1, p),
# 将权重衰减参数添加到列表中
maybe_decay.append(wd)
# 创建一个多重变换，包括梯度裁剪、优化器、学习率调度和权重衰减
tx = optax.multi_transform(
    {
        "main": optax.chain(*maybe_gradclip, optimizer, schedule, *maybe_decay),
        "codebook": optax.sgd(learning_rate=1.0),
    },
    param_labels=get_param_label_fn(),
)
# 返回多重变换
return tx

# 定义一个函数，用于获取训练状态
def get_train_state(init_rng):
    # 创建一个配置字典，包括训练参数和是否为训练状态
    config = dict(**vars(args), is_train=True)
    # 移除配置字典中的"block_len"键值对
    _ = config.pop("block_len")
    # 将"block_len"设置为8
    config["block_len"] = 8
    # 根据配置创建一个TransformerConfig对象
    config = TransformerConfig.create(**config)
    # 创建一个Transformer模型
    model = Transformer(config)
    # 将初始随机数生成器分成4个子生成器
    sk1, sk2, sk3, sk4 = jax.random.split(init_rng, 4)
# 创建包含参数的字典，包括params、ephemeral和timeless
rngs = dict(params=sk1, ephemeral=sk2, timeless=sk3)
# 创建一个全零数组作为模型输入
inputs = jnp.zeros([1, config.block_len], dtype=jnp.int32)
# 创建一个全零数组作为文档ID
doc_ids = jnp.zeros([1, config.block_len], dtype=jnp.int32)
# 初始化Transformer模型的状态
state = Transformer.initial_state(config=config, batch_size=1)
# 创建VQSpec对象，包括设备数量、每次更新的块数量和损失掩码
vq_spec = VQSpec.create(
    n_device=jnp.array([1]),
    n_block_per_update=jnp.array([1]),
    loss_mask=jnp.ones([1, config.block_len], jnp.int32),
)
# 初始化模型参数
params = model.init(
    rngs,
    inputs=inputs,
    doc_ids=doc_ids,
    state=state,
    vq_spec=vq_spec,
)["params"].unfreeze()
# 获取事务对象
tx = get_tx()
# 如果命令不是"bench"，则打印参数数量
if args.command != "bench":
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"param count: {param_count}")
# 创建一个TrainState对象，其中apply_fn为None，params为给定的参数，tx为给定的tx
return TrainState.create(apply_fn=None, params=params, tx=tx)

# 确保参数满足一定的条件
assert args.sequence_len % args.update_len == 0
assert args.update_len % args.block_len == 0
assert args.sequence_len % args.block_len == 0
assert args.n_save_step >= (args.sequence_len // args.update_len)
assert args.n_save_step % (args.sequence_len // args.update_len) == 0
assert args.n_print_step % (args.sequence_len // args.update_len) == 0

# 在函数内部始终初始化和复制train_state，以避免对第二个（未更新的）train_state的悬空引用，当尝试训练大型模型时，这会浪费大量内存！
train_state, rng, start_step = state_setup()
train_state = jax.block_until_ready(jax_utils.replicate(train_state))
local_batch_size = args.global_batch_size // jax.process_count()
n_update = args.sequence_len // args.update_len

# 创建一个TransformerConfig对象，is_train为True，其余参数为args的变量
train_config = TransformerConfig.create(**vars(args), is_train=True)
# 从数据集中获取训练数据迭代器
train_iter = dataset.get_iter(
    split_name="train",  # 指定数据集的划分名称为训练集
    batch_size=local_batch_size,  # 指定批量大小
    sequence_len=args.sequence_len,  # 指定序列长度
)

# 初始化验证指标字典
val_metrics = dict()

# 初始化最佳验证集语言模型损失为正无穷
best_val_loss_lm = float("inf")

# 记录开始时间
start_time = time.perf_counter()

# 循环迭代训练步骤
for step in range(start_step, args.n_max_step + args.n_extra_step + 1, n_update):
    # 每隔一定步骤进行模型验证
    if step % args.n_save_step == 0:
        # 对验证集进行评估，获取验证指标
        val_metrics = evaluate(
            train_state=train_state,  # 训练状态
            dataset=dataset,  # 数据集
            split_name="validation",  # 指定数据集的划分名称为验证集
            p_eval_op=eval_op,  # 评估操作
            n_eval_step=args.n_eval_step,  # 每个主机的评估步骤
            persist=False,  # 是否持久化
        )
        # 记录最近一次验证集语言模型损失
        last_val_loss_lm = val_metrics["loss_lm_per_token"].tolist()
# 如果最佳验证集损失大于上一次的验证集损失，则更新最佳验证集损失值，并打印信息
if best_val_loss_lm > last_val_loss_lm:
    best_val_loss_lm = last_val_loss_lm
    print("val loss improved")
    # 如果步数大于起始步数，则保存检查点，并打印信息
    if step > start_step:
        print("saving checkpoint")
        save_checkpoint(
            target=jax_utils.unreplicate(train_state),
            save_dir=args.out_checkpoint_dir,
            prefix="checkpoint",
            step=step,
            keep=args.n_save_keep,
        )
# 从训练数据迭代器中获取下一个批次数据
batch = next(train_iter)
# 使用随机数生成器分割随机数种子
rng, batch_rng = jax.random.split(rng)
# 执行训练操作，更新训练状态和度量指标
train_state, metrics = p_train_op(
    train_config,
    train_state=train_state,
    batch=common_utils.shard(batch),
    rng=common_utils.shard_prng_key(batch_rng),
)
# 如果当前步数能被 args.n_print_step 整除，则执行以下操作
if step % args.n_print_step == 0:
    # 打印内存信息
    print_mem_info()
    # 取消复制 metrics
    metrics = jax_utils.unreplicate(metrics)
    # 等待 metrics 变量变为可用
    metrics = jax.block_until_ready(metrics)
    # 计算每个 token 的语言模型损失
    train_loss_lm_per_token = metrics["loss_lm_per_token_unscaled"]
    train_loss_lm_per_token /= metrics["loss_mask_per_token"]
    # 记录结束时间
    end_time = time.perf_counter()
    # 创建日志信息字典
    logging_info = dict(
        loss_lm_per_token=train_loss_lm_per_token,
        **metrics,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        step=step,
        secs_per_step=(end_time - start_time) / args.n_print_step,
    )
    # 打印当前训练步数
    print(train_state.step)
    # 打印输入数据的形状
    print(batch["inputs"].shape)
    # 打印目标数据的形状
    print(batch["targets"].shape)
    # 打印日志信息
    print(logging_info)
    # 如果是进程索引为 0，则记录日志信息到 wandb
    if jax.process_index() == 0:
        wandb.log(logging_info)
# 设置起始时间为结束时间
start_time = end_time
# 返回训练状态
return train_state

# 评估函数，用于评估模型在数据集上的性能
def evaluate(
    train_state,  # 训练状态
    dataset,  # 数据集
    split_name,  # 数据集划分名称
    p_eval_op,  # 评估操作
    n_eval_step=None,  # 评估步数
    persist=False,  # 是否持久化评估结果
):
    # 断言确保序列长度能够被块长度整除
    assert args.sequence_len % args.block_len == 0
    # 如果训练状态为空，则进行状态设置
    if train_state is None:
        train_state, rng, start_step = state_setup()
        train_state = jax.block_until_ready(jax_utils.replicate(train_state))
    # 获取当前步数
    step = int(jax_utils.unreplicate(train_state.step))
    # 计算本地批量大小
    local_batch_size = args.global_batch_size // jax.process_count()
    # 创建用于评估的配置
    eval_config = TransformerConfig.create(**vars(args), is_train=False)
# 从数据集中获取迭代器，用于遍历数据集
eval_iter = dataset.get_iter(
    split_name=split_name,  # 数据集划分的名称
    batch_size=local_batch_size,  # 批量大小
    sequence_len=args.sequence_len,  # 序列长度
)

# 初始化累加器
accumulator = None

# 遍历迭代器中的每个批次数据
for i, batch in enumerate(eval_iter):
    # 打印评估步骤信息
    print(f"eval step {i}...")
    
    # 执行评估操作，获取统计信息
    stats = p_eval_op(
        eval_config,  # 评估配置
        params=train_state.params,  # 训练状态参数
        batch=common_utils.shard(batch),  # 分片后的批次数据
    )
    
    # 等待评估结果准备就绪
    stats = jax.block_until_ready(stats)
    
    # 将统计信息中的数据类型转换为 jnp.float32
    stats = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), stats)
    
    # 如果累加器不为空，则将统计信息与累加器相加
    if accumulator is not None:
        accumulator = jax.tree_util.tree_map(lambda a, b: a + b, stats, accumulator)
    else:
        accumulator = stats  # 否则，将统计信息赋值给累加器
# 如果指定了评估步骤数，则在达到指定步骤时跳出循环
if n_eval_step is not None:
    if i + 1 == n_eval_step:
        break

# 计算每个标记的语言模型损失，除以每个标记的损失掩码
loss_lm_per_token = accumulator["loss_lm_per_token_unscaled"]
loss_lm_per_token /= accumulator["loss_mask_per_token"]

# 下面的计算假设评估操作对全局批次、块、标记进行平均，
# 并且评估数据在主机上复制，并且在每个主机上的设备上进行分片。
mult = local_batch_size * args.sequence_len
total_tokens = mult * accumulator["loss_mask_per_token"]

# 创建评估指标字典，包括每个标记的语言模型损失和总标记数
eval_metrics = dict(loss_lm_per_token=loss_lm_per_token, total_tokens=total_tokens)
# 等待评估指标计算完成
eval_metrics = jax.block_until_ready(jax_utils.unreplicate(eval_metrics))

# 如果需要持久化评估指标
if persist:
    save_kwargs = dict(
        prefix=f"{split_name}_loss_lm_per_token",
        save_dir=args.out_checkpoint_dir,
        step=step,
        keep=args.n_save_keep,
    )
    # 保存评估指标
    save_checkpoint(eval_metrics, **save_kwargs)
# 返回评估指标
def sample(dataset, p_sample_op, persist=False):
    # 设置训练状态、随机数生成器和起始步骤
    train_state, rng, start_step = state_setup()
    # 将步骤转换为整数
    step = int(train_state.step)
    # 将训练状态复制到设备上，并等待其完成
    train_state = jax.block_until_ready(jax_utils.replicate(train_state))
    # 计算本地批量大小
    local_batch_size = args.global_batch_size // jax.process_count()

    # 将参数转换为字典
    args_dict = vars(args)
    # 删除字典中的"block_len"键
    _ = args_dict.pop("block_len")
    # 创建采样配置
    sample_config = TransformerConfig.create(**args_dict, block_len=1, is_train=False)
    # 将随机数生成器分片，并等待其完成
    rng = common_utils.shard_prng_key(rng).block_until_ready()

    # 记录开始时间
    start_time = time.perf_counter()
    # 执行采样操作
    samples = p_sample_op(
        sample_config,
        dataset.vocab.eos_id,
        params=train_state.params,
        rng=rng,
    ).block_until_ready()  # 等待直到结果准备就绪，以确保时间差是正确的
    end_time = time.perf_counter()  # 获取程序运行结束时的性能计数器时间

    total_time = end_time - start_time  # 计算总运行时间
    if persist:  # 如果需要持久化
        samples = jnp.reshape(samples, [local_batch_size, args.sequence_len])  # 重新塑形样本数据
        save_fn = dict(text=save_text, image=save_pixels)[dataset.modality]  # 根据数据类型选择保存函数
        suffix = dict(text=".txt", image=".png")[dataset.modality]  # 根据数据类型选择文件后缀
        for i in range(local_batch_size):  # 遍历每个样本
            save_fn(  # 调用保存函数
                target=dataset.decode(samples[i]),  # 解码样本数据
                dirname=args.out_checkpoint_dir,  # 指定保存目录
                fname=f"samples_step{step}_proc{jax.process_index()}_item{i}{suffix}",  # 指定文件名
            )
    return dict(total_time=total_time)  # 返回总运行时间

def print_args_and_devices():  # 打印参数和设备信息
    if args.command != "bench":  # 如果命令不是"bench"
        print(jax.devices())  # 打印设备信息
# 打印本地设备列表
print(jax.local_devices())

# 如果命令不是"bench"，则遍历参数的键值对并打印
if args.command != "bench":
    for k, v in vars(args).items():
        print(f"{k}: {v}")

# 如果运行 ID 为空字符串，则将其设置为 None
if args.run_id == "":
    setattr(args, "run_id", None)

# 如果命令是"train"且 JAX 进程索引为 0，则初始化 wandb
if args.command == "train" and jax.process_index() == 0:
    wandb.init(
        project=args.model_name,
        config=vars(args),
        resume="never" if args.run_id is None else "must",
        id=args.run_id,
    )

# 获取数据集类并进行设置
dataset_cls = Dataset.registry.get(args.dataset)
# 使用给定的词汇表路径和数据目录创建数据集对象
dataset = dataset_cls(vocab_path=args.vocab_path, data_dir=args.data_dir)
# 如果命令是训练词汇表，则退出程序
if args.command == "train_vocab":
    sys.exit(0)
# 设置参数中的词汇表大小
setattr(args, "n_vocab", dataset.vocab_size)
# 返回数据集对象
return dataset

# 设置状态
def state_setup():
    # 创建同步随机数生成器
    synced_rng = jax.random.PRNGKey(args.prng_seed)
    # 将同步随机数生成器分割成两个子生成器
    synced_rng, init_rng = jax.random.split(synced_rng)
    # 获取训练状态
    train_state = get_train_state(init_rng)
    # 加载检查点
    train_state = load_checkpoint(
        train_state=train_state,
        load_dir=args.in_checkpoint_dir,
        prefix="checkpoint",
    )
    # 获取起始步数
    start_step = train_state.step
    # 如果命令不是"bench"，则打印起始步数
    if args.command != "bench":
        print(f"start_step: {start_step}")
    # 将起始步数折叠到同步随机数生成器中
    synced_rng = jax.random.fold_in(synced_rng, start_step)
# 使用jax.random.fold_in函数将同步的随机数生成器和当前进程的索引结合起来，生成一个不同步的随机数生成器
unsynced_rng = jax.random.fold_in(synced_rng, jax.process_index())
# 返回训练状态、不同步的随机数生成器和起始步数
return train_state, unsynced_rng, start_step

# 主函数
def main():
    # 打印参数和设备信息
    print_args_and_devices()
    # 设置wandb
    wandb_setup()
    # 设置数据集
    dataset = dataset_setup()

    # 如果命令是"train"
    if args.command == "train":
        # 训练数据集
        train(dataset=dataset, p_train_op=train_op)

    # 如果命令是"validation"或"test"
    elif args.command in {"validation", "test"}:
        # 评估指标
        eval_metrics = evaluate(
            train_state=None,
            dataset=dataset,
            split_name=args.command,
            p_eval_op=eval_op,
            persist=True,
        )
# 打印评估指标
print(eval_metrics)

# 如果命令是"sample"或"bench"，则执行以下操作
elif args.command in {"sample", "bench"}:
    # 创建包含数据集和采样操作的参数字典
    sample_kwargs = dict(
        dataset=dataset,
        p_sample_op=sample_op,
    )
    # 如果命令是"sample"，则执行采样操作并打印结果
    if args.command == "sample":
        outputs = sample(**sample_kwargs, persist=True)
        print(outputs)
    # 如果命令是"bench"，则执行基准测试
    else:
        # 为基准测试进行预热，排除 p_sample_op 的 JIT 编译时间
        outputs = sample(**sample_kwargs, persist=False)  # 冷启动
        outputs = sample(**sample_kwargs, persist=False)  # 热启动
        print(outputs["total_time"])

# 如果命令不是"eval"、"sample"或"bench"，则抛出值错误
else:
    raise ValueError(f"Operation {args.command} not implemented in main.")
# 如果当前脚本被直接执行而不是被导入，那么执行 main() 函数。
```