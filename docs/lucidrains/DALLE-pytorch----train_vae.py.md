# `.\lucidrains\DALLE-pytorch\train_vae.py`

```py
# 导入数学库
import math
# 从数学库中导入平方根函数
from math import sqrt
# 导入参数解析库
import argparse
# 从路径库中导入路径类
from pathlib import Path

# 导入 torch 库
import torch
# 从 torch 优化模块中导入 Adam 优化器
from torch.optim import Adam
# 从 torch 优化学习率调度模块中导入指数衰减学习率调度器
from torch.optim.lr_scheduler import ExponentialLR

# 导入视觉库
from torchvision import transforms as T
# 从 torch 工具数据模块中导入数据加载器
from torch.utils.data import DataLoader
# 从 torchvision 数据集模块中导入图像文件夹数据集类
from torchvision.datasets import ImageFolder
# 从 torchvision 工具模块中导入制作网格、保存图像的函数
from torchvision.utils import make_grid, save_image

# 导入 dalle_pytorch 类和工具
from dalle_pytorch import distributed_utils
from dalle_pytorch import DiscreteVAE

# 参数解析
parser = argparse.ArgumentParser()

# 添加图像文件夹路径参数
parser.add_argument('--image_folder', type=str, required=True,
                    help='path to your folder of images for learning the discrete VAE and its codebook')
# 添加图像大小参数
parser.add_argument('--image_size', type=int, required=False, default=128,
                    help='image size')

# 将参数解析器包装为分布式工具的参数解析器
parser = distributed_utils.wrap_arg_parser(parser)

# 训练参数组
train_group = parser.add_argument_group('Training settings')

# 添加训练轮数参数
train_group.add_argument('--epochs', type=int, default=20, help='number of epochs')
# 添加批量大小参数
train_group.add_argument('--batch_size', type=int, default=8, help='batch size')
# 添加学习率参数
train_group.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
# 添加学习率衰减率参数
train_group.add_argument('--lr_decay_rate', type=float, default=0.98, help='learning rate decay')
# 添加初始温度参数
train_group.add_argument('--starting_temp', type=float, default=1., help='starting temperature')
# 添加最小温度参数
train_group.add_argument('--temp_min', type=float, default=0.5, help='minimum temperature to anneal to')
# 添加退火率参数
train_group.add_argument('--anneal_rate', type=float, default=1e-6, help='temperature annealing rate')
# 添加保存图像数量参数
train_group.add_argument('--num_images_save', type=int, default=4, help='number of images to save')

# 模型参数组
model_group = parser.add_argument_group('Model settings')

# 添加图��令牌数量参数
model_group.add_argument('--num_tokens', type=int, default=8192, help='number of image tokens')
# 添加层数参数
model_group.add_argument('--num_layers', type=int, default=3, help='number of layers (should be 3 or above)')
# 添加残差网络块数量参数
model_group.add_argument('--num_resnet_blocks', type=int, default=2, help='number of residual net blocks')
# 添加平滑 L1 损失参数
model_group.add_argument('--smooth_l1_loss', dest='smooth_l1_loss', action='store_true')
# 添加嵌入维度参数
model_group.add_argument('--emb_dim', type=int, default=512, help='embedding dimension')
# 添加隐藏维度参数
model_group.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
# 添加 KL 损失权重参数
model_group.add_argument('--kl_loss_weight', type=float, default=0., help='KL loss weight')
# 添加透明度参数
model_group.add_argument('--transparent', dest='transparent', action='store_true')

# 解析参数
args = parser.parse_args()

# 常量

# 图像大小
IMAGE_SIZE = args.image_size
# 图像文件夹路径
IMAGE_PATH = args.image_folder

# 训练轮数
EPOCHS = args.epochs
# 批量大小
BATCH_SIZE = args.batch_size
# 学习率
LEARNING_RATE = args.learning_rate
# 学习率衰减率
LR_DECAY_RATE = args.lr_decay_rate

# 图像令牌数量
NUM_TOKENS = args.num_tokens
# 层数
NUM_LAYERS = args.num_layers
# 残差网络块数量
NUM_RESNET_BLOCKS = args.num_resnet_blocks
# 平滑 L1 损失
SMOOTH_L1_LOSS = args.smooth_l1_loss
# 嵌入维度
EMB_DIM = args.emb_dim
# 隐藏维度
HIDDEN_DIM = args.hidden_dim
# KL 损失权重
KL_LOSS_WEIGHT = args.kl_loss_weight

# 透明度
TRANSPARENT = args.transparent
# 通道数
CHANNELS = 4 if TRANSPARENT else 3
# 图像模式
IMAGE_MODE = 'RGBA' if TRANSPARENT else 'RGB'

# 初始温度
STARTING_TEMP = args.starting_temp
# 最小温度
TEMP_MIN = args.temp_min
# 退火率
ANNEAL_RATE = args.anneal_rate

# 保存图像数量
NUM_IMAGES_SAVE = args.num_images_save

# 初始化分布式后端
distr_backend = distributed_utils.set_backend_from_args(args)
distr_backend.initialize()

# 是否使用 DeepSpeed
using_deepspeed = distributed_utils.using_backend(distributed_utils.DeepSpeedBackend)

# 数据

# 创建图像文件夹数据集
ds = ImageFolder(
    IMAGE_PATH,
    T.Compose([
        # 将图像转换为指定模式
        T.Lambda(lambda img: img.convert(IMAGE_MODE) if img.mode != IMAGE_MODE else img),
        # 调整大小
        T.Resize(IMAGE_SIZE),
        # 中心裁剪
        T.CenterCrop(IMAGE_SIZE),
        # 转换为张量
        T.ToTensor()
    ])
)

if distributed_utils.using_backend(distributed_utils.HorovodBackend):
    # 创建一个用于分布式训练的数据采样器，用于在不同进程之间分配数据
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=distr_backend.get_world_size(),
        rank=distr_backend.get_rank())
# 如果条件不成立，将数据采样器设置为 None
else:
    data_sampler = None

# 创建数据加载器，设置批量大小、是否打乱数据、数据采样器
dl = DataLoader(ds, BATCH_SIZE, shuffle = not data_sampler, sampler=data_sampler)

# 定义 VAE 的参数
vae_params = dict(
    image_size = IMAGE_SIZE,
    num_layers = NUM_LAYERS,
    num_tokens = NUM_TOKENS,
    channels = CHANNELS,
    codebook_dim = EMB_DIM,
    hidden_dim   = HIDDEN_DIM,
    num_resnet_blocks = NUM_RESNET_BLOCKS
)

# 创建离散 VAE 模型
vae = DiscreteVAE(
    **vae_params,
    smooth_l1_loss = SMOOTH_L1_LOSS,
    kl_div_loss_weight = KL_LOSS_WEIGHT
)

# 如果不使用 DeepSpeed，则将 VAE 模型移到 GPU 上
if not using_deepspeed:
    vae = vae.cuda()

# 断言数据集中有数据
assert len(ds) > 0, 'folder does not contain any images'
if distr_backend.is_root_worker():
    # 打印找到的图片数量
    print(f'{len(ds)} images found for training')

# 优化器
opt = Adam(vae.parameters(), lr = LEARNING_RATE)
sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)

if distr_backend.is_root_worker():
    # weights & biases 实验跟踪
    import wandb

    model_config = dict(
        num_tokens = NUM_TOKENS,
        smooth_l1_loss = SMOOTH_L1_LOSS,
        num_resnet_blocks = NUM_RESNET_BLOCKS,
        kl_loss_weight = KL_LOSS_WEIGHT
    )

    # 初始化 weights & biases 实验
    run = wandb.init(
        project = 'dalle_train_vae',
        job_type = 'train_model',
        config = model_config
    )

# 分布式
distr_backend.check_batch_size(BATCH_SIZE)
deepspeed_config = {'train_batch_size': BATCH_SIZE}

# 分布式训练
(distr_vae, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
    args=args,
    model=vae,
    optimizer=opt,
    model_parameters=vae.parameters(),
    training_data=ds if using_deepspeed else dl,
    lr_scheduler=sched if not using_deepspeed else None,
    config_params=deepspeed_config,
)

using_deepspeed_sched = False
# 如果没有使用 DeepSpeed 调度器，则使用 sched
if distr_sched is None:
    distr_sched = sched
elif using_deepspeed:
    # 使用 DeepSpeed LR 调度器，并让 DeepSpeed 处理调度
    using_deepspeed_sched = True

# 保存模型
def save_model(path):
    save_obj = {
        'hparams': vae_params,
    }
    if using_deepspeed:
        cp_path = Path(path)
        path_sans_extension = cp_path.parent / cp_path.stem
        cp_dir = str(path_sans_extension) + '-ds-cp'

        # 保存 DeepSpeed 检查点
        distr_vae.save_checkpoint(cp_dir, client_state=save_obj)
        # 不返回以获取一个“正常”的检查点来参考

    if not distr_backend.is_root_worker():
        return

    save_obj = {
        **save_obj,
        'weights': vae.state_dict()
    }

    # 保存模型权重
    torch.save(save_obj, path)

# 设置初始温度
global_step = 0
temp = STARTING_TEMP

# 训练循环
for epoch in range(EPOCHS):
    # 遍历数据加载器中的图像数据和标签，使用enumerate获取索引和数据
    for i, (images, _) in enumerate(distr_dl):
        # 将图像数据移动到GPU上进行加速处理
        images = images.cuda()

        # 使用分布式VAE模型计算损失和重构图像
        loss, recons = distr_vae(
            images,
            return_loss = True,
            return_recons = True,
            temp = temp
        )

        # 如果使用DeepSpeed，则自动将梯度清零并执行优化步骤
        if using_deepspeed:
            # 梯度在步骤后自动清零
            distr_vae.backward(loss)
            distr_vae.step()
        else:
            # 否则手动将优化器梯度清零，计算梯度并执行优化步骤
            distr_opt.zero_grad()
            loss.backward()
            distr_opt.step()

        # 初始化日志字典
        logs = {}

        # 每100个迭代打印日志
        if i % 100 == 0:
            # 如果是根节点工作进程
            if distr_backend.is_root_worker():
                k = NUM_IMAGES_SAVE

                # 使用无梯度计算获取编码和硬重构图像
                with torch.no_grad():
                    codes = vae.get_codebook_indices(images[:k])
                    hard_recons = vae.decode(codes)

                # 截取部分图像和重构图像
                images, recons = map(lambda t: t[:k], (images, recons))
                # 将图像、重构图像、硬重构图像、编码转移到CPU并去除梯度信息
                images, recons, hard_recons, codes = map(lambda t: t.detach().cpu(), (images, recons, hard_recons, codes))
                # 将图像、重构图像、硬重构图像转换为图像网格
                images, recons, hard_recons = map(lambda t: make_grid(t.float(), nrow = int(sqrt(k)), normalize = True, range = (-1, 1)), (images, recons, hard_recons))

                # 更新日志字典
                logs = {
                    **logs,
                    'sample images':        wandb.Image(images, caption = 'original images'),
                    'reconstructions':      wandb.Image(recons, caption = 'reconstructions'),
                    'hard reconstructions': wandb.Image(hard_recons, caption = 'hard reconstructions'),
                    'codebook_indices':     wandb.Histogram(codes),
                    'temperature':          temp
                }

                # 保存模型
                wandb.save('./vae.pt')
            save_model(f'./vae.pt')

            # 温度退火

            temp = max(temp * math.exp(-ANNEAL_RATE * global_step), TEMP_MIN)

            # 学习率衰减

            # 不要从`deepspeed_config`中提前调整调度器
            if not using_deepspeed_sched:
                distr_sched.step()

        # 计算集合损失，取平均值
        avg_loss = distr_backend.average_all(loss)

        # 如果是根节点工作进程
        if distr_backend.is_root_worker():
            # 每10个迭代打印学习率和损失
            if i % 10 == 0:
                lr = distr_sched.get_last_lr()[0]
                print(epoch, i, f'lr - {lr:6f} loss - {avg_loss.item()}')

                # 更新日志字典
                logs = {
                    **logs,
                    'epoch': epoch,
                    'iter': i,
                    'loss': avg_loss.item(),
                    'lr': lr
                }

            # 记录日志
            wandb.log(logs)
        global_step += 1

    # 如果是根节点工作进程
    if distr_backend.is_root_worker():
        # 在每个epoch结束时将训练好的模型保存到wandb作为artifact

        model_artifact = wandb.Artifact('trained-vae', type = 'model', metadata = dict(model_config))
        model_artifact.add_file('vae.pt')
        run.log_artifact(model_artifact)
# 如果当前进程是根节点工作进程
if distr_backend.is_root_worker():
    # 保存最终的 VAE 模型并清理工作

    # 保存模型到文件 './vae-final.pt'
    save_model('./vae-final.pt')
    # 将模型文件上传到 wandb 服务器
    wandb.save('./vae-final.pt')

    # 创建一个 wandb Artifact 对象，用于存储训练好的 VAE 模型
    model_artifact = wandb.Artifact('trained-vae', type='model', metadata=dict(model_config))
    # 将 'vae-final.pt' 文件添加到 Artifact 对象中
    model_artifact.add_file('vae-final.pt')
    # 记录 Artifact 对象到当前运行日志中
    run.log_artifact(model_artifact)

    # 结束当前 wandb 运行
    wandb.finish()
```