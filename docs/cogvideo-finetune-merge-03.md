# CogVideo & CogVideoX 微调代码源码解析（四）



# Contribution Guide

There may still be many incomplete aspects in this project.

We look forward to your contributions to the repository in the following areas. If you complete the work mentioned above
and are willing to submit a PR and share it with the community, upon review, we
will acknowledge your contribution on the project homepage.

## Model Algorithms

- Support for model quantization inference (Int4 quantization project)
- Optimization of model fine-tuning data loading (replacing the existing decord tool)

## Model Engineering

- Model fine-tuning examples / Best prompt practices
- Inference adaptation on different devices (e.g., MLX framework)
- Any tools related to the model
- Any minimal fully open-source project using the CogVideoX open-source model

## Code Standards

Good code style is an art. We have prepared a `pyproject.toml` configuration file for the project to standardize code
style. You can organize the code according to the following specifications:

1. Install the `ruff` tool

```py
pip install ruff
```

Then, run the `ruff` tool

```py
ruff check tools sat inference
```

Check the code style. If there are issues, you can automatically fix them using the `ruff format` command.

```py
ruff format tools sat inference
```

Once your code meets the standard, there should be no errors.

## Naming Conventions
1. Please use English names, do not use Pinyin or other language names. All comments should be in English.
2. Please strictly follow the PEP8 specification and use underscores to separate words. Do not use names like a, b, c.



# コントリビューションガイド

本プロジェクトにはまだ多くの未完成の部分があります。

以下の分野でリポジトリへの貢献をお待ちしています。上記の作業を完了し、PRを提出してコミュニティと共有する意志がある場合、レビュー後、プロジェクトのホームページで貢献を認識します。

## モデルアルゴリズム

- モデル量子化推論のサポート (Int4量子化プロジェクト)
- モデルのファインチューニングデータロードの最適化（既存のdecordツールの置き換え）

## モデルエンジニアリング

- モデルのファインチューニング例 / 最適なプロンプトの実践
- 異なるデバイスでの推論適応（例: MLXフレームワーク）
- モデルに関連するツール
- CogVideoXオープンソースモデルを使用した、完全にオープンソースの最小プロジェクト

## コード標準

良いコードスタイルは一種の芸術です。本プロジェクトにはコードスタイルを標準化するための `pyproject.toml`
設定ファイルを用意しています。以下の仕様に従ってコードを整理してください。

1. `ruff` ツールをインストールする

```py
pip install ruff
```

次に、`ruff` ツールを実行します

```py
ruff check tools sat inference
```

コードスタイルを確認します。問題がある場合は、`ruff format` コマンドを使用して自動修正できます。

```py
ruff format tools sat inference
```

コードが標準に準拠したら、エラーはなくなるはずです。

## 命名規則

1. 英語名を使用してください。ピンインや他の言語の名前を使用しないでください。すべてのコメントは英語で記載してください。
2. PEP8仕様に厳密に従い、単語をアンダースコアで区切ってください。a、b、cのような名前は使用しないでください。


# 贡献指南

本项目可能还存在很多不完善的内容。 我们期待您在以下方面与我们共建仓库, 如果您完成了上述工作并愿意PR和分享到社区，在通过审核后，我们将在项目首页感谢您的贡献。

## 模型算法

- 模型量化推理支持 (Int4量化工程)
- 模型微调数据载入优化支持(替换现有的decord工具)

## 模型工程

- 模型微调示例 / 最佳提示词实践
- 不同设备上的推理适配(MLX等框架)
- 任何模型周边工具
- 任何使用CogVideoX开源模型制作的最小完整开源项目

## 代码规范

良好的代码风格是一种艺术，我们已经为项目准备好了`pyproject.toml`配置文件，用于规范代码风格。您可以按照以下规范梳理代码:

1. 安装`ruff`工具

```py
pip install ruff
```

接着，运行`ruff`工具

```py
ruff check tools sat inference
```

检查代码风格，如果有问题，您可以通过`ruff format .`命令自动修复。

```py
ruff format tools sat inference
```

如果您的代码符合规范，应该不会出现任何的错误。

## 命名规范

- 请使用英文命名，不要使用拼音或者其他语言命名。所有的注释均使用英文。
- 请严格遵循 PEP8 规范，使用下划线分割单词。请勿使用 a,b,c 这样的命名。

## CogVideoX-5B

Videos 1-8:

1. A garden comes to life as a kaleidoscope of butterflies flutters amidst the blossoms, their delicate wings casting shadows on the petals below. In the background, a grand fountain cascades water with a gentle splendor, its rhythmic sound providing a soothing backdrop. Beneath the cool shade of a mature tree, a solitary wooden chair invites solitude and reflection, its smooth surface worn by the touch of countless visitors seeking a moment of tranquility in nature's embrace.

2. A small boy, head bowed and determination etched on his face, sprints through the torrential downpour as闪电 crackles and 雷鸣 rumbles in the distance. The relentless rain pounds the ground, creating a chaotic dance of water droplets that mirror the Dramatic sky's anger. In the far background, the silhouette of a cozy home beckons, a faint beacon of safety and warmth amidst the fierce weather. The scene is one of perseverance and the unyielding spirit of a child braving the elements.

3. A suited astronaut, with the red dust of Mars clinging to their boots, reaches out to shake hands with an alien being, their skin a shimmering blue, under the pink-tinged sky of the fourth planet. In the background, a sleek silver rocket, a beacon of human ingenuity, stands tall, its engines powered down, as the two representatives of different worlds exchange a historic greeting amidst the desolate beauty of the Martian landscape.

4. An elderly gentleman, with a serene expression, sits at the water's edge, a steaming cup of tea by his side. He is engrossed in his artwork, brush in hand, as he renders an oil painting on a canvas that's propped up against a small, weathered table. The sea breeze whispers through his silver hair, gently billowing his loose-fitting white shirt, while the salty air adds an intangible element to his masterpiece in progress. The scene is one of tranquility and inspiration, with the artist's canvas capturing the vibrant hues of the setting sun reflecting off the tranquil sea.

5. In a dimly lit bar, purplish light bathes the face of a mature man, his eyes blinking thoughtfully as he ponders in close-up, the background artfully blurred to focus on his introspective expression, the ambiance of the bar a mere suggestion of shadows and soft lighting.

6. A golden retriever, sporting sleek black sunglasses, with its lengthy fur flowing in the breeze, sprints playfully across a rooftop terrace, recently refreshed by a light rain. The scene unfolds from a distance, the dog's energetic bounds growing larger as it approaches the camera, its tail wagging with unrestrained joy, while droplets of water glisten on the concrete behind it. The overcast sky provides a dramatic backdrop, emphasizing the vibrant golden coat of the canine as it dashes towards the viewer.

7. On a brilliant sunny day, the lakeshore is lined with an array of willow trees, their slender branches swaying gently in the soft breeze. The tranquil surface of the lake reflects the clear blue sky, while several elegant swans glide gracefully through the still water, leaving behind delicate ripples that disturb the mirror-like quality of the lake. The scene is one of serene beauty, with the willows' greenery providing a picturesque frame for the peaceful avian visitors.

8. A Chinese mother, draped in a soft, pastel-colored robe, gently rocks back and forth in a cozy rocking chair positioned in the tranquil setting of a nursery. The dimly lit bedroom is adorned with whimsical mobiles dangling from the ceiling, casting shadows that dance on the walls. Her baby, swaddled in a delicate, patterned blanket, rests against her chest, the child's earlier cries now replaced by contented coos as the mother's soothing voice lulls the little one to sleep. The scent of lavender fills the air, adding to the serene atmosphere, while a warm, orange glow from a nearby nightlight illuminates the scene with a gentle hue, capturing a moment of tender love and comfort.

## CogVideoX-2B

Videos 1-4: 

1. A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.

2. The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from its tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds.

3. A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall.

4. In the haunting backdrop of a war-torn city, where ruins and crumbled walls tell a story of devastation, a poignant close-up frames a young girl. Her face is smudged with ash, a silent testament to the chaos around her. Her eyes glistening with a mix of sorrow and resilience, capturing the raw emotion of a world that has lost its innocence to the ravages of conflict.


<div align="center">
<img src=wechat.jpg width="60%"/>

<p> 扫码关注公众号，加入「 CogVideoX 交流群」 </p>
<p> Scan the QR code to follow the official account and join the "CogVLM Discussion Group" </p>
</div>



# `.\cogvideo-finetune\sat\arguments.py`

```py
# 导入所需的库和模块
import argparse  # 用于处理命令行参数
import os  # 提供与操作系统交互的功能
import torch  # PyTorch库，用于深度学习
import json  # 用于处理JSON数据
import warnings  # 用于发出警告
import omegaconf  # 用于配置管理
from omegaconf import OmegaConf  # 从omegaconf导入OmegaConf类
from sat.helpers import print_rank0  # 从sat.helpers导入print_rank0函数
from sat import mpu  # 导入sat模块中的mpu部分
from sat.arguments import set_random_seed  # 从sat.arguments导入设置随机种子的函数
from sat.arguments import add_training_args, add_evaluation_args, add_data_args  # 导入添加参数的函数
import torch.distributed  # 导入PyTorch分布式训练功能

def add_model_config_args(parser):
    """Model arguments"""  # 函数说明：添加模型参数配置

    group = parser.add_argument_group("model", "model configuration")  # 创建模型参数组
    group.add_argument("--base", type=str, nargs="*", help="config for input and saving")  # 添加基本配置参数
    group.add_argument(
        "--model-parallel-size", type=int, default=1, help="size of the model parallel. only use if you are an expert."
    )  # 添加模型并行大小参数
    group.add_argument("--force-pretrain", action="store_true")  # 添加强制预训练标志
    group.add_argument("--device", type=int, default=-1)  # 添加设备参数，默认为-1
    group.add_argument("--debug", action="store_true")  # 添加调试标志
    group.add_argument("--log-image", type=bool, default=True)  # 添加日志图像参数，默认为True

    return parser  # 返回更新后的解析器

def add_sampling_config_args(parser):
    """Sampling configurations"""  # 函数说明：添加采样配置参数

    group = parser.add_argument_group("sampling", "Sampling Configurations")  # 创建采样参数组
    group.add_argument("--output-dir", type=str, default="samples")  # 添加输出目录参数
    group.add_argument("--input-dir", type=str, default=None)  # 添加输入目录参数，默认为None
    group.add_argument("--input-type", type=str, default="cli")  # 添加输入类型参数，默认为"cli"
    group.add_argument("--input-file", type=str, default="input.txt")  # 添加输入文件参数，默认为"input.txt"
    group.add_argument("--final-size", type=int, default=2048)  # 添加最终尺寸参数，默认为2048
    group.add_argument("--sdedit", action="store_true")  # 添加sdedit标志
    group.add_argument("--grid-num-rows", type=int, default=1)  # 添加网格行数参数，默认为1
    group.add_argument("--force-inference", action="store_true")  # 添加强制推理标志
    group.add_argument("--lcm_steps", type=int, default=None)  # 添加最小公倍数步骤参数，默认为None
    group.add_argument("--sampling-num-frames", type=int, default=32)  # 添加采样帧数参数，默认为32
    group.add_argument("--sampling-fps", type=int, default=8)  # 添加采样帧率参数，默认为8
    group.add_argument("--only-save-latents", type=bool, default=False)  # 添加仅保存潜变量标志，默认为False
    group.add_argument("--only-log-video-latents", type=bool, default=False)  # 添加仅记录视频潜变量标志，默认为False
    group.add_argument("--latent-channels", type=int, default=32)  # 添加潜变量通道数参数，默认为32
    group.add_argument("--image2video", action="store_true")  # 添加图像转视频标志

    return parser  # 返回更新后的解析器

def get_args(args_list=None, parser=None):
    """Parse all the args."""  # 函数说明：解析所有参数
    if parser is None:  # 检查解析器是否为None
        parser = argparse.ArgumentParser(description="sat")  # 创建新的解析器
    else:
        assert isinstance(parser, argparse.ArgumentParser)  # 确保解析器是argparse.ArgumentParser的实例
    parser = add_model_config_args(parser)  # 添加模型参数配置
    parser = add_sampling_config_args(parser)  # 添加采样参数配置
    parser = add_training_args(parser)  # 添加训练参数
    parser = add_evaluation_args(parser)  # 添加评估参数
    parser = add_data_args(parser)  # 添加数据参数

    import deepspeed  # 导入DeepSpeed库

    parser = deepspeed.add_config_arguments(parser)  # 添加DeepSpeed配置参数

    args = parser.parse_args(args_list)  # 解析命令行参数
    args = process_config_to_args(args)  # 处理配置并转换为参数

    if not args.train_data:  # 检查是否指定训练数据
        print_rank0("No training data specified", level="WARNING")  # 打印警告信息

    assert (args.train_iters is None) or (args.epochs is None), "only one of train_iters and epochs should be set."  # 确保train_iters和epochs只能有一个被设置
    # 检查训练迭代次数和周期是否为 None
        if args.train_iters is None and args.epochs is None:
            # 如果两者均为 None，设置默认训练迭代次数为 10000
            args.train_iters = 10000  # default 10k iters
            # 输出警告信息，提示使用默认的 10k 迭代
            print_rank0("No train_iters (recommended) or epochs specified, use default 10k iters.", level="WARNING")
    
        # 检查 CUDA 是否可用，并设置 args.cuda
        args.cuda = torch.cuda.is_available()
    
        # 从环境变量获取 RANK，并转为整数，默认为 0
        args.rank = int(os.getenv("RANK", "0"))
        # 从环境变量获取 WORLD_SIZE，并转为整数，默认为 1
        args.world_size = int(os.getenv("WORLD_SIZE", "1"))
        # 如果 local_rank 为 None，从环境变量获取 LOCAL_RANK，并转为整数，默认为 0
        if args.local_rank is None:
            args.local_rank = int(os.getenv("LOCAL_RANK", "0"))  # torchrun
    
        # 如果 device 设置为 -1，进行设备选择
        if args.device == -1:
            # 如果没有可用的 CUDA 设备，设置为 CPU
            if torch.cuda.device_count() == 0:
                args.device = "cpu"
            # 如果 local_rank 不为 None，使用 local_rank 作为设备
            elif args.local_rank is not None:
                args.device = args.local_rank
            # 否则，使用 rank 对设备数量取模
            else:
                args.device = args.rank % torch.cuda.device_count()
    
        # 如果 local_rank 不等于 device，且模式不是推理，则抛出错误
        if args.local_rank != args.device and args.mode != "inference":
            raise ValueError(
                "LOCAL_RANK (default 0) and args.device inconsistent. "
                "This can only happens in inference mode. "
                "Please use CUDA_VISIBLE_DEVICES=x for single-GPU training. "
            )
    
        # 如果 rank 为 0，输出当前的 world size
        if args.rank == 0:
            print_rank0("using world size: {}".format(args.world_size))
    
        # 如果训练数据权重不为 None，检查其长度是否与训练数据一致
        if args.train_data_weights is not None:
            assert len(args.train_data_weights) == len(args.train_data)
    
        # 如果模式不是推理，进行 DeepSpeed 训练配置
        if args.mode != "inference":  # training with deepspeed
            args.deepspeed = True
            # 如果 DeepSpeed 配置未指定，构造配置路径
            if args.deepspeed_config is None:  # not specified
                deepspeed_config_path = os.path.join(
                    os.path.dirname(__file__), "training", f"deepspeed_zero{args.zero_stage}.json"
                )
                # 打开 DeepSpeed 配置文件并加载内容
                with open(deepspeed_config_path) as file:
                    args.deepspeed_config = json.load(file)
                # 标记需要覆盖配置
                override_deepspeed_config = True
            else:
                override_deepspeed_config = False
    
        # 确保不能同时指定 fp16 和 bf16
        assert not (args.fp16 and args.bf16), "cannot specify both fp16 and bf16."
    
        # 如果 zero_stage 大于 0，且未指定 fp16 或 bf16，则自动设置 fp16 为 True
        if args.zero_stage > 0 and not args.fp16 and not args.bf16:
            print_rank0("Automatically set fp16=True to use ZeRO.")
            args.fp16 = True
            args.bf16 = False
    # 检查是否启用 DeepSpeed
    if args.deepspeed:
        # 检查是否启用检查点激活
        if args.checkpoint_activations:
            # 启用 DeepSpeed 激活检查点
            args.deepspeed_activation_checkpointing = True
        else:
            # 禁用 DeepSpeed 激活检查点
            args.deepspeed_activation_checkpointing = False
        # 如果 DeepSpeed 配置不为 None，获取配置
        if args.deepspeed_config is not None:
            deepspeed_config = args.deepspeed_config

        # 如果覆盖 DeepSpeed 配置，则使用 args
        if override_deepspeed_config:  # not specify deepspeed_config, use args
            # 如果启用 FP16
            if args.fp16:
                deepspeed_config["fp16"]["enabled"] = True
            # 如果启用 BF16
            elif args.bf16:
                deepspeed_config["bf16"]["enabled"] = True
                deepspeed_config["fp16"]["enabled"] = False
            else:
                # 禁用 FP16
                deepspeed_config["fp16"]["enabled"] = False
            # 设置每个 GPU 的微批大小
            deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
            # 设置梯度累积步骤
            deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            # 获取优化器参数配置
            optimizer_params_config = deepspeed_config["optimizer"]["params"]
            # 设置学习率
            optimizer_params_config["lr"] = args.lr
            # 设置权重衰减
            optimizer_params_config["weight_decay"] = args.weight_decay
        else:  # override args with values in deepspeed_config
            # 如果当前是主进程，打印信息
            if args.rank == 0:
                print_rank0("Will override arguments with manually specified deepspeed_config!")
            # 检查 FP16 配置
            if "fp16" in deepspeed_config and deepspeed_config["fp16"]["enabled"]:
                args.fp16 = True
            else:
                # 禁用 FP16
                args.fp16 = False
            # 检查 BF16 配置
            if "bf16" in deepspeed_config and deepspeed_config["bf16"]["enabled"]:
                args.bf16 = True
            else:
                # 禁用 BF16
                args.bf16 = False
            # 获取每个 GPU 的微批大小
            if "train_micro_batch_size_per_gpu" in deepspeed_config:
                args.batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
            # 获取梯度累积步骤
            if "gradient_accumulation_steps" in deepspeed_config:
                args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
            else:
                # 如果没有设置，梯度累积步骤为 None
                args.gradient_accumulation_steps = None
            # 检查优化器配置
            if "optimizer" in deepspeed_config:
                optimizer_params_config = deepspeed_config["optimizer"].get("params", {})
                # 设置学习率
                args.lr = optimizer_params_config.get("lr", args.lr)
                # 设置权重衰减
                args.weight_decay = optimizer_params_config.get("weight_decay", args.weight_decay)
        # 更新 DeepSpeed 配置到 args
        args.deepspeed_config = deepspeed_config

    # 初始化分布式和随机种子，因为这似乎总是必要的
    initialize_distributed(args)
    # 为当前进程设置种子
    args.seed = args.seed + mpu.get_data_parallel_rank()
    # 设置随机种子
    set_random_seed(args.seed)
    # 返回更新后的 args
    return args
# 初始化分布式训练，使用 torch.distributed
def initialize_distributed(args):
    """Initialize torch.distributed."""
    # 检查分布式是否已初始化
    if torch.distributed.is_initialized():
        # 检查模型并行是否已初始化
        if mpu.model_parallel_is_initialized():
            # 如果模型并行大小与先前配置不一致，抛出错误
            if args.model_parallel_size != mpu.get_model_parallel_world_size():
                raise ValueError(
                    "model_parallel_size is inconsistent with prior configuration."
                    "We currently do not support changing model_parallel_size."
                )
            # 如果一致，返回 False
            return False
        else:
            # 如果模型并行大小大于 1，发出警告
            if args.model_parallel_size > 1:
                warnings.warn(
                    "model_parallel_size > 1 but torch.distributed is not initialized via SAT."
                    "Please carefully make sure the correctness on your own."
                )
            # 初始化模型并行
            mpu.initialize_model_parallel(args.model_parallel_size)
        # 返回 True，表示初始化成功
        return True
    # 自动设备分配已转移到 arguments.py
    if args.device == "cpu":
        # 如果设备是 CPU，什么也不做
        pass
    else:
        # 设置当前 CUDA 设备
        torch.cuda.set_device(args.device)
    # 设置初始化方法
    init_method = "tcp://"
    # 获取主节点 IP，默认是 localhost
    args.master_ip = os.getenv("MASTER_ADDR", "localhost")

    # 如果世界大小为 1，获取一个可用端口
    if args.world_size == 1:
        from sat.helpers import get_free_port

        default_master_port = str(get_free_port())
    else:
        # 否则设置默认端口为 6000
        default_master_port = "6000"
    # 获取主节点端口，优先使用环境变量
    args.master_port = os.getenv("MASTER_PORT", default_master_port)
    # 构造初始化方法的完整地址
    init_method += args.master_ip + ":" + args.master_port
    # 初始化进程组，设置后端、世界大小、当前进程的排名和初始化方法
    torch.distributed.init_process_group(
        backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method
    )

    # 设置模型并行和数据并行的通信器
    mpu.initialize_model_parallel(args.model_parallel_size)

    # 将 VAE 上下文并行组设置为模型并行组
    from sgm.util import set_context_parallel_group, initialize_context_parallel

    # 如果模型并行大小小于等于 2，设置上下文并行组
    if args.model_parallel_size <= 2:
        set_context_parallel_group(args.model_parallel_size, mpu.get_model_parallel_group())
    else:
        # 否则初始化上下文并行
        initialize_context_parallel(2)
    # mpu.initialize_model_parallel(1)
    # 可选的 DeepSpeed 激活检查点功能
    if args.deepspeed:
        import deepspeed

        # 初始化 DeepSpeed 分布式
        deepspeed.init_distributed(
            dist_backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method
        )
        # # 配置检查点，即使不使用也似乎没有负面影响
        # deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    # 否则分支，表示处于模型仅模式，不初始化 deepspeed，但仍需初始化随机数跟踪器
        else:
            # 在模型仅模式下，不初始化 deepspeed，但需要初始化 rng 跟踪器，以便在 dropout 时保存种子
            try:
                # 尝试导入 deepspeed 模块
                import deepspeed
                # 从 deepspeed 导入激活检查点相关的 RNG 跟踪器
                from deepspeed.runtime.activation_checkpointing.checkpointing import (
                    _CUDA_RNG_STATE_TRACKER,
                    _MODEL_PARALLEL_RNG_TRACKER_NAME,
                )
    
                # 将默认种子 1 添加到 CUDA RNG 状态跟踪器
                _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 1)  # default seed 1
            except Exception as e:
                # 如果发生异常，从 sat.helpers 导入打印函数
                from sat.helpers import print_rank0
    
                # 输出异常信息，级别为 DEBUG
                print_rank0(str(e), level="DEBUG")
    
        # 返回 True，表示执行成功
        return True
# 处理配置文件，将参数提取到 args 中
def process_config_to_args(args):
    """Fetch args from only --base"""  # 文档字符串，说明该函数从 --base 参数中获取配置

    # 加载每个配置文件并将其转换为 OmegaConf 对象，形成列表
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    # 合并多个配置对象，生成一个综合配置
    config = OmegaConf.merge(*configs)

    # 从合并的配置中提取 "args" 部分，默认为一个空的 OmegaConf 对象
    args_config = config.pop("args", OmegaConf.create())
    # 遍历 args_config 中的每一个键
    for key in args_config:
        # 检查当前键的值是否为字典配置或列表配置
        if isinstance(args_config[key], omegaconf.DictConfig) or isinstance(args_config[key], omegaconf.ListConfig):
            # 将 OmegaConf 对象转换为普通 Python 对象
            arg = OmegaConf.to_object(args_config[key])
        else:
            # 否则直接获取该键的值
            arg = args_config[key]
        # 如果 args 对象中有该键，则设置其值
        if hasattr(args, key):
            setattr(args, key, arg)

    # 如果配置中包含 "model" 键，提取其值并设置到 args 中
    if "model" in config:
        model_config = config.pop("model", OmegaConf.create())  # 从配置中移除 "model"，默认为空 OmegaConf 对象
        args.model_config = model_config  # 将模型配置存储到 args 中
    # 如果配置中包含 "deepspeed" 键，提取其值并转换为对象
    if "deepspeed" in config:
        deepspeed_config = config.pop("deepspeed", OmegaConf.create())  # 从配置中移除 "deepspeed"，默认为空 OmegaConf 对象
        args.deepspeed_config = OmegaConf.to_object(deepspeed_config)  # 转换为普通对象并存储
    # 如果配置中包含 "data" 键，提取其值
    if "data" in config:
        data_config = config.pop("data", OmegaConf.create())  # 从配置中移除 "data"，默认为空 OmegaConf 对象
        args.data_config = data_config  # 将数据配置存储到 args 中

    # 返回更新后的 args 对象
    return args
```

# `.\cogvideo-finetune\sat\data_video.py`

```py
# 导入io模块，处理输入输出操作
import io
# 导入os模块，进行操作系统相关的功能
import os
# 导入sys模块，访问与Python解释器相关的变量和函数
import sys
# 从functools模块导入partial，用于创建部分函数应用
from functools import partial
# 导入math模块，进行数学计算
import math
# 导入torchvision.transforms模块，进行图像变换
import torchvision.transforms as TT
# 从sgm.webds模块导入MetaDistributedWebDataset，进行分布式数据集处理
from sgm.webds import MetaDistributedWebDataset
# 导入random模块，进行随机数生成
import random
# 从fractions模块导入Fraction，处理有理数
from fractions import Fraction
# 从typing模块导入Union、Optional、Dict、Any和Tuple，进行类型注解
from typing import Union, Optional, Dict, Any, Tuple
# 从torchvision.io.video导入av，处理视频输入输出
from torchvision.io.video import av
# 导入numpy库，进行数组和矩阵操作
import numpy as np
# 导入torch库，进行深度学习操作
import torch
# 从torchvision.io导入_video_opt，处理视频选项
from torchvision.io import _video_opt
# 从torchvision.io.video导入多个函数，用于视频处理
from torchvision.io.video import _check_av_available, _read_from_stream, _align_audio_frames
# 从torchvision.transforms.functional导入center_crop和resize，进行图像裁剪和调整大小
from torchvision.transforms.functional import center_crop, resize
# 从torchvision.transforms导入InterpolationMode，处理插值模式
from torchvision.transforms import InterpolationMode
# 导入decord库，进行视频读取
import decord
# 从decord模块导入VideoReader，读取视频
from decord import VideoReader
# 从torch.utils.data导入Dataset，构建数据集类
from torch.utils.data import Dataset


# 定义读取视频的函数，返回视频帧和音频帧
def read_video(
    filename: str,  # 视频文件的路径
    start_pts: Union[float, Fraction] = 0,  # 视频开始的展示时间
    end_pts: Optional[Union[float, Fraction]] = None,  # 视频结束的展示时间
    pts_unit: str = "pts",  # 展示时间的单位，默认为'pts'
    output_format: str = "THWC",  # 输出视频张量的格式
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    从文件中读取视频，返回视频帧和音频帧

    参数:
        filename (str): 视频文件的路径
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            视频的开始展示时间
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            结束展示时间
        pts_unit (str, optional): start_pts和end_pts值的单位,
            可以是'pts'或'sec'。默认为'pts'。
        output_format (str, optional): 输出视频张量的格式，可以是'THWC'（默认）或'TCHW'。

    返回:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): 视频帧
        aframes (Tensor[K, L]): 音频帧，其中K为通道数，L为点数
        info (Dict): 视频和音频的元数据。可以包含video_fps（float）和audio_fps（int）字段
    """

    # 将输出格式转换为大写
    output_format = output_format.upper()
    # 检查输出格式是否有效
    if output_format not in ("THWC", "TCHW"):
        # 如果无效，抛出错误
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    # 检查AV（音频视频）是否可用
    _check_av_available()

    # 如果结束时间点为空，则设置为无穷大
    if end_pts is None:
        end_pts = float("inf")

    # 检查结束时间点是否大于开始时间点
    if end_pts < start_pts:
        # 如果不满足条件，抛出错误
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")

    # 初始化信息字典，用于存储视频和音频的元数据
    info = {}
    # 初始化音频帧列表，用于存储音频数据
    audio_frames = []
    # 设置音频时间基准
    audio_timebase = _video_opt.default_timebase
    # 使用指定的文件名打开音频/视频容器，并忽略元数据错误
    with av.open(filename, metadata_errors="ignore") as container:
        # 检查容器中是否有音频流
        if container.streams.audio:
            # 获取音频流的时间基准
            audio_timebase = container.streams.audio[0].time_base
        # 检查容器中是否有视频流
        if container.streams.video:
            # 从视频流中读取指定时间范围内的帧
            video_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.video[0],  # 指定视频流
                {"video": 0},  # 额外参数，指示视频流索引
            )
            # 获取视频流的平均帧率
            video_fps = container.streams.video[0].average_rate
            # 防止潜在的损坏文件导致错误
            if video_fps is not None:
                # 将视频帧率保存到信息字典中
                info["video_fps"] = float(video_fps)

        # 再次检查音频流
        if container.streams.audio:
            # 从音频流中读取指定时间范围内的帧
            audio_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.audio[0],  # 指定音频流
                {"audio": 0},  # 额外参数，指示音频流索引
            )
            # 将音频帧率保存到信息字典中
            info["audio_fps"] = container.streams.audio[0].rate

    # 将音频帧转换为 NumPy 数组格式
    aframes_list = [frame.to_ndarray() for frame in audio_frames]

    # 创建一个空的张量，用于存放视频帧
    vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    # 如果有音频帧
    if aframes_list:
        # 将音频帧列表沿着第一个维度拼接
        aframes = np.concatenate(aframes_list, 1)
        # 将 NumPy 数组转换为 PyTorch 张量
        aframes = torch.as_tensor(aframes)
        # 如果时间单位是秒，将开始和结束时间点转换为帧数
        if pts_unit == "sec":
            start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
            # 如果结束时间点不是无穷大，进行转换
            if end_pts != float("inf"):
                end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
        # 对齐音频帧
        aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        # 如果没有音频帧，创建一个空的音频张量
        aframes = torch.empty((1, 0), dtype=torch.float32)

    # 如果输出格式为 TCHW
    if output_format == "TCHW":
        # 将张量维度从 [T,H,W,C] 转换为 [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    # 返回视频帧、音频帧和信息字典
    return vframes, aframes, info
# 根据给定的图像尺寸调整数组，以便进行矩形裁剪
def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    # 判断数组的宽高比是否大于目标图像的宽高比
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        # 调整数组大小，使其适应目标图像的宽度，保持高度比例
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        # 调整数组大小，使其适应目标图像的高度，保持宽度比例
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    # 获取调整后数组的高度和宽度
    h, w = arr.shape[2], arr.shape[3]
    # 移除数组的第一个维度
    arr = arr.squeeze(0)

    # 计算高度和宽度的差值
    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    # 根据重塑模式确定裁剪的起始位置
    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        # 计算中心裁剪的起始位置
        top, left = delta_h // 2, delta_w // 2
    else:
        # 如果重塑模式不支持，则抛出异常
        raise NotImplementedError
    # 从数组中裁剪出指定区域
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    # 返回裁剪后的数组
    return arr


# 填充视频的最后一帧，以确保帧数达到指定数量
def pad_last_frame(tensor, num_frames):
    # 检查当前帧数是否少于指定帧数
    if len(tensor) < num_frames:
        # 计算需要填充的帧数
        pad_length = num_frames - len(tensor)
        # 使用最后一帧进行填充而不是使用零帧
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        # 将原始帧和填充帧拼接在一起
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        return padded_tensor
    else:
        # 如果帧数足够，返回前指定数量的帧
        return tensor[:num_frames]


# 加载视频并根据参数进行采样
def load_video(
    video_data,
    sampling="uniform",
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    # 设置视频读取的桥接方式为torch
    decord.bridge.set_bridge("torch")
    # 创建视频读取器对象
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    # 确定要读取的原始视频长度
    if nb_read_frames is not None:
        ori_vlen = nb_read_frames
    else:
        ori_vlen = min(int(duration * actual_fps) - 1, len(vr))

    # 计算最大寻址位置
    max_seek = int(ori_vlen - skip_frms_num - num_frames / wanted_fps * actual_fps)
    # 随机选择起始帧
    start = random.randint(skip_frms_num, max_seek + 1)
    # 计算结束帧
    end = int(start + num_frames / wanted_fps * actual_fps)
    n_frms = num_frames

    # 如果采样模式为均匀，生成采样索引
    if sampling == "uniform":
        indices = np.arange(start, end, (end - start) / n_frms).astype(int)
    else:
        # 如果采样模式不支持，则抛出异常
        raise NotImplementedError

    # 从视频读取器中获取一批帧
    temp_frms = vr.get_batch(np.arange(start, end))
    # 确保获取的帧不为空
    assert temp_frms is not None
    # 将获取的帧转换为张量
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    # 根据索引提取需要的帧
    tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]

    # 返回填充后的帧
    return pad_last_frame(tensor_frms, num_frames)


import threading


# 在子线程中加载视频并设置超时
def load_video_with_timeout(*args, **kwargs):
    video_container = {}

    # 定义目标函数，用于在子线程中执行视频加载
    def target_function():
        video = load_video(*args, **kwargs)
        video_container["video"] = video

    # 创建并启动新线程
    thread = threading.Thread(target=target_function)
    thread.start()
    # 设置超时为20秒
    timeout = 20
    # 等待线程执行完成，最多等待指定时间
    thread.join(timeout)
    # 检查线程是否仍在运行
        if thread.is_alive():
            # 如果线程还在运行，打印超时信息
            print("Loading video timed out")
            # 抛出超时异常
            raise TimeoutError
        # 从视频容器中获取视频数据，如果不存在则返回 None，并确保数据是连续的
        return video_container.get("video", None).contiguous()
# 定义处理视频的函数，接收多个参数以配置视频处理
def process_video(
    video_path,  # 视频文件路径或字节流
    image_size=None,  # 可选参数，处理后的图像大小
    duration=None,  # 可选参数，已知持续时间以加快处理
    num_frames=4,  # 希望处理的帧数，默认为4
    wanted_fps=None,  # 可选参数，期望的帧率
    actual_fps=None,  # 可选参数，实际的帧率
    skip_frms_num=0.0,  # 忽略的帧数，避免过渡帧
    nb_read_frames=None,  # 可选参数，读取的帧数
):
    """
    video_path: str or io.BytesIO  # 视频路径或字节流类型
    image_size: .  # 图像大小的描述
    duration: preknow the duration to speed up by seeking to sampled start. TODO by_pass if unknown.  # 预先知道持续时间以加快处理
    num_frames: wanted num_frames.  # 希望的帧数
    wanted_fps: .  # 期望的帧率描述
    skip_frms_num: ignore the first and the last xx frames, avoiding transitions.  # 忽略首尾帧以避免过渡
    """

    # 调用函数加载视频，设置超时处理
    video = load_video_with_timeout(
        video_path,  # 视频路径
        duration=duration,  # 视频持续时间
        num_frames=num_frames,  # 希望的帧数
        wanted_fps=wanted_fps,  # 期望的帧率
        actual_fps=actual_fps,  # 实际的帧率
        skip_frms_num=skip_frms_num,  # 忽略的帧数
        nb_read_frames=nb_read_frames,  # 读取的帧数
    )

    # --- 复制并修改图像处理 ---
    video = video.permute(0, 3, 1, 2)  # 将视频的维度顺序调整为 [时间, 通道, 高, 宽]

    # 如果指定了图像大小，则进行调整
    if image_size is not None:
        video = resize_for_rectangle_crop(video, image_size, reshape_mode="center")  # 调整视频尺寸

    return video  # 返回处理后的视频


# 定义处理视频数据的函数
def process_fn_video(src, image_size, fps, num_frames, skip_frms_num=0.0, txt_key="caption"):
    while True:  # 无限循环以处理数据源中的每个项
        r = next(src)  # 获取下一个视频数据项
        if "mp4" in r:  # 如果数据项包含 mp4 格式
            video_data = r["mp4"]  # 提取 mp4 视频数据
        elif "avi" in r:  # 如果数据项包含 avi 格式
            video_data = r["avi"]  # 提取 avi 视频数据
        else:  # 如果没有视频数据
            print("No video data found")  # 输出提示信息
            continue  # 继续下一个循环

        # 检查文本键是否存在
        if txt_key not in r:
            txt = ""  # 如果不存在，设置为空字符串
        else:
            txt = r[txt_key]  # 否则提取文本信息

        # 如果文本是字节类型，则解码为字符串
        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")  # 解码字节为 UTF-8 字符串
        else:
            txt = str(txt)  # 转换为字符串类型

        # 尝试获取视频的持续时间
        duration = r.get("duration", None)  # 获取持续时间，默认为 None
        if duration is not None:  # 如果持续时间存在
            duration = float(duration)  # 转换为浮点数
        else:
            continue  # 如果不存在，则跳过当前循环

        # 尝试获取视频的实际帧率
        actual_fps = r.get("fps", None)  # 获取实际帧率，默认为 None
        if actual_fps is not None:  # 如果实际帧率存在
            actual_fps = float(actual_fps)  # 转换为浮点数
        else:
            continue  # 如果不存在，则跳过当前循环

        # 计算所需帧数和持续时间
        required_frames = num_frames / fps * actual_fps + 2 * skip_frms_num  # 计算所需帧数
        required_duration = num_frames / fps + 2 * skip_frms_num / actual_fps  # 计算所需持续时间

        # 如果视频持续时间小于所需持续时间，则跳过
        if duration is not None and duration < required_duration:
            continue  # 跳过当前循环

        try:
            # 调用处理视频函数
            frames = process_video(
                io.BytesIO(video_data),  # 将视频数据转换为字节流
                num_frames=num_frames,  # 设置希望处理的帧数
                wanted_fps=fps,  # 设置期望的帧率
                image_size=image_size,  # 设置图像大小
                duration=duration,  # 设置视频持续时间
                actual_fps=actual_fps,  # 设置实际帧率
                skip_frms_num=skip_frms_num,  # 设置忽略的帧数
            )
            frames = (frames - 127.5) / 127.5  # 进行帧归一化处理
        except Exception as e:  # 捕获处理过程中的任何异常
            print(e)  # 输出异常信息
            continue  # 继续下一个循环

        # 创建包含处理结果的字典
        item = {
            "mp4": frames,  # 存储处理后的视频帧
            "txt": txt,  # 存储相关文本
            "num_frames": num_frames,  # 存储希望的帧数
            "fps": fps,  # 存储帧率
        }

        yield item  # 生成处理后的项目


# 定义视频数据集类，继承自 MetaDistributedWebDataset
class VideoDataset(MetaDistributedWebDataset):
    # 初始化方法，构造类的实例
        def __init__(
            self,
            path,  # 数据路径
            image_size,  # 图像尺寸
            num_frames,  # 帧数
            fps,  # 每秒帧数
            skip_frms_num=0.0,  # 跳过的帧数，默认为0.0
            nshards=sys.maxsize,  # 分片数，默认为系统最大值
            seed=1,  # 随机种子，默认为1
            meta_names=None,  # 元数据名称，默认为None
            shuffle_buffer=1000,  # 打乱缓冲区大小，默认为1000
            include_dirs=None,  # 包含的目录，默认为None
            txt_key="caption",  # 文本键，默认为"caption"
            **kwargs,  # 其他额外参数
        ):
            # 如果种子为-1，则生成一个随机种子
            if seed == -1:
                seed = random.randint(0, 1000000)
            # 如果元数据名称为空，则设置为空列表
            if meta_names is None:
                meta_names = []
    
            # 如果路径以";"开头，则将其分割为路径和包含目录
            if path.startswith(";"):
                path, include_dirs = path.split(";", 1)
            # 调用父类的初始化方法，传递必要的参数
            super().__init__(
                path,  # 传递的数据路径
                partial(  # 使用偏函数包装处理视频的函数
                    process_fn_video, num_frames=num_frames, image_size=image_size, fps=fps, skip_frms_num=skip_frms_num
                ),
                seed,  # 随机种子
                meta_names=meta_names,  # 元数据名称
                shuffle_buffer=shuffle_buffer,  # 打乱缓冲区大小
                nshards=nshards,  # 分片数
                include_dirs=include_dirs,  # 包含的目录
            )
    
        # 类方法，用于创建数据集的实例
        @classmethod
        def create_dataset_function(cls, path, args, **kwargs):
            # 返回类的实例，使用给定的路径和其他参数
            return cls(path, **kwargs)
# 定义一个 SFTDataset 类，继承自 Dataset
class SFTDataset(Dataset):
    # 初始化方法，接收数据目录、视频尺寸、帧率、最大帧数和跳过的帧数
    def __init__(self, data_dir, video_size, fps, max_num_frames, skip_frms_num=3):
        """
        skip_frms_num: 忽略前面和后面的 xx 帧，避免过渡效果。
        """
        # 调用父类的初始化方法
        super(SFTDataset, self).__init__()

        # 设置视频尺寸
        self.video_size = video_size
        # 设置帧率
        self.fps = fps
        # 设置最大帧数
        self.max_num_frames = max_num_frames
        # 设置跳过的帧数
        self.skip_frms_num = skip_frms_num

        # 初始化视频路径列表
        self.video_paths = []
        # 初始化字幕列表
        self.captions = []

        # 遍历数据目录，获取所有视频文件
        for root, dirnames, filenames in os.walk(data_dir):
            # 遍历当前目录下的所有文件
            for filename in filenames:
                # 检查文件是否以 ".mp4" 结尾
                if filename.endswith(".mp4"):
                    # 获取视频文件的完整路径
                    video_path = os.path.join(root, filename)
                    # 将视频路径添加到列表中
                    self.video_paths.append(video_path)

                    # 构造对应的字幕文件路径
                    caption_path = video_path.replace(".mp4", ".txt").replace("videos", "labels")
                    # 检查字幕文件是否存在
                    if os.path.exists(caption_path):
                        # 如果存在，读取第一行作为字幕
                        caption = open(caption_path, "r").read().splitlines()[0]
                    else:
                        # 如果不存在，字幕为空字符串
                        caption = ""
                    # 将字幕添加到列表中
                    self.captions.append(caption)
    # 定义获取视频帧的方法，支持通过索引访问
    def __getitem__(self, index):
        # 设置桥接库为 Torch
        decord.bridge.set_bridge("torch")

        # 根据索引获取视频文件路径
        video_path = self.video_paths[index]
        # 创建视频读取器，设置高度和宽度为 -1 表示自动
        vr = VideoReader(uri=video_path, height=-1, width=-1)
        # 获取视频的实际帧率
        actual_fps = vr.get_avg_fps()
        # 获取视频的原始帧数
        ori_vlen = len(vr)

        # 如果视频帧数与目标帧率计算后超过最大帧数限制
        if ori_vlen / actual_fps * self.fps > self.max_num_frames:
            # 将帧数限制为最大帧数
            num_frames = self.max_num_frames
            # 计算起始帧
            start = int(self.skip_frms_num)
            # 计算结束帧
            end = int(start + num_frames / self.fps * actual_fps)
            # 确保结束帧不超过原始帧数
            end_safty = min(int(start + num_frames / self.fps * actual_fps), int(ori_vlen))
            # 生成采样索引
            indices = np.arange(start, end, (end - start) // num_frames).astype(int)
            # 从视频读取器中获取帧数据
            temp_frms = vr.get_batch(np.arange(start, end_safty))
            # 确保获取到的帧数据不为空
            assert temp_frms is not None
            # 将帧数据转换为张量，如果已经是张量则保持不变
            tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
            # 根据采样索引选择相应的帧数据
            tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
        else:
            # 如果原始帧数大于最大帧数限制
            if ori_vlen > self.max_num_frames:
                # 将帧数限制为最大帧数
                num_frames = self.max_num_frames
                # 计算起始帧
                start = int(self.skip_frms_num)
                # 计算结束帧
                end = int(ori_vlen - self.skip_frms_num)
                # 生成采样索引
                indices = np.arange(start, end, max((end - start) // num_frames, 1)).astype(int)
                # 从视频读取器中获取全部帧数据
                temp_frms = vr.get_batch(np.arange(start, end))
                # 确保获取到的帧数据不为空
                assert temp_frms is not None
                # 将帧数据转换为张量，如果已经是张量则保持不变
                tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
                # 根据采样索引选择相应的帧数据
                tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
            else:
                # 定义一个函数用于计算小于等于 n 的 4k+1 的最近值
                def nearest_smaller_4k_plus_1(n):
                    remainder = n % 4
                    if remainder == 0:
                        return n - 3  # 返回 n 减去 3
                    else:
                        return n - remainder + 1  # 返回 n 减去余数再加 1

                # 计算起始帧
                start = int(self.skip_frms_num)
                # 计算结束帧
                end = int(ori_vlen - self.skip_frms_num)
                # 根据函数获取符合条件的帧数
                num_frames = nearest_smaller_4k_plus_1(end - start)  # 3D VAE 需要帧数为 4k+1
                # 重新计算结束帧
                end = int(start + num_frames)
                # 从视频读取器中获取帧数据
                temp_frms = vr.get_batch(np.arange(start, end))
                # 确保获取到的帧数据不为空
                assert temp_frms is not None
                # 将帧数据转换为张量，如果已经是张量则保持不变
                tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms

        # 填充最后一帧以满足最大帧数要求
        tensor_frms = pad_last_frame(
            tensor_frms, self.max_num_frames
        )  # 因为索引的长度可能小于帧数，需处理四舍五入误差
        # 调整张量维度，从 [T, H, W, C] 转为 [T, C, H, W]
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        # 对张量进行中心矩形裁剪处理
        tensor_frms = resize_for_rectangle_crop(tensor_frms, self.video_size, reshape_mode="center")
        # 归一化处理，将像素值转换到 [-1, 1]
        tensor_frms = (tensor_frms - 127.5) / 127.5

        # 构建返回的字典，包含处理后的帧数据、对应的文本和帧数及帧率
        item = {
            "mp4": tensor_frms,
            "txt": self.captions[index],
            "num_frames": num_frames,
            "fps": self.fps,
        }
        # 返回处理后的项
        return item

    # 定义获取视频路径数量的方法
    def __len__(self):
        # 返回视频路径的数量
        return len(self.video_paths)

    # 定义一个类方法
    @classmethod
    # 创建数据集的类方法，接收路径和其他参数
    def create_dataset_function(cls, path, args, **kwargs):
        # 实例化类，传入数据目录路径和额外参数
        return cls(data_dir=path, **kwargs)
```

# `.\cogvideo-finetune\sat\diffusion_video.py`

```py
# 导入随机数生成模块
import random

# 导入数学模块
import math
# 导入类型提示相关模块
from typing import Any, Dict, List, Tuple, Union
# 导入OmegaConf中的ListConfig类
from omegaconf import ListConfig
# 导入PyTorch中的功能模块
import torch.nn.functional as F

# 从sat.helpers模块导入print_rank0函数
from sat.helpers import print_rank0
# 导入PyTorch库
import torch
# 从PyTorch中导入nn模块
from torch import nn

# 从sgm.modules导入UNCONDITIONAL_CONFIG
from sgm.modules import UNCONDITIONAL_CONFIG
# 从sgm.modules.autoencoding.temporal_ae导入VideoDecoder类
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
# 从sgm.modules.diffusionmodules.wrappers导入OPENAIUNETWRAPPER
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
# 从sgm.util导入多个实用函数
from sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)
# 导入垃圾回收模块
import gc
# 从sat导入mpu模块
from sat import mpu


# 定义SATVideoDiffusionEngine类，继承自nn.Module
class SATVideoDiffusionEngine(nn.Module):
    # 禁用不可训练参数的方法
    def disable_untrainable_params(self):
        # 初始化可训练参数的总数
        total_trainable = 0
        # 遍历模型的所有参数
        for n, p in self.named_parameters():
            # 如果参数不可训练，跳过
            if p.requires_grad == False:
                continue
            # 初始化标志
            flag = False
            # 检查参数名是否以不可训练前缀开头
            for prefix in self.not_trainable_prefixes:
                if n.startswith(prefix) or prefix == "all":
                    flag = True
                    break

            # 定义LoRA前缀列表
            lora_prefix = ["matrix_A", "matrix_B"]
            # 检查参数名中是否包含LoRA前缀
            for prefix in lora_prefix:
                if prefix in n:
                    flag = False
                    break

            # 如果标志为真，禁用参数训练
            if flag:
                p.requires_grad_(False)
            else:
                # 统计可训练参数的数量
                total_trainable += p.numel()

        # 打印可训练参数的总数
        print_rank0("***** Total trainable parameters: " + str(total_trainable) + " *****")

    # 重初始化方法
    def reinit(self, parent_model=None):
        # 重新加载之前训练模块的初始参数
        # 可以通过parent_model.get_mixin()访问其他混合模型
        pass

    # 初始化第一个阶段的方法
    def _init_first_stage(self, config):
        # 根据配置实例化模型并设置为评估模式
        model = instantiate_from_config(config).eval()
        # 禁用训练模式
        model.train = disabled_train
        # 禁用模型参数的训练
        for param in model.parameters():
            param.requires_grad = False
        # 设置第一个阶段模型
        self.first_stage_model = model

    # 前向传播方法
    def forward(self, x, batch):
        # 计算损失
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        # 计算损失的平均值
        loss_mean = loss.mean()
        # 创建损失字典
        loss_dict = {"loss": loss_mean}
        # 返回平均损失和损失字典
        return loss_mean, loss_dict

    # 向第一帧添加噪声的方法
    def add_noise_to_first_frame(self, image):
        # 生成噪声标准差
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(self.device)
        # 将标准差转化为指数形式
        sigma = torch.exp(sigma).to(image.dtype)
        # 生成与图像同样形状的随机噪声
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        # 将噪声添加到图像中
        image = image + image_noise
        # 返回添加噪声后的图像
        return image
    # 处理共享步骤，接收一个批次的输入数据，返回损失值及其字典
        def shared_step(self, batch: Dict) -> Any:
            # 获取输入数据
            x = self.get_input(batch)
            # 如果学习率缩放因子不为空
            if self.lr_scale is not None:
                # 对输入进行下采样
                lr_x = F.interpolate(x, scale_factor=1 / self.lr_scale, mode="bilinear", align_corners=False)
                # 还原到原始大小
                lr_x = F.interpolate(lr_x, scale_factor=self.lr_scale, mode="bilinear", align_corners=False)
                # 编码下采样后的输入
                lr_z = self.encode_first_stage(lr_x, batch)
                # 将编码结果存入批次字典
                batch["lr_input"] = lr_z
    
            # 调整维度以便后续处理
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            # 如果使用带噪声的图像输入
            if self.noised_image_input:
                # 取出第一个帧作为图像
                image = x[:, :, 0:1]
                # 对图像添加噪声
                image = self.add_noise_to_first_frame(image)
                # 编码添加噪声的图像
                image = self.encode_first_stage(image, batch)
    
            # 编码输入数据
            x = self.encode_first_stage(x, batch)
            # 调整维度以便后续处理
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            # 如果使用带噪声的图像输入
            if self.noised_image_input:
                # 调整噪声图像的维度
                image = image.permute(0, 2, 1, 3, 4).contiguous()
                # 如果所有噪声图像需要拼接
                if self.noised_image_all_concat:
                    # 重复图像以匹配输入
                    image = image.repeat(1, x.shape[1], 1, 1, 1)
                else:
                    # 拼接零填充的张量
                    image = torch.concat([image, torch.zeros_like(x[:, 1:])], dim=1)
                # 根据概率决定是否丢弃图像
                if random.random() < self.noised_image_dropout:
                    image = torch.zeros_like(image)
                # 将拼接的图像存入批次字典
                batch["concat_images"] = image
    
            # 收集垃圾
            gc.collect()
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
            # 计算损失及其字典
            loss, loss_dict = self(x, batch)
            # 返回损失及字典
            return loss, loss_dict
    
        # 从批次中获取输入，转化为指定类型
        def get_input(self, batch):
            return batch[self.input_key].to(self.dtype)
    
        # 无梯度上下文中解码第一阶段
        @torch.no_grad()
        def decode_first_stage(self, z):
            # 对潜在变量进行缩放
            z = 1.0 / self.scale_factor * z
            # 计算每次解码的样本数
            n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
            # 计算轮数
            n_rounds = math.ceil(z.shape[0] / n_samples)
            all_out = []
            # 使用自动混合精度进行解码
            with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
                # 循环解码每个批次
                for n in range(n_rounds):
                    # 如果解码器是视频解码器
                    if isinstance(self.first_stage_model.decoder, VideoDecoder):
                        kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                    else:
                        kwargs = {}
                    # 解码当前批次
                    out = self.first_stage_model.decode(z[n * n_samples : (n + 1) * n_samples], **kwargs)
                    # 将解码结果添加到输出列表
                    all_out.append(out)
            # 合并所有解码结果
            out = torch.cat(all_out, dim=0)
            # 返回解码输出
            return out
    
        # 无梯度上下文中编码第一阶段
        @torch.no_grad()
        def encode_first_stage(self, x, batch):
            # 获取帧数
            frame = x.shape[2]
    
            # 如果帧数大于1且输入为潜在变量
            if frame > 1 and self.latent_input:
                # 调整维度
                x = x.permute(0, 2, 1, 3, 4).contiguous()
                # 返回已编码的输入
                return x * self.scale_factor  # already encoded
    
            # 计算每次编码的样本数
            n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
            # 计算轮数
            n_rounds = math.ceil(x.shape[0] / n_samples)
            all_out = []
            # 使用自动混合精度进行编码
            with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
                # 循环编码每个批次
                for n in range(n_rounds):
                    # 编码当前批次
                    out = self.first_stage_model.encode(x[n * n_samples : (n + 1) * n_samples])
                    # 将编码结果添加到输出列表
                    all_out.append(out)
            # 合并所有编码结果
            z = torch.cat(all_out, dim=0)
            # 对编码结果进行缩放
            z = self.scale_factor * z
            # 返回编码输出
            return z
    
        # 无梯度上下文中
    # 定义一个样本生成函数，接受条件、超参数和其他可选参数
    def sample(
        self,
        cond: Dict,  # 输入条件的字典
        uc: Union[Dict, None] = None,  # 可选的无条件输入，默认为 None
        batch_size: int = 16,  # 每次生成的样本数量，默认为 16
        shape: Union[None, Tuple, List] = None,  # 样本形状，默认为 None
        prefix=None,  # 可选的前缀，用于生成的样本
        concat_images=None,  # 用于连接的图像，默认为 None
        **kwargs,  # 其他关键字参数
    ):
        # 生成一个随机的高斯噪声张量，形状为 (batch_size, *shape)，并转为 float32 类型
        randn = torch.randn(batch_size, *shape).to(torch.float32).to(self.device)
        # 如果对象有已设置的噪声，则用该噪声处理 randn
        if hasattr(self, "seeded_noise"):
            randn = self.seeded_noise(randn)

        # 如果前缀不为 None，将前缀与 randn 进行拼接
        if prefix is not None:
            randn = torch.cat([prefix, randn[:, prefix.shape[1] :]], dim=1)

        # 获取模型并行的世界大小，用于广播噪声
        mp_size = mpu.get_model_parallel_world_size()
        # 如果模型并行的世界大小大于 1
        if mp_size > 1:
            # 计算当前全局 rank 和源节点
            global_rank = torch.distributed.get_rank() // mp_size
            src = global_rank * mp_size
            # 广播 randn 到模型并行组的所有节点
            torch.distributed.broadcast(randn, src=src, group=mpu.get_model_parallel_group())

        scale = None  # 初始化缩放因子为 None
        scale_emb = None  # 初始化缩放嵌入为 None

        # 定义去噪器的 lambda 函数，使用模型进行去噪
        denoiser = lambda input, sigma, c, **addtional_model_inputs: self.denoiser(
            self.model, input, sigma, c, concat_images=concat_images, **addtional_model_inputs
        )

        # 调用采样器生成样本，传入去噪器、随机噪声和条件等参数
        samples = self.sampler(denoiser, randn, cond, uc=uc, scale=scale, scale_emb=scale_emb)
        # 将生成的样本转换为指定的数据类型
        samples = samples.to(self.dtype)
        # 返回生成的样本
        return samples

    # 使用 torch.no_grad 装饰器，禁止梯度计算以节省内存
    @torch.no_grad()
    # 定义日志记录函数，用于记录不同的条件
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        定义记录不同条件的启发式方法。
        这些可以是字符串列表（文本到图像）、张量、整数等。
        """
        # 获取输入图像的高度和宽度
        image_h, image_w = batch[self.input_key].shape[3:]
        log = dict()  # 初始化日志字典

        # 遍历条件嵌入器进行记录
        for embedder in self.conditioner.embedders:
            # 检查是否需要记录该嵌入器的条件
            if ((self.log_keys is None) or (embedder.input_key in self.log_keys)) and not self.no_cond_log:
                # 选取批次中的前 n 个样本
                x = batch[embedder.input_key][:n]
                # 如果 x 是张量
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # 如果是类条件，转换整数为字符串
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        # 将文本转换为图像进行记录
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # 如果是二维张量，处理条件等
                        x = ["x".join([str(xx) for xx in x[i].tolist()]) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        # 如果维度不支持，抛出未实现错误
                        raise NotImplementedError()
                # 如果 x 是列表或列表配置
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # 如果是字符串列表，转换为图像记录
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        # 否则抛出未实现错误
                        raise NotImplementedError()
                else:
                    # 如果类型不支持，抛出未实现错误
                    raise NotImplementedError()
                # 将记录的内容加入日志字典
                log[embedder.input_key] = xc
        # 返回记录的日志字典
        return log

    # 使用 torch.no_grad 装饰器，禁止梯度计算以节省内存
    @torch.no_grad()
    # 定义一个日志记录视频的函数
        def log_video(
            self,  # 该方法的调用对象
            batch: Dict,  # 输入参数，表示一批数据，类型为字典
            N: int = 8,  # 可选参数，表示要处理的视频数量，默认为 8
            ucg_keys: List[str] = None,  # 可选参数，表示用户生成内容的关键字列表，默认为 None
            only_log_video_latents=False,  # 可选参数，布尔值，表示是否仅记录视频潜在变量，默认为 False
            **kwargs,  # 可变参数，允许传入额外的关键字参数
    ) -> Dict:
        # 从 conditioner 的 embedders 中提取输入键
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        # 如果定义了 ucg_keys
        if ucg_keys:
            # 断言所有 ucg_keys 都在 conditioner_input_keys 中
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        # 如果没有定义 ucg_keys，则使用 conditioner_input_keys
        else:
            ucg_keys = conditioner_input_keys
        # 初始化日志字典
        log = dict()

        # 获取输入数据
        x = self.get_input(batch)

        # 获取无条件的条件
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            # 如果有 embedders，则传入 ucg_keys，否则为空列表
            force_uc_zero_embeddings=ucg_keys if len(self.conditioner.embedders) > 0 else [],
        )

        # 初始化采样参数字典
        sampling_kwargs = {}

        # 获取输入数据的最小批大小
        N = min(x.shape[0], N)
        # 将输入数据转移到指定设备，并限制为前 N 个
        x = x.to(self.device)[:N]
        # 如果不是潜在输入，则将输入转为浮点32
        if not self.latent_input:
            log["inputs"] = x.to(torch.float32)
        # 调整输入数据的维度
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # 编码第一阶段的输入数据
        z = self.encode_first_stage(x, batch)
        # 如果不是只记录视频潜在
        if not only_log_video_latents:
            # 解码潜在 z 并转为浮点32
            log["reconstructions"] = self.decode_first_stage(z).to(torch.float32)
            # 调整重构数据的维度
            log["reconstructions"] = log["reconstructions"].permute(0, 2, 1, 3, 4).contiguous()
        # 调整潜在 z 的维度
        z = z.permute(0, 2, 1, 3, 4).contiguous()

        # 更新日志字典，记录条件
        log.update(self.log_conditionings(batch, N))

        # 遍历条件 c 的每个键
        for k in c:
            # 如果条件是张量类型
            if isinstance(c[k], torch.Tensor):
                # 从条件中获取前 N 个数据并转移到设备
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        # 如果有噪声图像输入
        if self.noised_image_input:
            # 选择输入数据的第一帧
            image = x[:, :, 0:1]
            # 向第一帧添加噪声
            image = self.add_noise_to_first_frame(image)
            # 编码第一帧
            image = self.encode_first_stage(image, batch)
            # 调整图像的维度
            image = image.permute(0, 2, 1, 3, 4).contiguous()
            # 将图像与潜在 z 的后续帧拼接
            image = torch.concat([image, torch.zeros_like(z[:, 1:])], dim=1)
            # 将拼接图像添加到条件字典
            c["concat"] = image
            uc["concat"] = image
            # 进行采样，生成样本
            samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)  # b t c h w
            # 调整样本的维度
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            # 如果只记录视频潜在
            if only_log_video_latents:
                # 计算潜在变量并记录
                latents = 1.0 / self.scale_factor * samples
                log["latents"] = latents
            else:
                # 解码样本并转为浮点32
                samples = self.decode_first_stage(samples).to(torch.float32)
                # 调整样本的维度
                samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                # 将样本添加到日志中
                log["samples"] = samples
        else:
            # 进行采样，生成样本
            samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)  # b t c h w
            # 调整样本的维度
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            # 如果只记录视频潜在
            if only_log_video_latents:
                # 计算潜在变量并记录
                latents = 1.0 / self.scale_factor * samples
                log["latents"] = latents
            else:
                # 解码样本并转为浮点32
                samples = self.decode_first_stage(samples).to(torch.float32)
                # 调整样本的维度
                samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                # 将样本添加到日志中
                log["samples"] = samples
        # 返回日志字典
        return log
```

# `.\cogvideo-finetune\sat\dit_video_concat.py`

```py
# 导入 functools 模块中的 partial 函数，用于部分应用函数
from functools import partial
# 从 einops 模块导入 rearrange 和 repeat 函数，用于张量重排和重复
from einops import rearrange, repeat
# 导入 numpy 库
import numpy as np

# 导入 PyTorch 库
import torch
# 从 torch 库导入神经网络模块
from torch import nn
# 导入 PyTorch 的功能性模块
import torch.nn.functional as F

# 从自定义模型库中导入基础模型和非冲突函数
from sat.model.base_model import BaseModel, non_conflict
# 从自定义模型库中导入基础混入类
from sat.model.mixins import BaseMixin
# 导入 transformer 的默认钩子和注意力函数
from sat.transformer_defaults import HOOKS_DEFAULT, attention_fn_default
# 从自定义模块中导入列并行线性层
from sat.mpu.layers import ColumnParallelLinear
# 从配置中实例化对象的工具函数
from sgm.util import instantiate_from_config

# 从扩散模块中导入时间步类
from sgm.modules.diffusionmodules.openaimodel import Timestep
# 从扩散模块中导入线性和时间嵌入的实用函数
from sgm.modules.diffusionmodules.util import (
    linear,
    timestep_embedding,
)
# 从自定义层归一化模块中导入层归一化和 RMS 归一化
from sat.ops.layernorm import LayerNorm, RMSNorm


# 定义图像补丁嵌入混入类，继承自基础混入类
class ImagePatchEmbeddingMixin(BaseMixin):
    # 初始化函数，接收输入通道数、隐藏层大小、补丁大小和其他可选参数
    def __init__(
        self,
        in_channels,
        hidden_size,
        patch_size,
        bias=True,
        text_hidden_size=None,
    ):
        # 调用父类构造函数
        super().__init__()
        # 创建卷积层以实现补丁嵌入
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        # 如果提供文本隐藏层大小，则创建线性层以处理文本嵌入
        if text_hidden_size is not None:
            self.text_proj = nn.Linear(text_hidden_size, hidden_size)
        # 否则将文本投影设置为 None
        else:
            self.text_proj = None

    # 定义词嵌入前向传播方法
    def word_embedding_forward(self, input_ids, **kwargs):
        # 获取 3D 图像补丁
        images = kwargs["images"]  # (b,t,c,h,w)
        # 获取批大小 B 和时间步 T
        B, T = images.shape[:2]
        # 将图像展平为 2D 形式以进行卷积操作
        emb = images.view(-1, *images.shape[2:])
        # 使用卷积层进行嵌入转换
        emb = self.proj(emb)  # ((b t),d,h/2,w/2)
        # 将嵌入重塑为三维形状
        emb = emb.view(B, T, *emb.shape[1:])
        # 扁平化嵌入并转置维度
        emb = emb.flatten(3).transpose(2, 3)  # (b,t,n,d)
        # 使用 rearrange 函数重排嵌入
        emb = rearrange(emb, "b t n d -> b (t n) d")

        # 如果存在文本投影，则计算文本嵌入
        if self.text_proj is not None:
            text_emb = self.text_proj(kwargs["encoder_outputs"])
            # 将文本嵌入与图像嵌入连接
            emb = torch.cat((text_emb, emb), dim=1)  # (b,n_t+t*n_i,d)

        # 确保嵌入在内存中是连续的
        emb = emb.contiguous()
        # 返回最终嵌入
        return emb  # (b,n_t+t*n_i,d)

    # 定义重新初始化函数
    def reinit(self, parent_model=None):
        # 获取卷积层的权重数据
        w = self.proj.weight.data
        # 使用 Xavier 均匀分布初始化权重
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # 将卷积层的偏置初始化为 0
        nn.init.constant_(self.proj.bias, 0)
        # 删除变压器的词嵌入
        del self.transformer.word_embeddings


# 定义获取 3D 正弦余弦位置嵌入的函数
def get_3d_sincos_pos_embed(
    embed_dim,
    grid_height,
    grid_width,
    t_size,
    cls_token=False,
    height_interpolation=1.0,
    width_interpolation=1.0,
    time_interpolation=1.0,
):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # 断言嵌入维度能够被 4 整除
    assert embed_dim % 4 == 0
    # 计算空间嵌入维度
    embed_dim_spatial = embed_dim // 4 * 3
    # 计算时间嵌入维度
    embed_dim_temporal = embed_dim // 4

    # 计算空间位置嵌入
    grid_h = np.arange(grid_height, dtype=np.float32) / height_interpolation
    grid_w = np.arange(grid_width, dtype=np.float32) / width_interpolation
    # 创建网格坐标，宽度优先
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # 将网格堆叠成一个数组
    grid = np.stack(grid, axis=0)

    # 重塑网格以适应后续计算
    grid = grid.reshape([2, 1, grid_height, grid_width])
    # 从网格获取 2D 正弦余弦位置嵌入
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # 处理时间位置嵌入
    # 创建一个一维数组 grid_t，值范围从 0 到 t_size-1，类型为 float32，并进行时间插值
    grid_t = np.arange(t_size, dtype=np.float32) / time_interpolation
    # 从 grid_t 生成一维的正弦余弦位置嵌入
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # 将位置嵌入的维度调整为 [T, 1, D] 形式，以便后续拼接
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    # 重复位置嵌入，以匹配网格高度和宽度，形成 [T, H*W, D // 4]
    pos_embed_temporal = np.repeat(pos_embed_temporal, grid_height * grid_width, axis=1)  # [T, H*W, D // 4]
    # 将空间位置嵌入调整为 [1, H*W, D // 4 * 3] 形式
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    # 重复空间位置嵌入，以匹配时间维度，形成 [T, H*W, D // 4 * 3]
    pos_embed_spatial = np.repeat(pos_embed_spatial, t_size, axis=0)  # [T, H*W, D // 4 * 3]

    # 将时间和空间位置嵌入在最后一个维度上进行拼接，形成最终的位置嵌入
    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    # pos_embed = pos_embed.reshape([-1, embed_dim])  # 将位置嵌入重塑为 [T*H*W, D]

    # 返回最终位置嵌入，维度为 [T, H*W, D]
    return pos_embed  # [T, H*W, D]
# 获取二维正弦余弦位置嵌入
def get_2d_sincos_pos_embed(embed_dim, grid_height, grid_width, cls_token=False, extra_tokens=0):
    # 定义网格高度和宽度的正弦余弦嵌入
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # 创建网格高度的数组
    grid_h = np.arange(grid_height, dtype=np.float32)
    # 创建网格宽度的数组
    grid_w = np.arange(grid_width, dtype=np.float32)
    # 生成网格的网格坐标，宽度在前
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # 将网格坐标堆叠成一个数组
    grid = np.stack(grid, axis=0)

    # 调整网格数组形状以便后续处理
    grid = grid.reshape([2, 1, grid_height, grid_width])
    # 从网格中获取二维正弦余弦位置嵌入
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    # 如果需要类别标记且有额外的标记，进行拼接
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    # 返回位置嵌入
    return pos_embed


# 从网格中获取二维正弦余弦位置嵌入
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # 确保嵌入维度是偶数
    assert embed_dim % 2 == 0

    # 使用一半的维度来编码网格高度
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    # 使用一半的维度来编码网格宽度
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    # 将高度和宽度的嵌入拼接在一起
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    # 返回嵌入
    return emb


# 从给定位置获取一维正弦余弦位置嵌入
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    # 定义每个位置的输出维度
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    # 确保嵌入维度是偶数
    assert embed_dim % 2 == 0
    # 生成频率因子数组
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    # 计算频率的倒数
    omega = 1.0 / 10000**omega  # (D/2,)

    # 将位置数组重塑为一维
    pos = pos.reshape(-1)  # (M,)
    # 计算外积以获取正弦和余弦的输入
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    # 计算正弦值
    emb_sin = np.sin(out)  # (M, D/2)
    # 计算余弦值
    emb_cos = np.cos(out)  # (M, D/2)

    # 将正弦和余弦值拼接在一起
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    # 返回嵌入
    return emb


# 基础二维位置嵌入混合类
class Basic2DPositionEmbeddingMixin(BaseMixin):
    def __init__(self, height, width, compressed_num_frames, hidden_size, text_length=0):
        # 初始化基础混合类
        super().__init__()
        # 保存高度
        self.height = height
        # 保存宽度
        self.width = width
        # 计算空间长度
        self.spatial_length = height * width
        # 初始化位置嵌入参数
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, int(text_length + self.spatial_length), int(hidden_size)), requires_grad=False
        )

    # 前向传播位置嵌入
    def position_embedding_forward(self, position_ids, **kwargs):
        return self.pos_embedding

    # 重新初始化位置嵌入
    def reinit(self, parent_model=None):
        # 删除原位置嵌入
        del self.transformer.position_embeddings
        # 生成新的二维正弦余弦位置嵌入
        pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1], self.height, self.width)
        # 将新嵌入拷贝到参数中
        self.pos_embedding.data[:, -self.spatial_length :].copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


# 基础三维位置嵌入混合类
class Basic3DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        text_length=0,
        height_interpolation=1.0,
        width_interpolation=1.0,
        time_interpolation=1.0,
    ):
        # 调用父类构造函数
        super().__init__()
        # 设置实例的高度属性
        self.height = height
        # 设置实例的宽度属性
        self.width = width
        # 设置文本长度属性
        self.text_length = text_length
        # 设置压缩帧数属性
        self.compressed_num_frames = compressed_num_frames
        # 计算空间长度（高度乘以宽度）
        self.spatial_length = height * width
        # 计算补丁数量（高度乘以宽度乘以压缩帧数）
        self.num_patches = height * width * compressed_num_frames
        # 创建位置嵌入参数，初始化为零，形状为 (1, 文本长度 + 补丁数量, 隐藏层大小)，不需要梯度
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, int(text_length + self.num_patches), int(hidden_size)), requires_grad=False
        )
        # 设置高度插值属性
        self.height_interpolation = height_interpolation
        # 设置宽度插值属性
        self.width_interpolation = width_interpolation
        # 设置时间插值属性
        self.time_interpolation = time_interpolation

    def position_embedding_forward(self, position_ids, **kwargs):
        # 检查输入图像的通道数是否为 1
        if kwargs["images"].shape[1] == 1:
            # 返回位置嵌入，包含文本长度和空间长度的部分
            return self.pos_embedding[:, : self.text_length + self.spatial_length]

        # 返回位置嵌入，包含文本长度和序列长度的部分
        return self.pos_embedding[:, : self.text_length + kwargs["seq_length"]]

    def reinit(self, parent_model=None):
        # 删除当前模型的变换器位置嵌入
        del self.transformer.position_embeddings
        # 获取新的三维正弦余弦位置嵌入
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embedding.shape[-1],
            self.height,
            self.width,
            self.compressed_num_frames,
            height_interpolation=self.height_interpolation,
            width_interpolation=self.width_interpolation,
            time_interpolation=self.time_interpolation,
        )
        # 将位置嵌入转换为张量并转为浮点型
        pos_embed = torch.from_numpy(pos_embed).float()
        # 重新排列位置嵌入的形状
        pos_embed = rearrange(pos_embed, "t n d -> (t n) d")
        # 更新位置嵌入的最后一部分为新的位置嵌入
        self.pos_embedding.data[:, -self.num_patches :].copy_(pos_embed)
# 定义一个用于广播连接张量的函数
def broadcat(tensors, dim=-1):
    # 获取输入张量的数量
    num_tensors = len(tensors)
    # 创建一个集合，存储每个张量的维度长度
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    # 确保所有张量都有相同数量的维度
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    # 获取张量的维度长度
    shape_len = list(shape_lens)[0]
    # 如果 dim 是负数，则将其转换为正索引
    dim = (dim + shape_len) if dim < 0 else dim
    # 获取每个张量的形状，按维度进行打包
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    # 创建一个可扩展的维度列表，排除目标维度
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    # 确保可扩展的维度中每个维度的值不超过2种
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    # 获取每个可扩展维度的最大值
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    # 扩展维度，将最大值对应的维度扩展到张量数量
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    # 将目标维度插入到扩展维度中
    expanded_dims.insert(dim, (dim, dims[dim]))
    # 将扩展维度的形状打包
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    # 扩展每个张量到对应的形状
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    # 沿指定维度连接所有张量
    return torch.cat(tensors, dim=dim)


# 定义一个旋转半分的函数
def rotate_half(x):
    # 重新排列输入张量，将最后一维拆分成两个维度
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    # 按最后一维拆分为两个张量
    x1, x2 = x.unbind(dim=-1)
    # 将第二个张量取负并与第一个张量堆叠
    x = torch.stack((-x2, x1), dim=-1)
    # 重新排列堆叠后的张量，将其合并回一维
    return rearrange(x, "... d r -> ... (d r)")


# 定义一个类，用于混合旋转三维位置嵌入
class Rotary3DPositionEmbeddingMixin(BaseMixin):
    # 初始化类的构造函数
    def __init__(
        # 高度
        height,
        # 宽度
        width,
        # 压缩帧数
        compressed_num_frames,
        # 隐藏层大小
        hidden_size,
        # 每个头的隐藏层大小
        hidden_size_head,
        # 文本长度
        text_length,
        # theta参数，默认为10000
        theta=10000,
        # 是否使用旋转向量，默认为False
        rot_v=False,
        # 是否使用可学习的位置嵌入，默认为False
        learnable_pos_embed=False,
    # 定义构造函数的参数
        ):
            # 调用父类的构造函数
            super().__init__()
            # 初始化旋转向量
            self.rot_v = rot_v
    
            # 计算时间维度的尺寸
            dim_t = hidden_size_head // 4
            # 计算高度维度的尺寸
            dim_h = hidden_size_head // 8 * 3
            # 计算宽度维度的尺寸
            dim_w = hidden_size_head // 8 * 3
    
            # 计算时间频率
            freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
            # 计算高度频率
            freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
            # 计算宽度频率
            freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
    
            # 创建时间维度的网格
            grid_t = torch.arange(compressed_num_frames, dtype=torch.float32)
            # 创建高度维度的网格
            grid_h = torch.arange(height, dtype=torch.float32)
            # 创建宽度维度的网格
            grid_w = torch.arange(width, dtype=torch.float32)
    
            # 计算时间频率与网格的外积
            freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
            # 计算高度频率与网格的外积
            freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
            # 计算宽度频率与网格的外积
            freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
    
            # 扩展时间频率的维度
            freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
            # 扩展高度频率的维度
            freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
            # 扩展宽度频率的维度
            freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)
    
            # 将时间、空间频率合并到一起
            freqs = broadcat((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
            # 重新排列频率的维度
            freqs = rearrange(freqs, "t h w d -> (t h w) d")
    
            # 确保频率数据在内存中的连续性
            freqs = freqs.contiguous()
            # 计算频率的正弦值
            freqs_sin = freqs.sin()
            # 计算频率的余弦值
            freqs_cos = freqs.cos()
            # 注册频率的正弦值为缓冲区
            self.register_buffer("freqs_sin", freqs_sin)
            # 注册频率的余弦值为缓冲区
            self.register_buffer("freqs_cos", freqs_cos)
    
            # 保存文本长度
            self.text_length = text_length
            # 如果学习位置嵌入
            if learnable_pos_embed:
                # 计算补丁数量
                num_patches = height * width * compressed_num_frames + text_length
                # 创建可学习的位置信息嵌入参数
                self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, int(hidden_size)), requires_grad=True)
            else:
                # 如果不学习位置嵌入，设置为 None
                self.pos_embedding = None
    
        # 定义旋转函数
        def rotary(self, t, **kwargs):
            # 获取序列长度
            seq_len = t.shape[2]
            # 提取对应序列长度的余弦频率
            freqs_cos = self.freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)
            # 提取对应序列长度的正弦频率
            freqs_sin = self.freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)
    
            # 返回旋转后的结果
            return t * freqs_cos + rotate_half(t) * freqs_sin
    
        # 定义位置嵌入前向传播函数
        def position_embedding_forward(self, position_ids, **kwargs):
            # 如果存在位置嵌入
            if self.pos_embedding is not None:
                # 返回对应位置的嵌入
                return self.pos_embedding[:, :self.text_length + kwargs["seq_length"]]
            else:
                # 如果没有位置嵌入，返回 None
                return None
    
        # 定义注意力函数
        def attention_fn(
            self,
            # 查询层
            query_layer,
            # 键层
            key_layer,
            # 值层
            value_layer,
            # 注意力掩码
            attention_mask,
            # 可选的注意力丢弃
            attention_dropout=None,
            # 可选的记录注意力权重
            log_attention_weights=None,
            # 是否缩放注意力得分
            scaling_attention_score=True,
            **kwargs,
    # 结束函数的定义部分
        ):
            # 从默认钩子中获取注意力函数
            attention_fn_default = HOOKS_DEFAULT["attention_fn"]
    
            # 对查询层的特定部分应用旋转操作
            query_layer[:, :, self.text_length :] = self.rotary(query_layer[:, :, self.text_length :])
            # 对键层的特定部分应用旋转操作
            key_layer[:, :, self.text_length :] = self.rotary(key_layer[:, :, self.text_length :])
            # 如果启用了旋转值，则对值层的特定部分应用旋转操作
            if self.rot_v:
                value_layer[:, :, self.text_length :] = self.rotary(value_layer[:, :, self.text_length :])
    
            # 返回默认的注意力函数，传入相关参数
            return attention_fn_default(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                attention_dropout=attention_dropout,
                log_attention_weights=log_attention_weights,
                scaling_attention_score=scaling_attention_score,
                **kwargs,
            )
# 定义调制函数，接受输入 x、偏移量 shift 和缩放因子 scale
def modulate(x, shift, scale):
    # 返回调制后的结果，通过 x、scale 和 shift 计算
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# 定义 unpatchify 函数，用于将输入 x 转换为图像格式
def unpatchify(x, c, p, w, h, rope_position_ids=None, **kwargs):
    """
    x: 输入形状为 (N, T/2 * S, patch_size**3 * C)
    imgs: 输出形状为 (N, T, H, W, C)
    """
    # 如果存在 rope_position_ids，执行未实现的检查
    if rope_position_ids is not None:
        assert NotImplementedError
        # 处理 pix2struct unpatchify
        L = x.shape[1]  # 获取 x 的第二维度大小
        x = x.reshape(shape=(x.shape[0], L, p, p, c))  # 重塑 x 以符合新的维度
        x = torch.einsum("nlpqc->ncplq", x)  # 使用爱因斯坦求和约定重新排列维度
        imgs = x.reshape(shape=(x.shape[0], c, p, L * p))  # 重塑为图像形状
    else:
        b = x.shape[0]  # 获取批次大小
        # 使用 rearrange 函数将 x 重新排列为图像格式
        imgs = rearrange(x, "b (t h w) (c p q) -> b t c (h p) (w q)", b=b, h=h, w=w, c=c, p=p, q=p)

    # 返回生成的图像
    return imgs


# 定义 FinalLayerMixin 类，继承自 BaseMixin
class FinalLayerMixin(BaseMixin):
    # 初始化类，设置各种参数和层
    def __init__(
        self,
        hidden_size,
        time_embed_dim,
        patch_size,
        out_channels,
        latent_width,
        latent_height,
        elementwise_affine,
    ):
        super().__init__()  # 调用父类初始化
        self.hidden_size = hidden_size  # 存储隐藏层大小
        self.patch_size = patch_size  # 存储补丁大小
        self.out_channels = out_channels  # 存储输出通道数
        # 初始化 LayerNorm 层
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=1e-6)
        # 初始化全连接层
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # 初始化调制层
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 2 * hidden_size, bias=True))

        # 计算空间长度
        self.spatial_length = latent_width * latent_height // patch_size**2
        self.latent_width = latent_width  # 存储潜在宽度
        self.latent_height = latent_height  # 存储潜在高度

    # 定义 final_forward 方法
    def final_forward(self, logits, **kwargs):
        # 从 logits 中提取 x 和 emb
        x, emb = logits[:, kwargs["text_length"] :, :], kwargs["emb"]  # x:(b,(t n),d)
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)  # 调制得到 shift 和 scale
        x = modulate(self.norm_final(x), shift, scale)  # 对 x 进行调制
        x = self.linear(x)  # 通过线性层转换 x

        # 调用 unpatchify 生成最终图像
        return unpatchify(
            x,
            c=self.out_channels,
            p=self.patch_size,
            w=self.latent_width // self.patch_size,
            h=self.latent_height // self.patch_size,
            rope_position_ids=kwargs.get("rope_position_ids", None),
            **kwargs,
        )

    # 定义 reinit 方法
    def reinit(self, parent_model=None):
        # 初始化线性层权重
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)  # 将偏置初始化为 0


# 定义 SwiGLUMixin 类，继承自 BaseMixin
class SwiGLUMixin(BaseMixin):
    # 初始化类，设置层的数量和特征
    def __init__(self, num_layers, in_features, hidden_features, bias=False):
        super().__init__()  # 调用父类初始化
        # 创建一个模块列表，包含多个 ColumnParallelLinear 层
        self.w2 = nn.ModuleList(
            [
                ColumnParallelLinear(
                    in_features,
                    hidden_features,
                    gather_output=False,
                    bias=bias,
                    module=self,
                    name="dense_h_to_4h_gate",
                )
                for i in range(num_layers)  # 根据层数生成相应数量的层
            ]
        )
    # 前向传播函数，接受隐藏状态和其他参数
        def mlp_forward(self, hidden_states, **kw_args):
            # 将输入的隐藏状态赋值给 x
            x = hidden_states
            # 获取指定层的 MLP（多层感知机）模块
            origin = self.transformer.layers[kw_args["layer_id"]].mlp
            # 通过第一层全连接将 x 映射到 4h 维度
            x1 = origin.dense_h_to_4h(x)
            # 使用权重矩阵将 x 映射到另一个维度
            x2 = self.w2[kw_args["layer_id"]](x)
            # 应用激活函数并乘以第一层输出，生成隐藏状态
            hidden = origin.activation_func(x2) * x1
            # 将隐藏状态通过最后一层全连接映射回原始维度
            x = origin.dense_4h_to_h(hidden)
            # 返回最终输出
            return x
# 定义一个混合类 AdaLNMixin，继承自 BaseMixin
class AdaLNMixin(BaseMixin):
    # 初始化方法，定义类的构造参数
    def __init__(
        self,
        width,  # 宽度参数
        height,  # 高度参数
        hidden_size,  # 隐藏层大小
        num_layers,  # 层数
        time_embed_dim,  # 时间嵌入维度
        compressed_num_frames,  # 压缩帧数
        qk_ln=True,  # 是否使用查询和键的层归一化
        hidden_size_head=None,  # 每个头的隐藏层大小
        elementwise_affine=True,  # 是否使用逐元素仿射变换
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存层数到实例变量
        self.num_layers = num_layers
        # 保存宽度到实例变量
        self.width = width
        # 保存高度到实例变量
        self.height = height
        # 保存压缩帧数到实例变量
        self.compressed_num_frames = compressed_num_frames

        # 创建一个包含多个线性层和激活函数的模块列表
        self.adaLN_modulations = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 12 * hidden_size)) for _ in range(num_layers)]
        )

        # 保存是否使用层归一化的标志
        self.qk_ln = qk_ln
        # 如果使用层归一化，则初始化查询和键的层归一化列表
        if qk_ln:
            # 创建查询层归一化的模块列表
            self.query_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )
            # 创建键层归一化的模块列表
            self.key_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )

    # 定义前向传播方法
    def layer_forward(
        self,
        hidden_states,  # 输入的隐藏状态
        mask,  # 输入的掩码
        *args,  # 其他可变参数
        **kwargs,  # 其他关键字参数
    ):
        pass  # 该方法未实现，留作将来的扩展

    # 定义重初始化方法
    def reinit(self, parent_model=None):
        # 对每个 adaLN 调制层进行初始化
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)  # 将最后一层的权重初始化为0
            nn.init.constant_(layer[-1].bias, 0)  # 将最后一层的偏置初始化为0

    # 定义注意力函数，带有非冲突装饰器
    @non_conflict
    def attention_fn(
        self,
        query_layer,  # 查询层
        key_layer,  # 键层
        value_layer,  # 值层
        attention_mask,  # 注意力掩码
        attention_dropout=None,  # 注意力的丢弃率
        log_attention_weights=None,  # 日志注意力权重
        scaling_attention_score=True,  # 是否缩放注意力得分
        old_impl=attention_fn_default,  # 默认的注意力实现
        **kwargs,  # 其他关键字参数
    ):
        # 如果使用查询和键的层归一化
        if self.qk_ln:
            # 获取当前层的查询层归一化模块
            query_layernorm = self.query_layernorm_list[kwargs["layer_id"]]
            # 获取当前层的键层归一化模块
            key_layernorm = self.key_layernorm_list[kwargs["layer_id"]]
            # 对查询层进行层归一化
            query_layer = query_layernorm(query_layer)
            # 对键层进行层归一化
            key_layer = key_layernorm(key_layer)

        # 返回注意力函数的结果
        return old_impl(
            query_layer,  # 归一化后的查询层
            key_layer,  # 归一化后的键层
            value_layer,  # 值层
            attention_mask,  # 注意力掩码
            attention_dropout=attention_dropout,  # 注意力的丢弃率
            log_attention_weights=log_attention_weights,  # 日志注意力权重
            scaling_attention_score=scaling_attention_score,  # 缩放注意力得分
            **kwargs,  # 其他关键字参数
        )

# 定义数据类型到 PyTorch 数据类型的映射
str_to_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# 定义扩散变换器类，继承自 BaseModel
class DiffusionTransformer(BaseModel):
    # 初始化方法，用于创建类的实例
    def __init__(
        # 变换器参数
        self,
        transformer_args,
        # 帧数
        num_frames,
        # 时间压缩率
        time_compressed_rate,
        # 潜在宽度
        latent_width,
        # 潜在高度
        latent_height,
        # 补丁大小
        patch_size,
        # 输入通道数
        in_channels,
        # 输出通道数
        out_channels,
        # 隐藏层大小
        hidden_size,
        # 层数
        num_layers,
        # 注意力头数
        num_attention_heads,
        # 是否进行逐元素仿射变换
        elementwise_affine,
        # 时间嵌入维度，默认为 None
        time_embed_dim=None,
        # 类别数量，默认为 None
        num_classes=None,
        # 模块配置，默认为空字典
        modules={},
        # 输入时间格式，默认为 "adaln"
        input_time="adaln",
        # 自适应输入通道数，默认为 None
        adm_in_channels=None,
        # 是否并行输出，默认为 True
        parallel_output=True,
        # 高度插值因子，默认为 1.0
        height_interpolation=1.0,
        # 宽度插值因子，默认为 1.0
        width_interpolation=1.0,
        # 时间插值因子，默认为 1.0
        time_interpolation=1.0,
        # 是否使用 SwiGLU 激活函数，默认为 False
        use_SwiGLU=False,
        # 是否使用 RMSNorm 归一化，默认为 False
        use_RMSNorm=False,
        # 是否将 y 嵌入初始化为零，默认为 False
        zero_init_y_embed=False,
        # 其他参数，使用关键字参数收集
        **kwargs,
    ):
        # 设置潜在宽度
        self.latent_width = latent_width
        # 设置潜在高度
        self.latent_height = latent_height
        # 设置补丁大小
        self.patch_size = patch_size
        # 设置帧数
        self.num_frames = num_frames
        # 设置时间压缩率
        self.time_compressed_rate = time_compressed_rate
        # 计算空间长度
        self.spatial_length = latent_width * latent_height // patch_size**2
        # 设置输入通道数
        self.in_channels = in_channels
        # 设置输出通道数
        self.out_channels = out_channels
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置模型通道数，等于隐藏层大小
        self.model_channels = hidden_size
        # 设置时间嵌入维度，如果未提供则使用隐藏层大小
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else hidden_size
        # 设置类别数量
        self.num_classes = num_classes
        # 设置自适应输入通道数
        self.adm_in_channels = adm_in_channels
        # 设置输入时间格式
        self.input_time = input_time
        # 设置层数
        self.num_layers = num_layers
        # 设置注意力头数
        self.num_attention_heads = num_attention_heads
        # 设置是否为解码器
        self.is_decoder = transformer_args.is_decoder
        # 设置是否进行逐元素仿射变换
        self.elementwise_affine = elementwise_affine
        # 设置高度插值因子
        self.height_interpolation = height_interpolation
        # 设置宽度插值因子
        self.width_interpolation = width_interpolation
        # 设置时间插值因子
        self.time_interpolation = time_interpolation
        # 计算内部隐藏层大小，等于隐藏层大小的四倍
        self.inner_hidden_size = hidden_size * 4
        # 设置是否将 y 嵌入初始化为零
        self.zero_init_y_embed = zero_init_y_embed
        # 尝试从关键字参数中提取数据类型
        try:
            self.dtype = str_to_dtype[kwargs.pop("dtype")]
        except:
            # 默认数据类型为 float32
            self.dtype = torch.float32

        # 如果使用 SwiGLU 激活函数，将其添加到关键字参数中
        if use_SwiGLU:
            kwargs["activation_func"] = F.silu
        # 如果没有指定激活函数，使用近似 GELU 激活函数
        elif "activation_func" not in kwargs:
            approx_gelu = nn.GELU(approximate="tanh")
            kwargs["activation_func"] = approx_gelu

        # 如果使用 RMSNorm 归一化，添加到关键字参数中
        if use_RMSNorm:
            kwargs["layernorm"] = RMSNorm
        else:
            # 否则使用带有逐元素仿射变换的 LayerNorm
            kwargs["layernorm"] = partial(LayerNorm, elementwise_affine=elementwise_affine, eps=1e-6)

        # 更新变换器参数中的层数、隐藏层大小和注意力头数
        transformer_args.num_layers = num_layers
        transformer_args.hidden_size = hidden_size
        transformer_args.num_attention_heads = num_attention_heads
        transformer_args.parallel_output = parallel_output
        # 调用父类的初始化方法
        super().__init__(args=transformer_args, transformer=None, **kwargs)

        # 模块配置
        module_configs = modules
        # 构建模块
        self._build_modules(module_configs)

        # 如果使用 SwiGLU 激活函数，添加混合层
        if use_SwiGLU:
            self.add_mixin(
                "swiglu", SwiGLUMixin(num_layers, hidden_size, self.inner_hidden_size, bias=False), reinit=True
            )
    # 前向传播函数，接收输入 x 及其他可选参数
        def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
            # 获取输入 x 的形状，分别为批量大小 b、时间步 t、通道 d、高 h 和宽 w
            b, t, d, h, w = x.shape
            # 如果输入 x 的数据类型与模型的数据类型不匹配，则转换为模型的数据类型
            if x.dtype != self.dtype:
                x = x.to(self.dtype)
    
            # 此部分在推理时不使用
            if "concat_images" in kwargs and kwargs["concat_images"] is not None:
                # 如果 concat_images 的批量大小与 x 不匹配，则重复 concat_images
                if kwargs["concat_images"].shape[0] != x.shape[0]:
                    concat_images = kwargs["concat_images"].repeat(2, 1, 1, 1, 1)
                else:
                    # 否则直接使用 concat_images
                    concat_images = kwargs["concat_images"]
                # 将 x 和 concat_images 在时间维度上进行拼接
                x = torch.cat([x, concat_images], dim=2)
    
            # 断言 y 的存在性与 num_classes 的存在性相对应，确保一致性
            assert (y is not None) == (
                self.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"
            # 生成时间步的嵌入向量
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
            # 通过时间嵌入获取最终的嵌入
            emb = self.time_embed(t_emb)
    
            if self.num_classes is not None:
                # 确保 y 的形状与 x 的批量大小相符
                # assert y.shape[0] == x.shape[0]
                # 确保 x 的批量大小能被 y 的批量大小整除
                assert x.shape[0] % y.shape[0] == 0
                # 重复 y 以匹配 x 的批量大小
                y = y.repeat_interleave(x.shape[0] // y.shape[0], dim=0)
                # 将类别嵌入添加到 emb 中
                emb = emb + self.label_emb(y)
    
            # 在 kwargs 中存储序列长度、图像、嵌入、编码器输出及文本长度
            kwargs["seq_length"] = t * h * w // (self.patch_size**2)
            kwargs["images"] = x
            kwargs["emb"] = emb
            kwargs["encoder_outputs"] = context
            kwargs["text_length"] = context.shape[1]
    
            # 初始化输入 ID、位置 ID 和注意力掩码为全 1 的张量
            kwargs["input_ids"] = kwargs["position_ids"] = kwargs["attention_mask"] = torch.ones((1, 1)).to(x.dtype)
            # 调用父类的 forward 方法，并获取第一个输出
            output = super().forward(**kwargs)[0]
            # 返回最终输出
            return output
```