
# `diffusers\examples\research_projects\diffusion_dpo\train_diffusion_dpo_sdxl.py` 详细设计文档

这是一个用于训练 Stable Diffusion XL (SDXL) 模型的 LoRA (Low-Rank Adaptation) 训练脚本，采用 Direct Preference Optimization (DPO) 方法进行微调。脚本支持多 GPU 分布式训练、混合精度训练、梯度累积、模型检查点保存、验证推理以及将训练好的 LoRA 权重推送至 Hugging Face Hub。

## 整体流程

```mermaid
graph TD
    A[开始] --> B[解析命令行参数]
B --> C[初始化Accelerator分布式训练环境]
C --> D[设置随机种子和日志级别]
D --> E[加载预训练模型和分词器]
E --> F[配置LoRA适配器并冻结原模型参数]
F --> G[加载和处理数据集]
G --> H[创建数据加载器DataLoader]
H --> I[初始化优化器和学习率调度器]
I --> J{进入训练循环}
J -->|每个step| K[前向传播:VAE编码图像]
K --> L[添加噪声到latent表示]
K --> M[编码文本提示词]
M --> N[UNet预测噪声残差]
N --> O[计算DPO损失函数]
O --> P[反向传播并更新参数]
P --> Q{检查是否需要保存检查点}
Q -->|是| R[保存模型检查点]
Q -->|否| S{检查是否需要验证}
S -->|是| T[运行验证生成图像]
S -->|否| J
R --> J
T --> J
J -->|训练完成| U[保存LoRA权重到输出目录]
U --> V[可选:推送至HuggingFace Hub]
V --> W[结束]
```

## 类结构

```
模块: 训练脚本 (无类定义)
├── 全局函数
│   ├── import_model_class_from_model_name_or_path
│   ├── log_validation
│   ├── parse_args
│   ├── tokenize_captions
│   ├── encode_prompt
│   └── main
├── 全局变量
│   ├── VALIDATION_PROMPTS
│   └── logger
└── 内部嵌套函数 (在main函数内)
    ├── enforce_zero_terminal_snr
    ├── preprocess_train
    ├── collate_fn
    ├── compute_time_ids
    ├── save_model_hook
    └── load_model_hook
```

## 全局变量及字段


### `logger`
    
Logger for outputting training progress, debugging information, and tracking events throughout the training pipeline

类型：`logging.Logger`
    


### `VALIDATION_PROMPTS`
    
A predefined list of text prompts used for model validation during the training process to generate sample images for quality assessment

类型：`List[str]`
    


    

## 全局函数及方法



### `import_model_class_from_model_name_or_path`

该函数是 Stable Diffusion XLoRA 训练脚本中的模型类导入工具函数，通过读取预训练模型的配置文件，动态获取对应的文本编码器类（CLIPTextModel 或 CLIPTextModelWithProjection），以支持不同版本和架构的 Stable Diffusion 模型。

参数：

- `pretrained_model_name_or_path`：`str`，预训练模型的名称（如 "stabilityai/stable-diffusion-xl-base-1.0"）或本地模型路径，用于指定要加载的预训练模型。
- `revision`：`str`，模型版本标识符，可以是提交哈希、分支名或标签名，用于加载特定版本的模型配置。
- `subfolder`：`str`，模型子文件夹路径，默认为 "text_encoder"，用于指定配置文件中 Text Encoder 所在的子目录（可选值包括 "text_encoder" 或 "text_encoder_2"）。

返回值：`type`，返回对应的文本编码器类对象（`CLIPTextModel` 或 `CLIPTextModelWithProjection`），用于后续通过 `.from_pretrained()` 方法实例化模型。

#### 流程图

```mermaid
graph TD
    A[开始] --> B[调用 PretrainedConfig.from_pretrained<br/>加载模型配置]
    B --> C[获取 text_encoder_config.architectures[0]<br/>即模型架构名称]
    C --> D{判断 model_class}
    D -->|CLIPTextModel| E[从 transformers 导入 CLIPTextModel 类]
    E --> F[返回 CLIPTextModel 类]
    D -->|CLIPTextModelWithProjection| G[从 transformers 导入 CLIPTextModelWithProjection 类]
    G --> H[返回 CLIPTextModelWithProjection 类]
    D -->|其他类型| I[抛出 ValueError 异常<br/>提示模型类型不支持]
    F --> Z[结束]
    H --> Z
    I --> Z
```

#### 带注释源码

```python
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    """
    根据预训练模型配置动态导入对应的 Text Encoder 类。
    
    参数:
        pretrained_model_name_or_path: 预训练模型名称或本地路径
        revision: 模型版本/分支
        subfolder: 模型子文件夹，默认 "text_encoder"
    
    返回:
        对应的 CLIP 文本编码器类
    """
    # 步骤1: 加载预训练模型的配置文件
    # PretrainedConfig 是 HuggingFace Transformers 库的基础配置类
    # 通过 from_pretrained 方法可以加载远程或本地模型配置
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    
    # 步骤2: 从配置中获取模型架构名称
    # architectures 是一个列表，包含模型使用的架构类名
    # 通常第一个元素即为主架构
    model_class = text_encoder_config.architectures[0]
    
    # 步骤3: 根据架构名称返回对应的类对象
    # 这里采用条件判断处理两种常见的 CLIP 文本编码器
    
    if model_class == "CLIPTextModel":
        # 标准 CLIP 文本编码器，用于大多数 SD 模型
        from transformers import CLIPTextModel
        
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        # 带投影层的 CLIP 文本编码器，用于 SDXL 等高级模型
        # 该版本会输出额外的 pooled 嵌入用于条件控制
        from transformers import CLIPTextModelWithProjection
        
        return CLIPTextModelWithProjection
    else:
        # 如果遇到不支持的模型架构，抛出明确异常
        # 便于开发者定位问题
        raise ValueError(f"{model_class} is not supported.")
```



### `log_validation`

该函数用于在训练过程中运行验证推理，根据预定义的验证提示词生成图像，并将生成的图像记录到训练跟踪器（TensorBoard 或 WANDB）中，以便监控模型的生成效果。

参数：

- `args`：命令行参数对象（argparse.Namespace），包含模型路径、输出目录、随机种子等配置信息
- `unet`：UNet2DConditionModel，Diffusion 模型的 UNet 组件，用于去噪过程
- `vae`：AutoencoderKL，变分自编码器，用于将图像编码为潜在空间表示
- `accelerator`：Accelerator，Hugging Face Accelerate 库提供的分布式训练加速器
- `weight_dtype`：torch.dtype，模型权重的精度类型（如 float16、bfloat16）
- `epoch`：int，当前训练的轮次编号
- `is_final_validation`：bool（可选，默认为 False），标识是否为训练结束后的最终验证

返回值：`None`，该函数不返回任何值，仅执行图像生成和日志记录操作

#### 流程图

```mermaid
flowchart TD
    A[开始 log_validation] --> B[记录验证开始日志]
    B --> C{is_final_validation<br/>且 mixed_precision == fp16?}
    C -->|Yes| D[将 VAE 转换为 weight_dtype]
    C -->|No| E[跳过此步骤]
    D --> F[从预训练模型创建 DiffusionPipeline]
    E --> F
    F --> G{is_final_validation?}
    G -->|Yes| H[加载 LoRA 权重到 pipeline]
    G -->|No| I[使用 accelerator.unwrap_model 加载 UNet]
    H --> J[将 pipeline 移到加速设备]
    I --> J
    J --> K[设置生成器种子和推理参数]
    K --> L[遍历 VALIDATION_PROMPTS]
    L --> M[使用 pipeline 生成图像]
    M --> N[收集所有生成的图像]
    N --> O{记录到 tracker?}
    O -->|TensorBoard| P[转换为 numpy 数组并记录图像]
    O -->|WANDB| Q[使用 wandb.Image 记录图像]
    P --> R[是最终验证?]
    Q --> R
    R -->|Yes| S[禁用 LoRA 并生成无 LoRA 图像]
    R -->|No| T[结束]
    S --> U[记录无 LoRA 的图像到 tracker]
    U --> T
```

#### 带注释源码

```python
def log_validation(args, unet, vae, accelerator, weight_dtype, epoch, is_final_validation=False):
    """
    运行验证推理，生成图像并记录到训练跟踪器
    
    参数:
        args: 命令行参数对象，包含模型路径、输出目录等配置
        unet: UNet2DConditionModel 模型实例
        vae: AutoencoderKL 模型实例
        accelerator: Accelerator 分布式训练加速器
        weight_dtype: torch.dtype，权重精度类型
        epoch: int，当前训练轮次
        is_final_validation: bool，是否为最终验证
    """
    # 记录验证开始的日志信息
    logger.info(f"Running validation... \n Generating images with prompts:\n {VALIDATION_PROMPTS}.")

    # 如果是最终验证且使用 fp16 混合精度，将 VAE 转换到对应精度
    if is_final_validation:
        if args.mixed_precision == "fp16":
            vae.to(weight_dtype)

    # 从预训练模型创建 DiffusionPipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    
    # 根据是否为最终验证决定如何加载 UNet
    if not is_final_validation:
        # 训练过程中使用 accelerator.unwrap_model 获取 unwrap 后的模型
        pipeline.unet = accelerator.unwrap_model(unet)
    else:
        # 最终验证时从输出目录加载 LoRA 权重
        pipeline.load_lora_weights(args.output_dir, weight_name="pytorch_lora_weights.safetensors")

    # 将 pipeline 移到加速设备并禁用进度条
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # 创建随机生成器，使用指定种子保证可复现性
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    images = []
    # 根据是否最终验证决定是否使用自动混合精度上下文
    context = contextlib.nullcontext() if is_final_validation else torch.cuda.amp.autocast()

    # 设置推理参数
    guidance_scale = 5.0
    num_inference_steps = 25
    # 如果是 turbo 模式，使用不同的参数
    if args.is_turbo:
        guidance_scale = 0.0
        num_inference_steps = 4
    
    # 遍历所有验证提示词进行推理
    for prompt in VALIDATION_PROMPTS:
        with context:
            image = pipeline(
                prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator
            ).images[0]
            images.append(image)

    # 确定 tracker 的键名
    tracker_key = "test" if is_final_validation else "validation"
    
    # 遍历所有 tracker 记录图像
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            # 将图像堆叠为 numpy 数组并记录
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(tracker_key, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    tracker_key: [
                        wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}") for i, image in enumerate(images)
                    ]
                }
            )

    # 最终验证时额外生成不含 LoRA 的图像用于对比
    if is_final_validation:
        pipeline.disable_lora()
        no_lora_images = [
            pipeline(
                prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator
            ).images[0]
            for prompt in VALIDATION_PROMPTS
        ]

        # 记录无 LoRA 的图像
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in no_lora_images])
                tracker.writer.add_images("test_without_lora", np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "test_without_lora": [
                            wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}")
                            for i, image in enumerate(no_lora_images)
                        ]
                    }
                )
```



### `parse_args`

该函数是训练脚本的命令行参数解析器，通过argparse模块定义并收集所有训练所需的超参数，包括模型路径、数据集配置、训练策略、优化器设置、分布式训练参数等，并进行基本的参数校验后返回解析结果。

参数：

- `input_args`：`Optional[List[str]]`，可选参数，用于指定要解析的参数列表（主要用于测试），默认为None，即从sys.argv解析

返回值：`argparse.Namespace`，包含所有命令行参数解析后的命名空间对象

#### 流程图

```mermaid
flowchart TD
    A[开始 parse_args] --> B[创建 ArgumentParser]
    B --> C[添加所有命令行参数]
    C --> D{input_args 是否为 None?}
    D -->|是| E[parser.parse_args]
    D -->|否| F[parser.parse_args input_args]
    E --> G[验证参数: dataset_name 必须提供]
    F --> G
    G --> H{is_turbo 为真?}
    H -->|是| I[断言 pretrained_model_name_or_path 包含 turbo]
    H -->|否| J[跳过断言]
    I --> K[检查环境变量 LOCAL_RANK]
    J --> K
    K --> L[同步 local_rank]
    L --> M[返回 args]
```

#### 带注释源码

```python
def parse_args(input_args=None):
    """
    解析命令行参数，返回包含所有训练配置的命名空间对象
    
    Args:
        input_args: 可选的参数列表，用于测试场景。如果为None，则从命令行解析。
    
    Returns:
        argparse.Namespace: 解析后的参数对象，包含所有定义的命令行参数
    """
    # 创建ArgumentParser实例，设置描述信息
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    # ========================================
    # 模型相关参数
    # ========================================
    # 预训练模型名称或路径，必填参数
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # 预训练VAE模型路径，用于更好的数值稳定性
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    # 模型版本修订号
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    # 模型文件变体，如fp16
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    
    # ========================================
    # 数据集相关参数
    # ========================================
    # 数据集名称，必填参数
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    # 数据集分割名称
    parser.add_argument(
        "--dataset_split_name",
        type=str,
        default="validation",
        help="Dataset split to be used during training. Helpful to specify for conducting experimental runs.",
    )
    
    # ========================================
    # 验证相关参数
    # ========================================
    # 是否在训练过程中运行验证
    parser.add_argument(
        "--run_validation",
        default=False,
        action="store_true",
        help="Whether to run validation inference in between training and also after training. Helps to track progress.",
    )
    # 验证执行步数间隔
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    # 调试用，限制训练样本数量
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    
    # ========================================
    # 输出与存储相关参数
    # ========================================
    # 输出目录
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diffusion-dpo-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # 缓存目录
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    # 随机种子
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # TensorBoard日志目录
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    
    # ========================================
    # 图像处理参数
    # ========================================
    # 输入图像分辨率
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    # VAE编码批次大小
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=8,
        help="Batch size to use for VAE encoding of the images for efficient processing.",
    )
    # 是否禁用水平翻转
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    # 是否随机裁剪
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to random crop the input images to the resolution. If not set, the images will be center-cropped."
        ),
    )
    
    # ========================================
    # 训练过程参数
    # ========================================
    # 训练批次大小
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    # 训练轮数
    parser.add_argument("--num_train_epochs", type=int, default=1)
    # 最大训练步数
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    # 检查点保存步数间隔
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    # 检查点总数限制
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    # 从检查点恢复训练
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # 梯度累积步数
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # 梯度检查点，用于节省显存
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    
    # ========================================
    # DPO相关参数
    # ========================================
    # DPO KL散度惩罚系数
    parser.add_argument(
        "--beta_dpo",
        type=int,
        default=5000,
        help="DPO KL Divergence penalty.",
    )
    
    # ========================================
    # 优化器与学习率参数
    # ========================================
    # 初始学习率
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    # 是否根据GPU数量、梯度累积和批次大小缩放学习率
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    # 学习率调度器类型
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    # 学习率预热步数
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    # 余弦调度器重启次数
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    # 多项式调度器幂次
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    # DataLoader工作进程数
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # 是否使用8位Adam优化器
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    # Adam优化器参数
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    # 梯度裁剪最大值
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    # ========================================
    # Hub推送相关参数
    # ========================================
    # 是否推送到Hub
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    # Hub访问令牌
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    # Hub模型ID
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    
    # ========================================
    # 精度与加速相关参数
    # ========================================
    # 是否允许TF32计算
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    # 日志报告目标
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # 混合精度训练类型
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    # 先验生成精度
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    # 分布式训练本地排名
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    # ========================================
    # 内存优化与特殊模型参数
    # ========================================
    # 是否启用xFormers高效注意力
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    # 是否使用SDXL Turbo模式
    parser.add_argument(
        "--is_turbo",
        action="store_true",
        help=("Use if tuning SDXL Turbo instead of SDXL"),
    )
    # LoRA秩维度
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    # 追踪器名称
    parser.add_argument(
        "--tracker_name",
        type=str,
        default="diffusion-dpo-lora-sdxl",
        help=("The name of the tracker to report results to."),
    )
    
    # ========================================
    # 参数解析
    # ========================================
    # 根据input_args是否为空决定解析方式
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    # ========================================
    # 参数校验
    # ========================================
    # 校验dataset_name必须提供
    if args.dataset_name is None:
        raise ValueError("Must provide a `dataset_name`.")
    
    # 校验Turbo模式模型路径
    if args.is_turbo:
        assert "turbo" in args.pretrained_model_name_or_path
    
    # ========================================
    # 环境变量处理
    # ========================================
    # 从环境变量获取LOCAL_RANK并同步到args
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    return args
```



### `tokenize_captions`

该函数是用于对文本描述（captions）进行分词处理的核心函数，主要应用于Stable Diffusion XL (SDXL) 模型的训练流程中。SDXL使用两个文本编码器（tokenizer_one和tokenizer_two），因此该函数同时返回两组token IDs，分别对应两个不同的tokenizer，以满足双文本编码器的条件输入需求。

参数：

- `tokenizers`：元组或列表，包含两个`PreTrainedTokenizer`对象，分别对应SDXL的主tokenizer和辅助tokenizer（tokenizer_2）
- `examples`：字典，包含键`"caption"`，其值为待tokenize的文本描述列表

返回值：`Tuple[torch.Tensor, torch.Tensor]`，返回两个tokenizer生成的token IDs张量，形状为`(batch_size, max_length)`

#### 流程图

```mermaid
flowchart TD
    A[开始 tokenize_captions] --> B[从 examples['caption'] 提取 captions 列表]
    B --> C[使用第一个 tokenizer 对 captions 进行分词]
    C --> D[truncation=True, padding='max_length']
    D --> E[返回 token IDs: tokens_one]
    E --> F[使用第二个 tokenizer 对 captions 进行分词]
    F --> G[truncation=True, padding='max_length']
    G --> H[返回 token IDs: tokens_two]
    H --> I[返回元组 tokens_one, tokens_two]
```

#### 带注释源码

```python
def tokenize_captions(tokenizers, examples):
    """
    对文本描述进行tokenize处理，返回两组token IDs。
    
    参数:
        tokenizers: 包含两个PreTrainedTokenizer的列表/元组
        examples: 包含'caption'键的字典，值为文本列表
    
    返回:
        tokens_one: 第一个tokenizer生成的token IDs
        tokens_two: 第二个tokenizer生成的token IDs
    """
    # 1. 从examples字典中提取caption列表
    captions = []
    for caption in examples["caption"]:
        captions.append(caption)

    # 2. 使用第一个tokenizer（主tokenizer）进行分词
    #    - truncation=True: 超过最大长度的序列进行截断
    #    - padding='max_length': 填充到最大长度
    #    - return_tensors='pt': 返回PyTorch张量
    tokens_one = tokenizers[0](
        captions, 
        truncation=True, 
        padding="max_length", 
        max_length=tokenizers[0].model_max_length, 
        return_tensors="pt"
    ).input_ids
    
    # 3. 使用第二个tokenizer（辅助tokenizer，用于SDXL的CLIP Text Encoder 2）进行分词
    #    参数与第一个tokenizer相同
    tokens_two = tokenizers[1](
        captions, 
        truncation=True, 
        padding="max_length", 
        max_length=tokenizers[1].model_max_length, 
        return_tensors="pt"
    ).input_ids

    # 4. 返回两组token IDs，供后续encode_prompt函数使用
    return tokens_one, tokens_two
```



### `encode_prompt`

该函数用于将文本提示（caption）通过文本编码器编码为向量表示，支持双文本编码器（SDXL架构），返回用于UNet条件注入的提示嵌入和池化后的提示嵌入。

参数：

- `text_encoders`：`List[torch.nn.Module]`，文本编码器列表，通常包含主文本编码器和IP-Adapter文本编码器（如CLIPTextModel或CLIPTextModelWithProjection）
- `text_input_ids_list`：`List[torch.Tensor]`，与文本编码器对应的tokenized输入ID列表，每个元素为批量大小的序列

返回值：

- `prompt_embeds`：`torch.Tensor`，形状为`(batch_size, seq_len, hidden_dim)`的提示嵌入向量，用于UNet的conditioning
- `pooled_prompt_embeds`：`torch.Tensor`，形状为`(batch_size, hidden_dim)`的池化提示嵌入，用于额外的文本条件注入

#### 流程图

```mermaid
flowchart TD
    A[开始 encode_prompt] --> B[初始化空列表 prompt_embeds_list]
    B --> C{遍历 text_encoders 和 text_input_ids_list}
    C -->|第i个编码器| D[获取第i个text_input_ids]
    D --> E[调用text_encoder forward]
    E --> F[提取pooled输出 prompt_embeds[0]]
    F --> G[提取倒数第二层隐藏状态 hidden_states[-2]]
    G --> H[reshape为2D: bs_embed x seq_len x -1]
    H --> I[追加到prompt_embeds_list]
    I --> C
    C -->|遍历完成| J[沿最后一维concat所有嵌入]
    J --> K[reshape pooled_prompt_embeds为2D]
    K --> L[返回 prompt_embeds 和 pooled_prompt_embeds]
```

#### 带注释源码

```python
@torch.no_grad()  # 禁用梯度计算，减少显存占用
def encode_prompt(text_encoders, text_input_ids_list):
    """
    将文本提示编码为嵌入向量，支持双文本编码器架构（SDXL/SDXL-Turbo）
    
    Args:
        text_encoders: 文本编码器列表（通常为2个：text_encoder_one, text_encoder_two）
        text_input_ids_list: 对应的tokenized输入ID列表
    
    Returns:
        prompt_embeds: 拼接后的提示嵌入 (batch_size, seq_len, hidden_dim)
        pooled_prompt_embeds: 池化后的提示嵌入 (batch_size, hidden_dim)
    """
    prompt_embeds_list = []  # 存储每个编码器的输出嵌入

    # 遍历所有文本编码器（SDXL有2个，SD 1.x有1个）
    for i, text_encoder in enumerate(text_encoders):
        # 获取当前编码器对应的tokenized输入
        text_input_ids = text_input_ids_list[i]

        # 执行前向传播，获取隐藏状态
        # output_hidden_states=True 确保返回所有层的隐藏状态
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),  # 将输入移到编码器设备上
            output_hidden_states=True,
        )

        # 从输出元组中提取pooled输出（用于text_embeds条件）
        # prompt_embeds[0] 是pooled output（CLIP pooled CLS token）
        pooled_prompt_embeds = prompt_embeds[0]
        
        # 提取倒数第二层的隐藏状态（SDXL惯例：-2层是最终可用层，-1层是额外的）
        # 这是SDXL训练脚本中的标准做法
        prompt_embeds = prompt_embeds.hidden_states[-2]
        
        # 获取批次大小、序列长度、隐藏维度
        bs_embed, seq_len, _ = prompt_embeds.shape
        
        # 重新reshape为3D张量 (bs, seq_len, hidden)
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        
        # 添加到列表，稍后拼接
        prompt_embeds_list.append(prompt_embeds)

    # 沿最后一维（hidden_dim）拼接所有文本编码器的嵌入
    # 将多个编码器的输出合并为一个大的嵌入向量
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    
    # 将pooled嵌入reshape为2D (bs, hidden_dim)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    
    return prompt_embeds, pooled_prompt_embeds
```



### `main`

该函数是SDXL LoRA训练脚本的核心入口，负责整个DPO（Direct Preference Optimization）训练流程，包括参数验证、环境配置、模型加载与初始化、数据集处理、训练循环执行、模型保存以及可选的验证与推送。

参数：

- `args`：`argparse.Namespace`，通过`parse_args()`解析的命令行参数对象，包含所有训练配置（如模型路径、学习率、批量大小等）

返回值：`None`，该函数无返回值，执行完成后直接结束

#### 流程图

```mermaid
flowchart TD
    A[开始 main 函数] --> B{检查 report_to 与 hub_token 组合}
    B -->|有效| C[创建日志目录]
    B -->|无效| Z[抛出 ValueError]
    C --> D[创建 Accelerator 实例]
    D --> E{检查 MPS 后端}
    E -->|是| F[禁用原生 AMP]
    E -->|否| G[继续]
    F --> G
    G --> H[配置日志系统]
    H --> I{设置随机种子}
    I -->|提供| J[调用 set_seed]
    I -->|未提供| K[跳过]
    J --> K
    K --> L{主进程检查}
    L -->|是| M[创建输出目录]
    L -->|否| N[跳过]
    M --> O{检查 push_to_hub}
    O -->|是| P[创建 Hub 仓库]
    O -->|否| Q[跳过]
    P --> Q
    Q --> R[加载 Tokenizers]
    R --> S[导入 Text Encoder 类]
    S --> T[加载 DDPMScheduler]
    T --> U[应用零终端 SNR 调整]
    U --> V[加载 Text Encoders 和 VAE 和 UNet]
    V --> W[冻结非 LoRA 参数]
    W --> X[设置权重精度]
    X --> Y[移动模型到设备]
    Y --> Z1[配置 LoRA]
    Z1 --> Z2[启用 xformers 注意力]
    Z2 --> Z3[启用梯度检查点]
    Z3 --> Z4[注册模型保存/加载钩子]
    Z4 --> Z5[启用 TF32]
    Z5 --> Z6[计算学习率]
    Z6 --> Z7[创建优化器]
    Z7 --> Z8[加载数据集]
    Z8 --> Z9[预处理数据集]
    Z9 --> Z10[创建 DataLoader]
    Z10 --> Z11[创建学习率调度器]
    Z11 --> Z12[准备模型和优化器]
    Z12 --> Z13[初始化 Trackers]
    Z13 --> Z14[进入训练循环]
    Z14 --> Z15{遍历 epochs}
    Z15 --> Z16{遍历 batches}
    Z16 --> Z17[前向传播与损失计算]
    Z17 --> Z18[反向传播与参数更新]
    Z18 --> Z19{检查同步}
    Z19 -->|是| Z20[更新进度条与检查点]
    Z19 -->|否| Z16
    Z20 --> Z21{达到最大步数}
    Z21 -->|否| Z16
    Z21 -->|是| Z22[保存 LoRA 权重]
    Z22 --> Z23{运行最终验证}
    Z23 -->|是| Z24[执行验证]
    Z23 -->|否| Z25
    Z24 --> Z25
    Z25 --> Z26{推送到 Hub}
    Z26 -->|是| Z27[上传文件夹]
    Z26 -->|否| Z28[结束训练]
    Z27 --> Z28
```

#### 带注释源码

```python
def main(args):
    # 检查是否同时使用了 wandb 报告和 hub_token（安全风险）
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    # 构建日志目录路径
    logging_dir = Path(args.output_dir, args.logging_dir)

    # 配置 Accelerator 项目参数
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # 初始化 Accelerator（分布式训练、混合精度等）
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # 如果使用 MPS 后端，禁用原生 AMP
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # 配置日志格式并在每个进程输出
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    # 主进程设置详细日志，子进程只显示错误
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 如果提供了种子，设置随机种子以确保可复现性
    if args.seed is not None:
        set_seed(args.seed)

    # 处理仓库创建（主进程）
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # 如果需要推送到 Hub，创建远程仓库
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # 加载两个 Tokenizers（SDXL 使用双文本编码器）
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # 根据模型配置导入正确的文本编码器类
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # 加载噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # 定义零终端 SNR 强制函数（用于 Turbo 模式）
    def enforce_zero_terminal_snr(scheduler):
        # 修改自 https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L93
        # 原始实现 https://huggingface.co/papers/2305.08891
        # Turbo 需要零终端 SNR
        # 将 betas 转换为 alphas_bar_sqrt
        alphas = 1 - scheduler.betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # 保存旧值
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # 移位，使最后一步为零
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # 缩放，使第一步恢复到旧值
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        scheduler.alphas_cumprod = alphas_cumprod
        return

    # 如果是 Turbo 模式，强制零终端 SNR
    if args.is_turbo:
        enforce_zero_terminal_snr(noise_scheduler)

    # 加载文本编码器、VAE 和 UNet
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # 只训练额外的 LoRA 适配器层，冻结其他所有参数
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # 设置权重精度（用于混合精度训练）
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 将 UNet 和文本编码器移到设备并转换为指定精度
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # VAE 始终使用 float32 以避免 NaN 损失
    vae.to(accelerator.device, dtype=torch.float32)

    # 配置 LoRA
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # 添加适配器并确保可训练参数为 float32
    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        for param in unet.parameters():
            # 只将可训练参数（LoRA）转换为 fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    # 启用 xformers 高效注意力
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # 启用梯度检查点以节省内存
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # 创建自定义保存和加载钩子
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # 这里只有两种选择：UNet 注意力处理器层或 UNet 和文本编码器注意力层
            unet_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # 确保弹出权重，以免再次保存相应模型
                weights.pop()

            StableDiffusionXLLoraLoaderMixin.save_lora_weights(output_dir, unet_lora_layers=unet_lora_layers_to_save)

    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = StableDiffusionXLLoraLoaderMixin.lora_state_dict(input_dir)
        StableDiffusionXLLoraLoaderMixin.load_lora_into_unet(
            lora_state_dict, network_alphas=network_alphas, unet=unet_
        )

    # 注册保存和加载状态钩子
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # 为 Ampere GPU 启用 TF32 以加速训练
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 如果启用学习率缩放
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # 使用 8-bit Adam 以节省内存
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 创建优化器
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 数据集和 DataLoader 创建
    train_dataset = load_dataset(
        args.dataset_name,
        cache_dir=args.cache_dir,
        split=args.dataset_split_name,
    )

    # 预处理数据集
    train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.5], [0.5])

    def preprocess_train(examples):
        # 处理训练数据的预处理（图像、裁剪、翻转、归一化等）
        all_pixel_values = []
        images = [Image.open(io.BytesIO(im_bytes)).convert("RGB") for im_bytes in examples["jpg_0"]]
        original_sizes = [(image.height, image.width) for image in images]
        crop_top_lefts = []

        for col_name in ["jpg_0", "jpg_1"]:
            images = [Image.open(io.BytesIO(im_bytes)).convert("RGB") for im_bytes in examples[col_name]]
            if col_name == "jpg_1":
                # 需要将图像降级到相同分辨率
                images = [image.resize(original_sizes[i][::-1]) for i, image in enumerate(images)]
            pixel_values = [to_tensor(image) for image in images]
            all_pixel_values.append(pixel_values)

        # 组合像素值（成对处理）
        im_tup_iterator = zip(*all_pixel_values)
        combined_pixel_values = []
        for im_tup, label_0 in zip(im_tup_iterator, examples["label_0"]):
            if label_0 == 0:
                im_tup = im_tup[::-1]

            combined_im = torch.cat(im_tup, dim=0)

            # 调整大小
            combined_im = train_resize(combined_im)

            # 翻转
            if not args.no_hflip and random.random() < 0.5:
                combined_im = train_flip(combined_im)

            # 裁剪
            if not args.random_crop:
                y1 = max(0, int(round((combined_im.shape[1] - args.resolution) / 2.0)))
                x1 = max(0, int(round((combined_im.shape[2] - args.resolution) / 2.0)))
                combined_im = train_crop(combined_im)
            else:
                y1, x1, h, w = train_crop.get_params(combined_im, (args.resolution, args.resolution))
                combined_im = crop(combined_im, y1, x1, h, w)

            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            combined_im = normalize(combined_im)
            combined_pixel_values.append(combined_im)

        examples["pixel_values"] = combined_pixel_values
        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        tokens_one, tokens_two = tokenize_captions([tokenizer_one, tokenizer_two], examples)
        examples["input_ids_one"] = tokens_one
        examples["input_ids_two"] = tokens_two
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # 设置训练转换
        train_dataset = train_dataset.with_transform(preprocess_train)

    # 整理函数
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
        input_ids_two = torch.stack([example["input_ids_two"] for example in examples])

        return {
            "pixel_values": pixel_values,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # 调度器和训练步数计算
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # 创建学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # 准备模型和优化器
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # 重新计算总训练步数（因为 DataLoader 大小可能改变）
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # 重新计算训练 epoch 数
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 初始化 trackers
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_name, config=vars(args))

    # 训练！
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # 可能从之前的保存加载权重和状态
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # 获取最新的 checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # 进度条
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    unet.train()
    # 训练循环
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 图像编码为 latents
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                feed_pixel_values = torch.cat(pixel_values.chunk(2, dim=1))

                latents = []
                for i in range(0, feed_pixel_values.shape[0], args.vae_encode_batch_size):
                    latents.append(
                        vae.encode(feed_pixel_values[i : i + args.vae_encode_batch_size]).latent_dist.sample()
                    )
                latents = torch.cat(latents, dim=0)
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)

                # 采样噪声
                noise = torch.randn_like(latents).chunk(2)[0].repeat(2, 1, 1, 1)

                # 为每张图像采样随机时间步
                bsz = latents.shape[0] // 2
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long
                ).repeat(2)
                if args.is_turbo:
                    # 学习 4 步调度
                    timesteps_0_to_3 = timesteps % 4
                    timesteps = 250 * timesteps_0_to_3 + 249

                # 将噪声添加到模型输入
                noisy_model_input = noise_scheduler.add_noise(latents, noise, timesteps)

                # 时间 IDs
                def compute_time_ids(original_size, crops_coords_top_left):
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                ).repeat(2, 1)

                # 获取文本嵌入作为条件
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    [text_encoder_one, text_encoder_two], [batch["input_ids_one"], batch["input_ids_two"]]
                )
                prompt_embeds = prompt_embeds.repeat(2, 1, 1)
                pooled_prompt_embeds = pooled_prompt_embeds.repeat(2, 1)

                # 预测噪声残差
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds},
                ).sample

                # 根据预测类型获取目标
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # 计算损失
                model_losses = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
                model_losses_w, model_losses_l = model_losses.chunk(2)

                # 记录日志
                raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                model_diff = model_losses_w - model_losses_l

                # 参考模型预测
                accelerator.unwrap_model(unet).disable_adapters()
                with torch.no_grad():
                    ref_preds = unet(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds},
                    ).sample
                    ref_loss = F.mse_loss(ref_preds.float(), target.float(), reduction="none")
                    ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))

                    ref_losses_w, ref_losses_l = ref_loss.chunk(2)
                    ref_diff = ref_losses_w - ref_losses_l
                    raw_ref_loss = ref_loss.mean()

                # 重新启用适配器
                accelerator.unwrap_model(unet).enable_adapters()

                # 最终损失（DPO 损失）
                scale_term = -0.5 * args.beta_dpo
                inside_term = scale_term * (model_diff - ref_diff)
                loss = -1 * F.logsigmoid(inside_term).mean()

                # 隐式准确率
                implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                implicit_acc += 0.5 * (inside_term == 0).sum().float() / inside_term.size(0)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 检查加速器是否执行了优化步骤
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    # 检查点保存
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # 验证
                    if args.run_validation and global_step % args.validation_steps == 0:
                        log_validation(
                            args, unet=unet, vae=vae, accelerator=accelerator, weight_dtype=weight_dtype, epoch=epoch
                        )

            # 记录日志
            logs = {
                "loss": loss.detach().item(),
                "raw_model_loss": raw_model_loss.detach().item(),
                "ref_loss": raw_ref_loss.detach().item(),
                "implicit_acc": implicit_acc.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # 保存 LoRA 层
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        StableDiffusionXLLoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=None,
            text_encoder_2_lora_layers=None,
        )

        # 最终验证
        if args.run_validation:
            log_validation(
                args,
                unet=None,
                vae=vae,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                epoch=epoch,
                is_final_validation=True,
            )

        # 推送到 Hub
        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()
```

## 关键组件





### LoraConfig

用于配置LoRA（Low-Rank Adaptation）训练的超参数，包括秩、α值、初始化方式和目标模块

### UNet2DConditionModel

SDXL的条件UNet模型，用于预测噪声残差，执行扩散过程

### AutoencoderKL

VAE编码器，将图像像素转换为latent表示，支持SDXL的潜在空间

### DPO训练逻辑

实现Direct Preference Optimization算法，计算模型预测与参考模型预测之间的KL散度损失，并使用对比学习方法优化LoRA权重

### VAE批量编码

使用`chunk`方法对图像进行批量编码，通过`vae_encode_batch_size`参数控制每个编码批次的大小，实现高效的latent表示生成

### 文本编码器

支持双文本编码器（CLIPTextModel和CLIPTextModelWithProjection），将文本提示编码为条件embedding，用于引导图像生成

### 噪声调度器

DDPMScheduler实现扩散过程的前向噪声添加，支持`add_noise`方法进行噪声调度

### 梯度检查点

通过`enable_gradient_checkpointing`启用，以内存换计算效率，支持更深的网络训练

### xFormers优化

启用内存高效注意力机制，通过`enable_xformers_memory_efficient_attention`减少注意力计算的显存占用

### Accelerate分布式训练

使用Accelerator进行多GPU分布式训练和混合精度管理，支持梯度累积和模型状态保存/加载

### 数据预处理管道

包含图像resize、裁剪、翻转和归一化操作，支持中心裁剪和随机裁剪模式

### LoRA权重保存/加载

通过`StableDiffusionXLLoraLoaderMixin`保存和加载LoRA权重，使用`safe_tensors`格式存储

### 验证流程

使用`DiffusionPipeline`进行推理验证，支持带LoRA和不带LoRA的图像生成对比

### 检查点管理

实现检查点数量限制和自动清理机制，防止存储空间过度占用



## 问题及建议



### 已知问题

-   **类型声明错误**: `--beta_dpo` 参数定义为 `int` 类型（`default=5000`），但实际应该使用 `float` 类型，因为它是 DPO KL 散度惩罚的缩放系数。
-   **硬编码的验证提示**: `VALIDATION_PROMPTS` 列表在代码中硬编码，没有提供命令行参数来配置，降低了灵活性。
-   **数据预处理假设不灵活**: `preprocess_train` 函数硬编码依赖特定的列名 `jpg_0`、`jpg_1`、`label_0`，这些列名应该作为参数可配置。
-   **随机种子设置不完整**: 只调用了 `set_seed(args.seed)`，这只设置了 PyTorch 的随机种子，没有设置 NumPy 和 Python 内置 `random` 模块的种子，影响实验可复现性。
-   **缺失的数据集验证**: `train_dataset` 加载后没有验证必需的列是否存在，可能导致运行时错误。
-   **checkpoint 删除时序问题**: 在保存新 checkpoint 之前检查 checkpoint 数量，但删除操作发生在保存后，如果磁盘空间不足可能导致问题。
-   **缺失的超参数验证**: 没有验证 `beta_dpo`、`learning_rate` 等关键超参数值的合理性（如非负值检查）。

### 优化建议

-   **修复类型声明**: 将 `--beta_dpo` 的类型改为 `float`，默认值建议调整为合适的浮点数（如 `2500.0` 或更小的值）。
-   **添加验证提示参数**: 添加 `--validation_prompts` 参数，允许用户通过命令行传递自定义验证提示。
-   **参数化数据集列名**: 添加参数如 `--image_column_preferred`、`--image_column_rejected`、`--label_column` 来配置数据集列名。
-   **完善随机种子设置**: 在 `set_seed` 之后添加 `random.seed(args.seed)` 和 `np.random.seed(args.seed)`。
-   **添加数据集验证**: 在 `main` 函数开始处检查数据集是否包含必需的列，并给出友好的错误提示。
-   **改进 checkpoint 逻辑**: 在保存新 checkpoint 前先计算是否需要删除旧 checkpoint，并在删除后再保存。
-   **添加超参数范围检查**: 在 `parse_args` 或 `main` 函数中添加对关键超参数的验证逻辑。
-   **添加数据加载器预检查**: 在创建 `train_dataloader` 之前验证数据集不为空。

## 其它





### 设计目标与约束

**设计目标：**
- 实现SDXL LoRA模型的DPO（Direct Preference Optimization）微调训练
- 支持文本到图像生成任务的定制化训练
- 提供高效且可扩展的分布式训练能力
- 支持与HuggingFace Hub的模型推送集成

**约束条件：**
- 最低依赖diffusers 0.25.0.dev0版本
- 需要NVIDIA Ampere GPU或更高版本以支持bf16混合精度
- 训练数据必须包含jpg_0、jpg_1、caption、label_0字段
- LoRA rank维度默认为4，支持自定义配置

### 错误处理与异常设计

**关键异常处理点：**
1. **模型加载失败**：通过try-except捕获bitsandbytes导入失败，提示安装依赖
2. **数据集缺失**：parse_args中强制要求提供dataset_name，否则抛出ValueError
3. **xFormers兼容性**：检测版本为0.0.16时发出警告，不可用时抛出ValueError
4. **Hub Token安全**：检测到同时使用wandb和hub_token时抛出安全风险警告
5. **MPS设备处理**：自动检测并禁用MPS设备的AMP混合精度
6. **Checkpoint恢复**：处理checkpoint不存在的情况，自动创建新训练

### 数据流与状态机

**训练数据流：**
1. 数据加载 → load_dataset获取原始数据
2. 数据预处理 → preprocess_train进行图像转换、tokenization
3. VAE编码 → 将图像转换为latent表示（vae.encode + scaling_factor）
4. 噪声添加 → DDPMScheduler.add_noise进行前向扩散
5. 模型预测 → UNet预测噪声残差
6. 损失计算 → DPO损失函数结合reference model对比
7. 反向传播 → accelerator.backward + gradient clipping
8. 参数更新 → optimizer.step + lr_scheduler.step

**状态转换：**
- 初始状态 → 模型加载 → 训练中 → 验证 → Checkpoint保存 → 训练结束
- LoRA状态：disable_adapters（reference预测） ↔ enable_adapters（训练预测）

### 外部依赖与接口契约

**核心依赖库：**
- diffusers: 模型加载、pipeline构建、调度器
- transformers: 文本编码器、tokenizer
- accelerate: 分布式训练、混合精度、自动日志
- peft: LoRA配置与状态管理
- huggingface_hub: 模型推送
- wandb/tensorboard: 训练监控
- bitsandbytes: 8-bit Adam优化器（可选）
- xformers: 内存高效注意力（可选）

**关键接口契约：**
- parse_args(): 返回argparse.Namespace，包含所有训练超参数
- encode_prompt(): 输入text_encoders和text_input_ids_list，返回(prompt_embeds, pooled_prompt_embeds)
- log_validation(): 执行推理验证并记录到tracker
- main(args): 入口函数，执行完整训练流程

### 性能优化策略

**已实现的优化：**
1. **梯度累积**：gradient_accumulation_steps减少显存压力
2. **混合精度**：fp16/bf16自动选择，LoRA参数fp32训练
3. **梯度检查点**：gradient_checkpointing降低30%显存
4. **xFormers**：Memory Efficient Attention
5. **VAE分批编码**：vae_encode_batch_size控制批处理大小
6. **TF32加速**：allow_tf32启用TensorFloat32计算
7. **8-bit Adam**：可选的内存优化优化器

### 安全性考虑

1. **Hub Token保护**：禁止与wandb同时使用，防止token泄露
2. **MPS设备处理**：自动检测并禁用不兼容的AMP
3. **分布式安全**：local_rank环境变量正确处理
4. **Checkpoint清理**：自动管理checkpoint数量，防止磁盘溢出

### 兼容性考虑

1. **多版本支持**：检测diffusers最小版本要求
2. **SDXL变体**：支持variant参数加载不同模型变体
3. **VAE替换**：支持独立VAE模型加载
4. **Turbo模式**：特殊处理SDXL Turbo模型的zero terminal SNR要求
5. **LoRA兼容性**：使用StableDiffusionXXLoraLoaderMixin确保加载兼容性

### 配置与超参数说明

| 参数类别 | 关键参数 | 默认值 | 说明 |
|---------|---------|-------|------|
| 模型 | pretrained_model_name_or_path | 必填 | 预训练模型路径 |
| LoRA | rank | 4 | LoRA矩阵维度 |
| 优化 | beta_dpo | 5000 | DPO KL散度惩罚系数 |
| 训练 | train_batch_size | 4 | 单设备批大小 |
| 学习率 | learning_rate | 5e-4 | 初始学习率 |
| 调度器 | lr_scheduler | constant | 学习率调度类型 |
| 图像 | resolution | 1024 | 输入图像分辨率 |

### 潜在技术债务与优化空间

1. **硬编码验证提示**：VALIDATION_PROMPTS列表应可配置化
2. **魔法数字**：250、249等Turbo特定参数缺乏注释说明
3. **日志分散**：多处logger.info/print可统一管理
4. **验证流程冗余**：log_validation中重复的pipeline构建逻辑
5. **错误恢复**：网络加载失败缺乏重试机制
6. **类型提示**：部分函数缺少完整的类型注解
7. **数据增强**：仅支持horizontal flip，缺乏更多增强策略

### 部署与运维建议

1. **推荐环境**：Python 3.10+、CUDA 11.8+、24GB+ VRAM GPU
2. **典型配置**：单卡A100-40GB可运行batch_size=4的完整训练
3. **监控指标**：loss、raw_model_loss、ref_loss、implicit_acc、lr
4. **Checkpoint策略**：建议每500-1000步保存，保留最近3-5个
5. **验证频率**：每200步运行验证以跟踪生成质量变化


    