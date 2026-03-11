
# `diffusers\src\diffusers\pipelines\ltx2\pipeline_ltx2_latent_upsample.py` 详细设计文档

LTX2LatentUpsamplePipeline是一个用于视频latent上采样的扩散管道，通过VAE编码输入视频或latent，使用latent_upsampler模型进行上采样，并可选地应用AdaIN风格滤波和tone mapping压缩，最后通过VAE解码器生成最终视频。

## 整体流程

```mermaid
graph TD
A[开始] --> B{检查输入参数}
B --> C{是否有输入视频?}
C -- 是 --> D[预处理视频]
C -- 否 --> E[使用提供的latents]
D --> F[通过VAE编码视频获取latents]
E --> G[解包latents(如果需要)]
F --> H[准备latents]
G --> H
H --> I{latents已标准化?}
I -- 是 --> J[反标准化latents]
I -- 否 --> K[保持原样]
J --> L[转换为upsampler dtype]
K --> L
L --> M[调用latent_upsampler上采样]
M --> N{adain_factor > 0?}
N -- 是 --> O[应用AdaIN滤波]
N -- 否 --> P[使用上采样结果]
O --> Q{-tone_map_compression_ratio > 0?}
P --> Q
Q -- 是 --> R[应用tone mapping压缩]
Q -- 否 --> S{output_type == 'latent'?}
R --> S
S -- 是 --> T[返回latents]
S -- 否 --> U[VAE解码为视频]
U --> V[后处理视频]
T --> W[结束]
V --> W
```

## 类结构

```
DiffusionPipeline (基类)
└── LTX2LatentUpsamplePipeline
```

## 全局变量及字段


### `logger`
    
模块级日志记录器，用于输出调试和信息日志

类型：`logging.Logger`
    


### `EXAMPLE_DOC_STRING`
    
示例文档字符串，包含LTX2潜在上采样管道使用方法示例

类型：`str`
    


### `LTX2LatentUpsamplePipeline.model_cpu_offload_seq`
    
模型CPU卸载顺序，指定VAE和latent_upsampler的卸载优先级

类型：`str`
    


### `LTX2LatentUpsamplePipeline.vae`
    
VAE编码器/解码器，用于视频的编码和解码操作

类型：`AutoencoderKLLTX2Video`
    


### `LTX2LatentUpsamplePipeline.latent_upsampler`
    
Latent上采样模型，用于对潜在表示进行上采样处理

类型：`LTX2LatentUpsamplerModel`
    


### `LTX2LatentUpsamplePipeline.vae_spatial_compression_ratio`
    
VAE空间压缩比，用于计算潜在空间的高度和宽度

类型：`int`
    


### `LTX2LatentUpsamplePipeline.vae_temporal_compression_ratio`
    
VAE时间压缩比，用于计算潜在空间的帧数

类型：`int`
    


### `LTX2LatentUpsamplePipeline.video_processor`
    
视频处理器，用于视频的预处理和后处理操作

类型：`VideoProcessor`
    
    

## 全局函数及方法



### `retrieve_latents`

从编码器输出中提取潜在表示（latents）的工具函数，根据 `sample_mode` 参数决定是从潜在分布中采样还是取众数，或者直接返回预存的潜在张量。

参数：

- `encoder_output`：`torch.Tensor`，编码器的输出结果，通常包含 `latent_dist` 或 `latents` 属性
- `generator`：`torch.Generator | None`，可选的随机数生成器，用于控制采样过程的随机性
- `sample_mode`：`str`，采样模式，支持 `"sample"`（从分布中采样）或 `"argmax"`（取分布的众数），默认为 `"sample"`

返回值：`torch.Tensor`，提取出的潜在表示张量

#### 流程图

```mermaid
flowchart TD
    A[开始: retrieve_latents] --> B{encoder_output 是否有 latent_dist 属性?}
    B -->|是| C{sample_mode == 'sample'?}
    B -->|否| D{encoder_output 是否有 latents 属性?}
    C -->|是| E[返回 encoder_output.latent_dist.sample<br/>(generator)]
    C -->|否| F{sample_mode == 'argmax'?}
    F -->|是| G[返回 encoder_output.latent_dist.mode<br/>()]
    F -->|否| H[抛出 AttributeError]
    D -->|是| I[返回 encoder_output.latents]
    D -->|否| J[抛出 AttributeError]
    E --> K[结束]
    G --> K
    I --> K
    H --> K
    J --> K
```

#### 带注释源码

```python
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    """
    从编码器输出中提取潜在表示（latents）。

    该函数支持三种提取方式：
    1. 从 latent_dist 属性中采样（当 sample_mode="sample"）
    2. 从 latent_dist 属性中取众数（当 sample_mode="argmax"）
    3. 直接返回 latents 属性中存储的潜在张量

    Args:
        encoder_output: 编码器的输出，包含 latent_dist 或 latents 属性
        generator: 可选的随机生成器，用于控制采样随机性
        sample_mode: 采样模式，"sample" 或 "argmax"

    Returns:
        torch.Tensor: 提取出的潜在表示

    Raises:
        AttributeError: 当无法从 encoder_output 中获取潜在表示时
    """
    # 检查是否存在 latent_dist 属性，并使用采样模式
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # 从潜在分布中采样，返回采样结果
        return encoder_output.latent_dist.sample(generator)
    # 检查是否存在 latent_dist 属性，并使用 argmax 模式
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # 返回潜在分布的众数（最大值对应的索引）
        return encoder_output.latent_dist.mode()
    # 检查是否存在直接的 latents 属性
    elif hasattr(encoder_output, "latents"):
        # 直接返回预存的潜在张量
        return encoder_output.latents
    else:
        # 如果无法识别任何有效的潜在表示格式，抛出异常
        raise AttributeError("Could not access latents of provided encoder_output")
```



### `LTX2LatentUpsamplePipeline.__init__`

这是 `LTX2LatentUpsamplePipeline` 类的构造函数，用于初始化视频潜在空间上采样管道。它接收 VAE 模型和潜在上采样器模型，注册这些模块，并配置视频处理器的压缩比参数。

参数：

- `self`：隐式参数，管道实例本身
- `vae`：`AutoencoderKLLTX2Video`，用于视频编码/解码的 VAE 模型
- `latent_upsampler`：`LTX2LatentUpsamplerModel`，用于潜在空间上采样的模型

返回值：`None`，构造函数无返回值

#### 流程图

```mermaid
flowchart TD
    A[开始 __init__] --> B[调用 super().__init__]
    B --> C[调用 self.register_modules 注册 vae 和 latent_upsampler]
    C --> D{检查 vae 是否存在}
    D -->|是| E[使用 self.vae.spatial_compression_ratio]
    D -->|否| F[使用默认值 32]
    E --> G[设置 self.vae_spatial_compression_ratio]
    F --> G
    G --> H{检查 vae 是否存在}
    H -->|是| I[使用 self.vae.temporal_compression_ratio]
    H -->|否| J[使用默认值 8]
    I --> K[设置 self.vae_temporal_compression_ratio]
    J --> K
    K --> L[创建 VideoProcessor 并赋值给 self.video_processor]
    L --> M[结束 __init__]
```

#### 带注释源码

```python
def __init__(
    self,
    vae: AutoencoderKLLTX2Video,
    latent_upsampler: LTX2LatentUpsamplerModel,
) -> None:
    """
    初始化 LTX2LatentUpsamplePipeline 管道。
    
    Args:
        vae: 用于视频编码/解码的 VAE 模型 (AutoencoderKLLTX2Video)
        latent_upsampler: 用于潜在空间上采样的模型 (LTX2LatentUpsamplerModel)
    """
    # 调用父类 DiffusionPipeline 的初始化方法
    super().__init__()

    # 注册 VAE 和潜在上采样器模块，使其可通过 self.vae 和 self.latent_upsampler 访问
    self.register_modules(vae=vae, latent_upsampler=latent_upsampler)

    # 获取 VAE 的空间压缩比，如果 VAE 不存在则使用默认值 32
    # 这是视频帧到潜在空间的空间维度压缩比例
    self.vae_spatial_compression_ratio = (
        self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
    )
    
    # 获取 VAE 的时间压缩比，如果 VAE 不存在则使用默认值 8
    # 这是视频帧到潜在空间的时间维度（帧数）压缩比例
    self.vae_temporal_compression_ratio = (
        self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
    )
    
    # 创建视频处理器，用于视频的预处理和后处理
    # 使用空间压缩比作为 VAE 缩放因子
    self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
```



### `LTX2LatentUpsamplePipeline.prepare_latents`

该方法负责为潜在上采样管道准备潜在向量（latents）。它接受视频输入或预计算的潜在向量，对输入进行必要的形状变换和设备迁移，并可选地将潜在向量从打包形式解包为视频张量形式。

参数：

- `video`：`torch.Tensor | None`，要编码为潜在向量的视频张量
- `batch_size`：`int`，批次中视频的数量
- `num_frames`：`int`，视频的帧数
- `height`：`int`，视频的高度
- `width`：`int`，视频的宽度
- `spatial_patch_size`：`int`，用于解包潜在向量的空间 patch 大小
- `temporal_patch_size`：`int`，用于解包潜在向量的时间 patch 大小
- `dtype`：`torch.dtype | None`，输出潜在向量的数据类型
- `device`：`torch.device | None`，放置潜在向量的设备
- `generator`：`torch.Generator | None`，用于随机生成的生成器
- `latents`：`torch.Tensor | None`，可选的预计算潜在向量

返回值：`torch.Tensor`，准备好的潜在向量张量

#### 流程图

```mermaid
flowchart TD
    A[开始 prepare_latents] --> B{latents 是否已提供?}
    B -- 是 --> C{latents 维度是否为 3?}
    C -- 是 --> D[计算 latent_num_frames<br/>latent_height<br/>latent_width]
    D --> E[调用 _unpack_latents<br/>将 [B, S, D] 转换为 [B, C, F, H, W]]
    C -- 否 --> F[直接使用 latents]
    E --> G[将 latents 移动到指定 device 和 dtype]
    F --> G
    G --> H[返回 latents]
    
    B -- 否 --> I[将 video 移动到指定 device 和 dtype]
    I --> J{generator 是否为列表?}
    J -- 是 --> K[遍历批次<br/>使用对应 generator 编码每个视频]
    J -- 否 --> L[使用统一 generator 编码所有视频]
    K --> M[收集所有 init_latents]
    L --> M
    M --> N[沿批次维度拼接并转换数据类型]
    N --> O[返回 init_latents]
    
    H --> P[结束]
    O --> P
```

#### 带注释源码

```python
def prepare_latents(
    self,
    video: torch.Tensor | None = None,
    batch_size: int = 1,
    num_frames: int = 121,
    height: int = 512,
    width: int = 768,
    spatial_patch_size: int = 1,
    temporal_patch_size: int = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    latents: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    准备用于潜在上采样的潜在向量。
    
    如果提供了 latents，则将其解包为视频潜在向量格式；如果提供了 video，
    则使用 VAE 编码视频生成潜在向量。
    """
    
    # 情况1: 已提供 latents
    if latents is not None:
        # 检查 latents 是否为打包的 token 序列形式 [B, S, D]
        if latents.ndim == 3:
            # 计算潜在视频的空间维度
            # 时间压缩后的帧数 = (原始帧数 - 1) / 时间压缩比 + 1
            latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
            latent_height = height // self.vae_spatial_compression_ratio
            latent_width = width // self.vae_spatial_compression_ratio
            
            # 调用内部方法解包 latents
            # 从 [B, S, D] 转换为 [B, C, F, H, W] 格式的潜在视频
            latents = self._unpack_latents(
                latents, 
                latent_num_frames, 
                latent_height, 
                latent_width, 
                spatial_patch_size, 
                temporal_patch_size
            )
        
        # 将 latents 移动到目标设备并转换数据类型后返回
        return latents.to(device=device, dtype=dtype)

    # 情况2: 未提供 latents，需要从 video 编码生成
    # 将视频移动到指定设备和数据类型
    video = video.to(device=device, dtype=self.vae.dtype)
    
    # 处理多个 generator 的情况（每个视频一个 generator）
    if isinstance(generator, list):
        # 验证 generator 列表长度与批次大小是否匹配
        if len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 为批次中的每个视频分别编码
        init_latents = [
            # 使用辅助函数 retrieve_latents 从编码器输出中提取潜在向量
            retrieve_latents(self.vae.encode(video[i].unsqueeze(0)), generator[i]) 
            for i in range(batch_size)
        ]
    else:
        # 使用统一的 generator 编码所有视频
        init_latents = [
            retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator) 
            for vid in video
        ]

    # 沿批次维度拼接所有潜在向量，并转换为目标数据类型
    init_latents = torch.cat(init_latents, dim=0).to(dtype)
    
    # 注意: 潜在上采样器在非归一化的 latents 上操作，因此不在此处归一化
    # init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
    
    return init_latents
```



### `LTX2LatentUpsamplePipeline.adain_filter_latent`

对输入的潜在张量进行自适应实例归一化（AdaIN），通过参考潜在张量的统计信息（均值和标准差）来转换输入潜在变量的分布，实现风格迁移或特征适配的效果，最后根据混合因子将转换后的潜在变量与原始潜在变量进行线性插值。

参数：

- `self`：`LTX2LatentUpsamplePipeline` 类实例，隐式参数，表示管道对象本身
- `latents`：`torch.Tensor`，输入的待归一化潜在张量，通常为经过上采样后的潜在表示
- `reference_latents`：`torch.Tensor`，提供目标统计特性的参考潜在张量，用于计算目标均值和标准差
- `factor`：`float`，混合因子，控制在原始潜在张量和转换后潜在张量之间的插值比例，范围通常为 [-10.0, 10.0]，默认值为 1.0

返回值：`torch.Tensor`，返回经过 AdaIN 变换并混合后的潜在张量

#### 流程图

```mermaid
flowchart TD
    A[开始: adain_filter_latent] --> B[克隆输入latents到result]
    B --> C[遍历批次维度 i: 0 to batch_size-1]
    C --> D[遍历通道维度 c: 0 to channels-1]
    D --> E[计算参考潜在张量第i个样本第c通道的均值r_mean和标准差r_sd]
    E --> F[计算输入潜在张量第i个样本第c通道的均值i_mean和标准差i_sd]
    F --> G[result[i, c] = (result[i, c] - i_mean / i_sd * r_sd + r_mean]
    G --> D
    D --> C
    C --> H[使用lerp函数: result = lerp latents, result, factor]
    H --> I[返回变换后的result张量]
```

#### 带注释源码

```python
def adain_filter_latent(self, latents: torch.Tensor, reference_latents: torch.Tensor, factor: float = 1.0):
    """
    Applies Adaptive Instance Normalization (AdaIN) to a latent tensor based on statistics from a reference latent
    tensor.

    Args:
        latent (`torch.Tensor`):
            Input latents to normalize
        reference_latents (`torch.Tensor`):
            The reference latents providing style statistics.
        factor (`float`):
            Blending factor between original and transformed latent. Range: -10.0 to 10.0, Default: 1.0

    Returns:
        torch.Tensor: The transformed latent tensor
    """
    # 创建输入张量的深拷贝，避免修改原始数据
    result = latents.clone()

    # 遍历批次维度（通常为B）
    for i in range(latents.size(0)):
        # 遍历通道维度（通常为C，对应潜在空间的特征维度）
        for c in range(latents.size(1)):
            # 计算参考潜在张量在指定通道上的标准差和均值
            # 使用dim=None对整个空间/时间维度进行统计
            r_sd, r_mean = torch.std_mean(reference_latents[i, c], dim=None)
            
            # 计算输入潜在张量在指定通道上的标准差和均值
            i_sd, i_mean = torch.std_mean(result[i, c], dim=None)

            # 应用AdaIN变换：首先将输入归一化（减去均值除以标准差）
            # 然后使用参考统计量进行重新缩放和偏移
            result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean

    # 使用线性插值在原始潜在张量和变换后的潜在张量之间进行混合
    # factor=1.0表示完全使用变换结果，factor=0.0表示保留原始输入
    result = torch.lerp(latents, result, factor)
    return result
```



### `LTX2LatentUpsamplePipeline.tone_map_latents`

对潜在表示进行非线性色调映射，使用基于sigmoid的压缩来以感知平滑的方式降低其动态范围。这对于正则化高方差潜在表示或在生成过程中控制动态行为特别有用。

参数：

- `self`：隐式参数，LTX2LatentUpsamplePipeline类的实例方法调用所需的隐式参数。
- `latents`：`torch.Tensor`，输入的潜在张量，形状任意，期望值在[-1, 1]或[0, 1]范围内。
- `compression`：`float`，压缩强度，范围[0, 1]。0.0表示无色调映射（恒等变换），1.0表示完全压缩效果。

返回值：`torch.Tensor`，与输入形状相同的色调映射后的潜在张量。

#### 流程图

```mermaid
flowchart TD
    A[开始 tone_map_latents] --> B[计算缩放因子<br/>scale_factor = compression × 0.75]
    B --> C[计算绝对值<br/>abs_latents = torch.abs latents]
    C --> D[计算Sigmoid压缩项<br/>sigmoid_term = sigmoid4.0 × scale_factor × (abs_latents - 1.0)]
    D --> E[计算缩放系数<br/>scales = 1.0 - 0.8 × scale_factor × sigmoid_term]
    E --> F[应用缩放<br/>filtered = latents × scales]
    F --> G[返回filtered张量]
```

#### 带注释源码

```python
def tone_map_latents(self, latents: torch.Tensor, compression: float) -> torch.Tensor:
    """
    Applies a non-linear tone-mapping function to latent values to reduce their dynamic range in a perceptually
    smooth way using a sigmoid-based compression.

    This is useful for regularizing high-variance latents or for conditioning outputs during generation, especially
    when controlling dynamic behavior with a `compression` factor.

    Args:
        latents : torch.Tensor
            Input latent tensor with arbitrary shape. Expected to be roughly in [-1, 1] or [0, 1] range.
        compression : float
            Compression strength in the range [0, 1].
            - 0.0: No tone-mapping (identity transform)
            - 1.0: Full compression effect

    Returns:
        torch.Tensor
            The tone-mapped latent tensor of the same shape as input.
    """
    # 将[0-1]范围重新映射到[0-0.75]并一次性应用sigmoid压缩
    # 压缩因子由compression参数控制，最大为0.75
    scale_factor = compression * 0.75
    
    # 取绝对值，因为色调映射通常对正值和负值进行对称处理
    abs_latents = torch.abs(latents)

    # Sigmoid压缩：sigmoid将大值向0.2移动，小值保持接近1.0
    # 当scale_factor=0时，sigmoid项消失；当scale_factor=0.75时，效果最大
    # 公式：sigmoid(4.0 × scale_factor × (abs_latents - 1.0))
    # 当abs_latents > 1.0时，sigmoid输出较大值，对应压缩
    # 当abs_latents < 1.0时，sigmoid输出较小值，保持原状
    sigmoid_term = torch.sigmoid(4.0 * scale_factor * (abs_latents - 1.0))
    
    # 计算最终的缩放系数
    # 范围从1.0（compression=0）到0.2（compression=1）
    scales = 1.0 - 0.8 * scale_factor * sigmoid_term

    # 将原始latents乘以缩放系数
    filtered = latents * scales
    
    return filtered
```



### `LTX2LatentUpsamplePipeline._denormalize_latents`

该函数是一个静态方法，用于将已经标准化的潜在向量（latents）去规范化，即将其从标准化状态（零均值、单位方差）恢复到原始的数值范围。在扩散模型的潜在上采样管道中，用于在将潜在向量传递给上采样器之前将其反标准化。

参数：

- `latents`：`torch.Tensor`，输入的标准化潜在向量，形状为 [B, C, F, H, W]
- `latents_mean`：`torch.Tensor`，用于去标准化的均值向量
- `latents_std`：`torch.Tensor`，用于去标准化的标准差向量
- `scaling_factor`：`float`，可选参数，默认为 1.0，缩放因子

返回值：`torch.Tensor`，去规范化后的潜在向量，形状与输入相同

#### 流程图

```mermaid
flowchart TD
    A[开始 _denormalize_latents] --> B[将 latents_mean reshape 为 [1, -1, 1, 1, 1]]
    B --> C[将 latents_std reshape 为 [1, -1, 1, 1, 1]]
    C --> D[将 latents_mean 移动到 latents 的设备和数据类型]
    D --> E[将 latents_std 移动到 latents 的设备和数据类型]
    E --> F[计算: latents = latents * latents_std / scaling_factor + latents_mean]
    F --> G[返回去规范化的 latents]
```

#### 带注释源码

```python
@staticmethod
# Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._denormalize_latents
def _denormalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    # Denormalize latents across the channel dimension [B, C, F, H, W]
    # 将均值向量reshape为[1, C, 1, 1, 1]以匹配latents的维度顺序
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    # 将标准差向量reshape为[1, C, 1, 1, 1]以匹配latents的维度顺序
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    # 执行去规范化: 先乘以标准差并除以缩放因子,再加上均值
    # 这是标准化操作的逆操作: normalized = (x - mean) / (std * scaling_factor)
    # 去规范化: x = normalized * std * scaling_factor + mean
    latents = latents * latents_std / scaling_factor + latents_mean
    return latents
```



### `LTX2LatentUpsamplePipeline._unpack_latents`

将打包的latents张量（形状为[B, S, D]）解包并重塑为视频张量（形状为[B, C, F, H, W]）。这是`_pack_latents`方法的逆操作，用于将序列形式的latents转换为可处理的视频帧格式。

参数：

- `latents`：`torch.Tensor`，输入的打包latents张量，形状为[B, S, D]，其中B是批次大小，S是有效视频序列长度，D是有效特征维度
- `num_frames`：`int`，输入视频的帧数
- `height`：`int`，输入视频的高度（像素）
- `width`：`int`，输入视频的宽度（像素）
- `patch_size`：`int`，空间patch大小，默认为1
- `patch_size_t`：`int`，时间patch大小，默认为1

返回值：`torch.Tensor`，解包后的视频张量，形状为[B, C, F, H, W]，其中C是通道数，F是帧数，H是高度，W是宽度

#### 流程图

```mermaid
flowchart TD
    A[开始: 输入latents [B, S, D]] --> B[获取batch_size]
    B --> C[reshape: [B, num_frames, height, width, -1, patch_size_t, patch_size, patch_size]]
    C --> D[permute: 重新排列维度顺序]
    D --> E[flatten 6,7: 合并空间patch维度]
    E --> F[flatten 4,5: 合并时间patch维度]
    F --> G[flatten 2,3: 合并帧维度]
    G --> H[返回: latents [B, C, F, H, W]]
```

#### 带注释源码

```python
@staticmethod
# Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._unpack_latents
def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
    # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
    # what happens in the `_pack_latents` method.
    
    # 获取批次大小
    batch_size = latents.size(0)
    
    # 将latents从[B, S, D]重塑为[B, num_frames, height, width, -1, patch_size_t, patch_size, patch_size]
    # -1会自动计算特征维度
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    
    # 使用permute重新排列维度顺序: 从[0,1,2,3,4,5,6,7] -> [0,4,1,5,2,6,3,7]
    # 这样可以将通道维度提前，方便后续flatten操作
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    
    # 依次flatten:
    # 1. flatten(6,7): 将最后两个空间patch维度合并
    # 2. flatten(4,5): 将时间patch维度合并
    # 3. flatten(2,3): 将帧维度合并
    
    # 最终输出形状: [B, C, F, H, W]
    return latents
```



### `LTX2LatentUpsamplePipeline.check_inputs`

该方法用于验证LTX2LatentUpsamplePipeline的输入参数是否合法，包括检查height和width是否能被VAE空间压缩比整除、video和latents不能同时提供或同时为空、以及tone_map_compression_ratio必须在[0,1]范围内。

参数：

- `video`：`torch.Tensor | None`，待上采样的视频输入，与latents互斥
- `height`：`int`，输入视频的高度（像素）
- `width`：`int`，输入视频的宽度（像素）
- `latents`：`torch.Tensor | None`，预生成的视频潜在向量，与video互斥
- `tone_map_compression_ratio`：`float`，色调映射压缩比，用于控制潜在值的动态范围

返回值：`None`，该方法仅进行参数验证，不返回任何值

#### 流程图

```mermaid
flowchart TD
    A[开始 check_inputs] --> B{检查 height 和 width}
    B --> C{height % vae_spatial_compression_ratio == 0<br/>且 width % vae_spatial_compression_ratio == 0?}
    C -->|否| D[抛出 ValueError:<br/>height和width必须能被32整除]
    C -->|是| E{检查 video 和 latents}
    E --> F{video is not None<br/>且 latents is not None?}
    F -->|是| G[抛出 ValueError:<br/>只能提供video或latents之一]
    F -->|否| H{video is None<br/>且 latents is None?}
    H -->|是| I[抛出 ValueError:<br/>必须提供video或latents之一]
    H -->|否| J{检查 tone_map_compression_ratio}
    J --> K{0 <= tone_map_compression_ratio <= 1?}
    K -->|否| L[抛出 ValueError:<br/>tone_map_compression_ratio必须在[0, 1]范围内]
    K -->|是| M[验证通过]
    D --> N[结束]
    G --> N
    I --> N
    L --> N
    M --> N
```

#### 带注释源码

```python
def check_inputs(self, video, height, width, latents, tone_map_compression_ratio):
    """
    验证输入参数的有效性。
    
    检查以下条件：
    1. height和width必须能被VAE空间压缩比整除
    2. video和latents不能同时提供
    3. video和latents不能同时为空
    4. tone_map_compression_ratio必须在[0, 1]范围内
    
    Args:
        video: 待上采样的视频输入，与latents互斥
        height: 输入视频的高度（像素）
        width: 输入视频的宽度（像素）
        latents: 预生成的视频潜在向量，与video互斥
        tone_map_compression_ratio: 色调映射压缩比，用于控制潜在值的动态范围
    
    Raises:
        ValueError: 当任一验证条件不满足时抛出
    """
    # 检查height和width是否能被VAE空间压缩比整除
    # VAE的空间压缩比默认为32，用于确保潜在空间的正确对齐
    if height % self.vae_spatial_compression_ratio != 0 or width % self.vae_spatial_compression_ratio != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

    # 检查video和latents的互斥关系：不能同时提供两者
    if video is not None and latents is not None:
        raise ValueError("Only one of `video` or `latents` can be provided.")
    
    # 检查video和latents的必要性：至少需要提供其中之一
    if video is None and latents is None:
        raise ValueError("One of `video` or `latents` has to be provided.")

    # 检查tone_map_compression_ratio是否在有效范围内[0, 1]
    # 该参数用于控制色调映射的压缩强度
    if not (0 <= tone_map_compression_ratio <= 1):
        raise ValueError("`tone_map_compression_ratio` must be in the range [0, 1]")
```



### `LTX2LatentUpsamplePipeline.__call__`

该方法是 LTX2 潜空间上采样流水线的核心调用函数。它接收低分辨率的视频或对应的潜在向量（latents），首先对其进行预处理和标准化处理，然后通过专用的 `latent_upsampler` 模型进行上采样，以提升视频的时空分辨率。在上采样后，可选地应用 AdaIN 过滤进行风格融合或 Tone Mapping 进行动态范围压缩。最后，根据配置的条件解码（可选加入噪声混合）并通过 VAE 解码器将潜在向量重建为最终的视频格式。

参数：

-  `self`：隐式参数，管道实例本身。
-  `video`：`list[PipelineImageInput] | None`，需要进行上采样的输入视频（如 LTX 2.0 第一阶段的输出）。如果未提供，则必须提供 `latents`。
-  `height`：`int`，输入视频的像素高度（生成的视频将具有更大的分辨率）。
-  `width`：`int`，输入视频的像素宽度。
-  `num_frames`：`int`，输入视频的帧数。
-  `spatial_patch_size`：`int`，视频潜在向量的空间分块大小。当需要解包 `latents` 时使用。
-  `temporal_patch_size`：`int`，视频潜在向量的时间分块大小。当需要解包 `latents` 时使用。
-  `latents`：`torch.Tensor | None`，预生成的视频潜在向量。可以代替 `video` 参数提供。可以是形状为 `(batch_size, seq_len, hidden_dim)` 的补丁序列，也可以是形状为 `(batch_size, latent_channels, latent_frames, latent_height, latent_width)` 的视频潜在向量。
-  `latents_normalized`：`bool`，如果提供了 `latents`，该参数指示 `latents` 是否已使用 VAE 潜在向量均值和标准差进行了归一化。如果为 `True`，在送入潜水上采样器之前会对 `latents` 进行反归一化处理。
-  `decode_timestep`：`float | list[float]`，生成视频进行解码的时间步。
-  `decode_noise_scale`：`float | list[float] | None`，在解码时间步处随机噪声与去噪潜在向量之间的插值因子。
-  `adain_factor`：`float`，AdaIN（自适应实例归一化）混合因子，用于在上采样和原始潜在向量之间进行风格融合。范围 [-10.0, 10.0]；提供 0.0（默认）意味着不执行 AdaIN。
-  `tone_map_compression_ratio`：`float`，色调映射的压缩强度，用于降低潜在向量的动态范围。这对于规范化高方差潜在向量或在生成过程中调节输出很有用。应在 [0, 1] 范围内，其中 0.0（默认）表示不应用色调映射，1.0 表示完全压缩效果。
-  `generator`：`torch.Generator | list[torch.Generator] | None`，一个或多个 PyTorch 生成器，用于确保生成的可确定性。
-  `output_type`：`str | None`，生成图像的输出格式。可选择 [PIL](https://pillow.readthedocs.io/en/stable/)：`PIL.Image.Image` 或 `np.array`。默认为 `"pil"`。
-  `return_dict`：`bool`，是否返回 [`~pipelines.ltx.LTXPipelineOutput`] 而不是普通元组。默认为 `True`。

返回值：`LTXPipelineOutput | tuple`，如果 `return_dict` 为 `True`，则返回 [`~pipelines.ltx.LTXPipelineOutput`]，否则返回一个元组，其中第一个元素是上采样后的视频。

#### 流程图

```mermaid
graph TD
    A[Start __call__] --> B[check_inputs: 验证 height, width, video/latents, tone_map_compression_ratio]
    B --> C{Is video provided?}
    C -->|Yes| D[batch_size = 1]
    C -->|No| E[batch_size = latents.shape[0]]
    D --> F[Preprocess video: 调整 num_frames, 预处理, 移至 device]
    F --> G[prepare_latents: 编码 video 或处理 latents]
    E --> G
    G --> H{latents_normalized?}
    H -->|Yes| I[_denormalize_latents: 使用 vae.mean/std 反归一化]
    H -->|No| J[latents.to: 转换为 latent_upsampler 的 dtype]
    I --> J
    J --> K[latent_upsampler forward: 执行上采样]
    K --> L{adain_factor > 0?}
    L -->|Yes| M[adain_filter_latent: 应用 AdaIN 风格迁移]
    L -->|No| N[latents = latents_upsampled]
    M --> N
    N --> O{tone_map_compression_ratio > 0?}
    O -->|Yes| P[tone_map_latents: 应用色调映射压缩]
    O -->|No| Q{output_type == 'latent'?}
    P --> Q
    Q -->|Yes| R[video = latents: 直接输出潜在向量]
    Q -->|No| S{VAE timestep_conditioning?}
    S -->|No| T[vae.decode: 无时间步直接解码]
    S -->|Yes| U[Noise mixing: 混合 noise 与 latents 基于 decode_noise_scale]
    U --> V[vae.decode: 带时间步解码]
    T --> W[postprocess_video: 后处理视频格式]
    V --> W
    W --> X[maybe_free_model_hooks: 释放模型资源]
    X --> Y{return_dict?}
    Y -->|Yes| Z[Return LTXPipelineOutput]
    Y -->|No| AA[Return tuple(video,)]
    R --> X
```

#### 带注释源码

```python
@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(
    self,
    video: list[PipelineImageInput] | None = None,
    height: int = 512,
    width: int = 768,
    num_frames: int = 121,
    spatial_patch_size: int = 1,
    temporal_patch_size: int = 1,
    latents: torch.Tensor | None = None,
    latents_normalized: bool = False,
    decode_timestep: float | list[float] = 0.0,
    decode_noise_scale: float | list[float] | None = None,
    adain_factor: float = 0.0,
    tone_map_compression_ratio: float = 0.0,
    generator: torch.Generator | list[torch.Generator] | None = None,
    output_type: str | None = "pil",
    return_dict: bool = True,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        video (`list[PipelineImageInput]`, *optional*)
            ...
    """

    # 1. 输入校验：检查尺寸、输入源冲突、参数范围
    self.check_inputs(
        video=video,
        height=height,
        width=width,
        latents=latents,
        tone_map_compression_ratio=tone_map_compression_ratio,
    )

    # 2. 确定批处理大小
    # 注意：目前不支持批量视频输入
    if video is not None:
        batch_size = 1
    else:
        batch_size = latents.shape[0]
    
    # 获取执行设备（例如 CUDA:0）
    device = self._execution_device

    # 3. 视频预处理（如果提供）
    if video is not None:
        # 获取帧数
        num_frames = len(video)
        # VAE 的时间压缩比通常要求帧数为 k * ratio + 1，进行调整和截断
        if num_frames % self.vae_temporal_compression_ratio != 1:
            num_frames = (
                num_frames // self.vae_temporal_compression_ratio * self.vae_temporal_compression_ratio + 1
            )
            video = video[:num_frames]
            logger.warning(
                f"Video length expected to be of the form `k * {self.vae_temporal_compression_ratio} + 1` but is {len(video)}. Truncating to {num_frames} frames."
            )
        # 预处理：缩放、归一化等
        video = self.video_processor.preprocess_video(video, height=height, width=width)
        video = video.to(device=device, dtype=torch.float32)

    # 4. 准备 Latents
    # 如果提供了 video，则编码为 latents；如果直接提供了 latents，则进行形状整理
    latents_supplied = latents is not None
    latents = self.prepare_latents(
        video=video,
        batch_size=batch_size,
        num_frames=num_frames,
        height=height,
        width=width,
        spatial_patch_size=spatial_patch_size,
        temporal_patch_size=temporal_patch_size,
        dtype=torch.float32,
        device=device,
        generator=generator,
        latents=latents,
    )

    # 5. 反归一化（如果需要）
    # 如果 latents 来自已经归一化的流水线（如 LTX2ImageToVideoPipeline），这里需要还原
    if latents_supplied and latents_normalized:
        latents = self._denormalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )
    
    # 转换为上采样器所需的精度（如 bfloat16）
    latents = latents.to(self.latent_upsampler.dtype)
    
    # 6. 执行上采样
    latents_upsampled = self.latent_upsampler(latents)

    # 7. 可选：AdaIN 风格过滤
    # 用于保留原始视频的某些风格特征
    if adain_factor > 0.0:
        latents = self.adain_filter_latent(latents_upsampled, latents, adain_factor)
    else:
        latents = latents_upsampled

    # 8. 可选：Tone Mapping
    # 用于处理高方差或特定动态范围的潜在向量
    if tone_map_compression_ratio > 0.0:
        latents = self.tone_map_latents(latents, tone_map_compression_ratio)

    # 9. 输出处理
    if output_type == "latent":
        # 如果只需要潜在向量（用于后续处理），直接返回
        video = latents
    else:
        # 需要解码为视频像素
        # 检查是否需要时间步条件
        if not self.vae.config.timestep_conditioning:
            timestep = None
        else:
            # 需要混合噪声以在特定时间步解码
            noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            if not isinstance(decode_timestep, list):
                decode_timestep = [decode_timestep] * batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * batch_size

            timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
            decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                :, None, None, None, None
            ]
            # 混合公式：latents = (1 - scale) * latents + scale * noise
            latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

        # VAE 解码
        video = self.vae.decode(latents, timestep, return_dict=False)[0]
        # 后处理：转换格式（PIL / numpy）
        video = self.video_processor.postprocess_video(video, output_type=output_type)

    # 10. 资源清理：offload 模型
    self.maybe_free_model_hooks()

    # 11. 返回结果
    if not return_dict:
        return (video,)

    return LTXPipelineOutput(frames=video)
```

## 关键组件




### 张量索引与形状变换

负责将输入的潜在变量张量从序列形式（token seq [B, S, D]）解包重塑为视频张量形式（latent video [B, C, F, H, W]），支持空间和时间patch维度处理，确保与VAE压缩率对齐。

### 潜在变量反归一化

提供静态方法_denormalize_latents，使用VAE的latents_mean和latents_std对潜在变量进行反归一化处理，恢复到原始潜在空间分布，支持scaling_factor调节。

### 色调映射压缩

tone_map_latents方法实现基于sigmoid的非线性色调映射，通过compression参数（0-1范围）控制压缩强度，将高动态范围的潜在值映射到感知平滑的输出，用于正则化高方差潜在变量或调节生成动态行为。

### AdaIN风格迁移

adain_filter_latent方法实现自适应实例归一化，从参考潜在变量中提取风格统计量（均值和标准差），通过factor参数（-10.0到10.0）控制原始与转换潜在变量之间的混合程度，实现潜在空间风格控制。

### 潜在变量获取

retrieve_latents全局函数封装了从编码器输出中提取潜在变量的逻辑，支持多种模式（sample/argmax）以及直接访问latents属性的fallback机制，为VAE编码提供统一的潜在变量提取接口。

### 视频预处理与后处理

VideoProcessor组件负责将输入图像列表预处理为张量格式，并在生成后进行后处理转换（支持pil/np/latent等输出类型），同时处理视频长度与VAE时间压缩率的兼容性检查与截断。


## 问题及建议



### 已知问题

-   **硬编码的压缩比默认值**：在 `__init__` 方法中，当 VAE 不存在时使用硬编码的默认值（spatial=32, temporal=8），如果 VAE 配置不同会导致潜在的不一致行为
-   **AdaIN 因子缺少输入验证**：`adain_factor` 参数的有效范围是 [-10.0, 10.0]，但在 `check_inputs` 方法中未对此进行验证，仅在 `adain_filter_latent` 方法中使用
-   **批量视频输入未完全支持**：代码注释表明 "Batched video input is not yet tested/supported"，但没有在输入阶段进行阻止或警告，可能导致意外行为
-   **循环实现效率低下**：`adain_filter_latent` 方法使用双层 Python for 循环遍历 latents 维度，在 GPU 上运行效率较低，应使用向量化操作替代
-   **代码重复**：`_denormalize_latents` 和 `_unpack_latents` 方法标记为从其他 pipeline 复制过来，表明存在代码重复问题，应考虑提取到共享模块
-   **类型注解不完整**：部分函数参数缺少类型注解，如 `retrieve_latents` 中的 `sample_mode` 参数
-   **API 复杂性**：可选参数 `decode_timestep` 和 `decode_noise_scale` 可以是 float 或 list，增加了 API 的复杂度和用户理解成本

### 优化建议

-   在 `check_inputs` 方法中添加 `adain_factor` 的范围验证，确保输入在 [-10.0, 10.0] 范围内
-   将 `adain_filter_latent` 方法改写为完全向量化的实现，避免 Python for 循环，利用 torch 的逐元素操作和广播机制
-   考虑在输入阶段对批量视频输入进行明确检查和警告，或者移除该限制并完善支持
-   将复用的静态方法 `_denormalize_latents` 和 `_unpack_latents` 提取到共享的工具类或基类中，减少代码重复
-   为所有函数参数添加完整的类型注解，提高代码可读性和 IDE 支持
-   考虑简化 API 设计，为 `decode_timestep` 和 `decode_noise_scale` 参数提供更清晰的文档说明或使用更严格的类型约束

## 其它




### 设计目标与约束

本Pipeline的核心设计目标是实现视频latent的高质量上采样，将低分辨率的latent表示提升至高分辨率，为LTX-2视频生成模型提供高效的latent上采样能力。主要约束包括：输入视频长度必须满足`k * vae_temporal_compression_ratio + 1`的格式要求，height和width必须能被vae_spatial_compression_ratio（32）整除，tone_map_compression_ratio必须在[0,1]范围内，adain_factor应在[-10.0, 10.0]范围内。

### 错误处理与异常设计

代码中的错误处理主要通过check_inputs方法实现，包含以下异常场景：当height或width不能被32整除时抛出ValueError；当video和latents同时提供或都未提供时抛出ValueError；当tone_map_compression_ratio超出[0,1]范围时抛出ValueError；当提供的generator列表长度与batch_size不匹配时抛出ValueError。对于外部依赖（如encoder_output）的latents获取失败时，通过raise AttributeError("Could not access latents of provided encoder_output")向上传递错误。

### 数据流与状态机

Pipeline的数据流遵循以下状态转换：首先检查输入有效性（check_inputs），然后根据是否有video输入确定batch_size，接着对video进行预处理（video_processor.preprocess_video），之后准备latents（prepare_latents），可选地进行denormalize，然后通过latent_upsampler进行上采样，接着可选地应用AdaIN滤波和tone mapping，最后通过VAE解码得到输出视频，最后进行后处理和模型卸载。

### 外部依赖与接口契约

本Pipeline依赖以下核心组件：AutoencoderKLLTX2Video（VAE模型）用于视频编解码，LTX2LatentUpsamplerModel用于latent上采样，VideoProcessor用于视频预处理和后处理，PipelineImageInput类型定义输入视频格式，LTXPipelineOutput定义输出格式。输入video应为PIL图像列表或numpy数组，latents支持packed形式[B, S, D]或unpacked形式[B, C, F, H, W]。输出默认返回PIL图像，可通过output_type参数调整为"np"（numpy数组）、"latent"或其他格式。

### 性能考虑

当前实现中，预处理video时强制使用torch.float32设备转移，latents准备时默认使用torch.float32。Batch处理video的能力标记为"not yet tested/supported"，存在优化空间。VideoProcessor的vae_scale_factor基于vae_spatial_compression_ratio自动计算。模型卸载通过maybe_free_model_hooks()在流程结束时触发。

### 安全性考虑

本Pipeline主要处理图像和视频数据，无明显安全风险。需要注意的点包括：generator参数用于确保可复现性但不影响安全性；模型卸载（model offload）有助于释放GPU显存资源；CPU offload通过enable_model_cpu_offload()调用实现。

### 兼容性说明

本代码源自Apache License 2.0开源项目，与HuggingFace diffusers库兼容。部分方法从diffusers.pipelines.stable_diffusion和diffusers.pipelines.ltx2直接复制（如retrieve_latents、_denormalize_latents、_unpack_latents），确保与LTX2Pipeline和Stable Diffusion Pipeline系列的一致性。输出类型支持"pil"、"np"和"latent"，与diffusers标准接口对齐。

### 使用示例与测试场景

基本使用场景：接收LTX2ImageToVideoPipeline生成的视频作为输入，通过latent_upsampler上采样，输出更高分辨率的视频。高级功能包括：通过adain_factor控制AdaIN滤波实现风格迁移，通过tone_map_compression_ratio调节latent动态范围，通过decode_timestep和decode_noise_scale控制解码阶段的噪声混合。测试应覆盖：不同分辨率输入（需能被32整除）、不同帧数输入（需满足temporal_compression_ratio+1格式）、packed和unpacked两种latents格式、normalized和denormalized latents输入等边界情况。

    