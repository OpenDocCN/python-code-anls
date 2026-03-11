
# `diffusers\src\diffusers\pipelines\ltx2\export_utils.py` 详细设计文档

这是一个视频编码工具模块，使用PyAV库将图像帧序列（支持PIL Images、numpy数组或torch tensor格式）与音频数据（torch tensor）编码成MP4格式的视频文件，支持分块编码以适应不同的视频VAE配置。

## 整体流程

```mermaid
graph TD
    A[开始 encode_video] --> B{检查video类型}
    B -->|list[PIL.Image]| C[转换为numpy再转torch]
    B -->|np.ndarray| D[归一化处理并转torch]
    B -->|torch.Tensor| E[直接处理]
    C --> F[torch.tensor_split分块]
    D --> F
    E --> F
    F --> G[创建av.Container和VideoStream]
    G --> H{audio is not None?}
    H -->|是| I[创建AudioStream]
    H -->|否| J[跳过音频处理]
    I --> K[遍历视频块编码帧]
    J --> K
    K --> L[Flush编码器]
    L --> M{audio is not None?}
    M -->|是| N[_write_audio写入音频]
    M -->|否| O[container.close结束]
    N --> O
    O --> P[结束]
```

## 类结构

```
无类定义（纯函数模块）
└── 全局函数
    ├── _prepare_audio_stream (内部函数)
    ├── _resample_audio (内部函数)
    ├── _write_audio (内部函数)
    └── encode_video (公开API)
```

## 全局变量及字段


### `logger`
    
日志记录器实例，用于输出模块运行时的日志信息

类型：`logging.Logger`
    


### `_CAN_USE_AV`
    
布尔值标志，检查PyAV库是否可用以支持LTX 2.0视频导出功能

类型：`bool`
    


    

## 全局函数及方法



### `_prepare_audio_stream`

该函数用于准备音频编码流，通过PyAV库创建AAC编码的音频流，并配置采样率、声道布局和时间基等关键参数，确保音频数据能够正确编码并写入到容器中。

参数：

- `container`：`av.container.Container`，PyAV容器对象，用于写入音频流
- `audio_sample_rate`：`int`，音频采样率（如24000表示24kHz）

返回值：`av.audio.AudioStream`，配置完成后的音频流对象，可用于后续音频帧的编码和写入

#### 流程图

```mermaid
flowchart TD
    A[开始: _prepare_audio_stream] --> B[接收container和audio_sample_rate参数]
    B --> C[container.add_stream添加aac音频流]
    C --> D[设置codec_context.sample_rate]
    D --> E[设置codec_context.layout为stereo]
    E --> F[设置codec_context.time_base为Fraction{1, sample_rate}]
    F --> G[返回配置好的audio_stream]
```

#### 带注释源码

```python
def _prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    """
    Prepare the audio stream for writing.
    
    创建一个AAC编码的音频流，并配置相关的编解码器参数。
    
    参数:
        container: PyAV容器对象，用于写入音频流
        audio_sample_rate: 音频采样率，单位Hz
    
    返回:
        配置好的音频流对象
    """
    # 使用aac编码器创建音频流，rate参数指定采样率
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    
    # 设置采样率，确保与输入音频一致
    audio_stream.codec_context.sample_rate = audio_sample_rate
    
    # 设置声道布局为立体声
    audio_stream.codec_context.layout = "stereo"
    
    # 设置时间基为1/sample_rate，用于PTS/DTS时间戳计算
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    
    # 返回配置好的音频流对象
    return audio_stream
```



### `_resample_audio`

该函数负责将输入的音频帧重采样到目标格式（由编码器决定），然后编码并写入到容器中。它是音频写入流程中的关键步骤，确保输入音频的采样率、布局和格式与AAC编码器兼容。

参数：

- `container`：`av.container.Container`，输出容器，用于多路复用编码后的音频数据包
- `audio_stream`：`av.audio.AudioStream`，音频流，包含编码器上下文，定义了目标格式、布局和采样率
- `frame_in`：`av.AudioFrame`，输入的音频帧，待重采样的原始音频数据

返回值：`None`，该函数不返回任何值，而是直接将编码后的音频数据包写入容器

#### 流程图

```mermaid
flowchart TD
    A[开始 _resample_audio] --> B[获取 audio_stream.codec_context]
    B --> C[确定目标格式: format = cc.format 或 'fltp']
    C --> D[确定目标布局: layout = cc.layout 或 'stereo']
    D --> E[确定目标采样率: rate = cc.sample_rate 或 frame_in.sample_rate]
    E --> F[创建 AudioResampler]
    F --> G[初始化 audio_next_pts = 0]
    G --> H{遍历 resample 结果帧}
    H -->|是| I{检查 rframe.pts 是否为 None}
    I -->|是| J[设置 rframe.pts = audio_next_pts]
    I -->|否| K[跳过设置pts]
    J --> L[更新 audio_next_pts += rframe.samples]
    K --> L
    L --> M[设置 rframe.sample_rate = frame_in.sample_rate]
    M --> N[编码: audio_stream.encode(rframe)]
    N --> O[多路复用: container.mux]
    O --> H
    H -->|否| P[刷新编码器: audio_stream.encode]
    P --> Q{遍历编码器flush包}
    Q -->|是| R[多路复用: container.mux]
    R --> Q
    Q -->|否| S[结束]
```

#### 带注释源码

```python
def _resample_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame
) -> None:
    """
    重采样音频帧到目标格式并编码写入容器
    
    参数:
        container: PyAV容器，用于写入编码后的音频
        audio_stream: 音频流，包含编码器上下文
        frame_in: 输入的音频帧
    """
    # 获取音频流的编解码器上下文，从中提取编码器所需的参数
    cc = audio_stream.codec_context

    # 使用编码器的格式/布局/采样率作为目标参数
    # AAC 编码器通常使用 fltp (float planar) 格式
    target_format = cc.format or "fltp"  # AAC → usually fltp
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    # 创建音频重采样器，将输入帧转换为编码器所需的格式
    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    # 初始化音频_pts（显示时间戳），用于设置每个输出帧的时间戳
    audio_next_pts = 0
    # 遍历重采样后的每一帧
    for rframe in audio_resampler.resample(frame_in):
        # 如果帧没有pts（显示时间戳），则使用计算的值
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        # 更新pts计数器，加上本帧的样本数
        audio_next_pts += rframe.samples
        # 确保采样率与输入帧一致
        rframe.sample_rate = frame_in.sample_rate
        # 编码重采样后的帧并写入容器
        container.mux(audio_stream.encode(rframe))

    # 刷新音频编码器，获取编码器缓冲区中剩余的包
    for packet in audio_stream.encode():
        container.mux(packet)
```



### `_write_audio`

将 PyTorch 张量格式的音频数据转换为 PyAV 的 AudioFrame，并写入到容器中。

参数：

- `container`：`av.container.Container`，PyAV 容器对象，用于写入音频数据
- `audio_stream`：`av.audio.AudioStream`，已配置的音频流，用于编码和复用音频帧
- `samples`：`torch.Tensor`，音频样本张量，形状为 `[channels, samples]` 或 `[samples,]`
- `audio_sample_rate`：`int`，音频采样率（Hz）

返回值：`None`，该函数通过副作用将音频数据写入容器

#### 流程图

```mermaid
flowchart TD
    A[开始 _write_audio] --> B{检查 samples 维度}
    B -->|dim == 1| C[添加单通道维度]
    C --> D
    B -->|dim > 1| D{检查通道顺序}
    D -->|shape[1] != 2 且 shape[0] == 2| E[转置样本]
    E --> F
    D -->|shape[1] == 2| F{验证通道数}
    F -->|shape[1] != 2| G[抛出 ValueError]
    F -->|shape[1] == 2| H{检查数据类型}
    H -->|dtype != int16| I[裁剪到 [-1.0, 1.0] 并转换为 int16]
    H -->|dtype == int16| J
    I --> J[重塑为 1D 并转为 NumPy]
    J --> K[创建 av.AudioFrame]
    K --> L[设置采样率]
    L --> M[调用 _resample_audio]
    M --> N[结束]
    
    G --> N
```

#### 带注释源码

```python
def _write_audio(
    container: av.container.Container,
    audio_stream: av.audio.AudioStream,
    samples: torch.Tensor,
    audio_sample_rate: int,
) -> None:
    """
    将 PyTorch 音频张量转换为 PyAV AudioFrame 并写入容器。
    
    参数:
        container: PyAV 容器对象
        audio_stream: 已配置的音频流
        samples: 音频样本张量，形状为 [channels, samples] 或 [samples,]
        audio_sample_rate: 音频采样率
    """
    # 处理单通道输入，添加通道维度使其成为 2D 张量 [1, samples]
    if samples.ndim == 1:
        samples = samples[:, None]

    # 处理行向量形式的立体声输入 [2, samples]，转置为列形式
    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T

    # 验证音频通道数，必须为立体声（2 通道）
    if samples.shape[1] != 2:
        raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")

    # 将浮点音频数据转换为 16 位整数格式
    # PyAV 的 AudioFrame.from_ndarray 需要 int16 格式
    if samples.dtype != torch.int16:
        # 裁剪到 [-1.0, 1.0] 范围，防止削波
        samples = torch.clip(samples, -1.0, 1.0)
        # 缩放到 int16 范围 [-32767, 32767]
        samples = (samples * 32767.0).to(torch.int16)

    # 将 PyTorch 张量转换为 NumPy 数组并创建 PyAV 音频帧
    # 形状重塑为 [1, total_samples] 表示单帧包含所有样本
    frame_in = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",  # 16 位有符号整数
        layout="stereo",  # 立体声布局
    )
    # 设置采样率
    frame_in.sample_rate = audio_sample_rate

    # 调用内部重采样函数处理音频编码和复用
    _resample_audio(container, audio_stream, frame_in)
```



### `encode_video`

该函数是视频编码的核心入口，负责将视频帧（支持 PIL Image、NumPy 数组、PyTorch 张量或迭代器格式）与音频数据（PyTorch 张量）整合编码为最终的 MP4 输出文件。函数内部处理了多种输入格式的转换、视频分块编码以及音频流的创建与写入，并利用 PyAV 库完成实际的编解码工作。

参数：

- `video`：`list[PIL.Image.Image] | np.ndarray | torch.Tensor | Iterator[torch.Tensor]`，视频帧数据，形状为 [frames, height, width, channels]，像素值范围根据类型不同可能为 [0, 255]（PIL/numpy int 或 torch uint8）或 [0, 1]（numpy float）
- `fps`：`int`，输出视频的帧率（FPS）
- `audio`：`torch.Tensor`，音频波形数据，形状为 [audio_channels, samples]，可选参数
- `audio_sample_rate`：`int`，音频采样率，LTX 2.0 典型值为 24000 (24 kHz)，当 audio 不为 None 时必须提供
- `output_path`：`str`，输出视频文件的保存路径
- `video_chunks_number`：`int`，视频分块数量，默认为 1，用于分块编码以适应视频 VAE 的 tiling 配置

返回值：`None`，函数直接写入文件，无返回值

#### 流程图

```mermaid
flowchart TD
    A[开始 encode_video] --> B{video 类型检查}
    B -->|PIL Image 列表| C[转换为 numpy 数组再转 torch.Tensor]
    B -->|numpy.ndarray| D{检查是否为归一化 [0,1]}
    D -->|是| E[乘以 255 并转为 uint8]
    D -->|否| F[警告并保持原值]
    E --> G[转换为 torch.Tensor]
    F --> G
    C --> G
    G --> H[沿帧维度分块为 video_chunks_number]
    H --> I[获取首块获取 height/width]
    I --> J[打开输出容器 av.open]
    J --> K[创建视频流 libx264 yuv420p]
    K --> L{audio 不为 None?}
    L -->|是| M[准备音频流 _prepare_audio_stream]
    L -->|否| N[跳过音频准备]
    M --> O
    N --> O
    O[遍历视频块编码] --> P[每帧转为 av.VideoFrame]
    P --> Q[编码帧并 mux 到容器]
    Q --> R{还有视频块?}
    R -->|是| O
    R -->|否| S[刷新视频编码器]
    S --> T{audio 不为 None?}
    T -->|是| U[写入音频 _write_audio]
    T -->|否| V[关闭容器]
    U --> V
    V[结束]
```

#### 带注释源码

```python
def encode_video(
    video: list[PIL.Image.Image] | np.ndarray | torch.Tensor | Iterator[torch.Tensor],
    fps: int,
    audio: torch.Tensor,
    audio_sample_rate: int,
    output_path: str,
    video_chunks_number: int = 1,
) -> None:
    """
    Encodes a video with audio using the PyAV library. Based on code from the original LTX-2 repo:
    https://github.com/Lightricks/LTX-2/blob/4f410820b198e05074a1e92de793e3b59e9ab5a0/packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py#L182

    Args:
        video (`List[PIL.Image.Image]` or `np.ndarray` or `torch.Tensor`):
            A video tensor of shape [frames, height, width, channels] with integer pixel values in [0, 255]. If the
            input is a `np.ndarray`, it is expected to be a float array with values in [0, 1] (which is what pipelines
            usually return with `output_type="np"`).
        fps (`int`)
            The frames per second (FPS) of the encoded video.
        audio (`torch.Tensor`, *optional*):
            An audio waveform of shape [audio_channels, samples].
        audio_sample_rate: (`int`, *optional*):
            The sampling rate of the audio waveform. For LTX 2, this is typically 24000 (24 kHz).
        output_path (`str`):
            The path to save the encoded video to.
        video_chunks_number (`int`, *optional*, defaults to `1`):
            The number of chunks to split the video into for encoding. Each chunk will be encoded separately. The
            number of chunks to use often depends on the tiling config for the video VAE.
    """
    # ========== 步骤1: 视频格式预处理 ==========
    # 处理 PIL Image 列表输入（来自 output_type="pil"）
    if isinstance(video, list) and isinstance(video[0], PIL.Image.Image):
        # 假设每个图像都是 RGB 模式
        video_frames = [np.array(frame) for frame in video]
        video = np.stack(video_frames, axis=0)
        video = torch.from_numpy(video)
    
    # 处理 numpy ndarray 输入（来自 output_type="np"）
    elif isinstance(video, np.ndarray):
        # 检查是否为归一化的 [0, 1] 范围
        is_denormalized = np.logical_and(np.zeros_like(video) <= video, video <= np.ones_like(video))
        if np.all(is_denormalized):
            # 转换为像素值 [0, 255] 的 uint8 格式
            video = (video * 255).round().astype("uint8")
        else:
            logger.warning(
                "Supplied `numpy.ndarray` does not have values in [0, 1]. The values will be assumed to be pixel "
                "values in [0, ..., 255] and will be used as is."
            )
        video = torch.from_numpy(video)

    # ========== 步骤2: 视频分块处理 ==========
    # 将视频沿帧维度分割为多个块（用于 VAE tiling 模式）
    if isinstance(video, torch.Tensor):
        video = torch.tensor_split(video, video_chunks_number, dim=0)
        video = iter(video)  # 转换为迭代器

    # 获取第一块以确定视频尺寸
    first_chunk = next(video)
    _, height, width, _ = first_chunk.shape

    # ========== 步骤3: 创建输出容器和视频流 ==========
    container = av.open(output_path, mode="w")  # 创建输出容器
    stream = container.add_stream("libx264", rate=int(fps))  # 添加 H.264 视频流
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"  # 使用 YUV420p 像素格式以确保兼容性

    # ========== 步骤4: 准备音频流 ==========
    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided")
        # 创建 AAC 音频流
        audio_stream = _prepare_audio_stream(container, audio_sample_rate)

    # ========== 步骤5: 编码视频帧 ==========
    # 遍历所有视频块进行编码
    for video_chunk in tqdm(chain([first_chunk], video), total=video_chunks_number, desc="Encoding video chunks"):
        # 将张量移到 CPU 并转为 numpy 数组
        video_chunk_cpu = video_chunk.to("cpu").numpy()
        # 逐帧编码
        for frame_array in video_chunk_cpu:
            # 从 numpy 数组创建 PyAV 视频帧
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            # 编码帧并 mux 到容器
            for packet in stream.encode(frame):
                container.mux(packet)

    # ========== 步骤6: 刷新编码器 ==========
    # 刷新编码器以处理剩余的编码缓冲
    for packet in stream.encode():
        container.mux(packet)

    # ========== 步骤7: 写入音频 ==========
    if audio is not None:
        _write_audio(container, audio_stream, audio, audio_sample_rate)

    # ========== 步骤8: 关闭容器 ==========
    container.close()
```

## 关键组件





### 视频编码核心函数 (encode_video)

主函数，负责将视频帧和音频编码为MP4文件。支持多种输入格式（PIL Image列表、numpy数组、PyTorch张量），并处理视频分块编码。

### 音频流准备 (_prepare_audio_stream)

配置AAC音频流的采样率、声道布局和时间基，为后续音频编码做准备。

### 音频重采样 (_resample_audio)

使用PyAV的AudioResampler将输入音频帧重采样为目标编码器格式（采样率、布局、格式），确保与编码器兼容。

### 音频写入 (_write_audio)

将PyTorch音频张量转换为int16格式，创建AudioFrame并调用重采样器写入容器。同时处理单声道转立体声和形状调整。

### 张量分块与惰性加载

使用torch.tensor_split沿帧维度将视频分为多个chunk，配合迭代器实现惰性加载，避免一次性加载整个视频到内存。适用于视频VAE的tiling配置场景。

### 反量化支持

代码自动检测输入numpy数组的数值范围：若在[0,1]区间则乘以255转换为uint8；若已在[0,255]范围则直接使用。处理pipeline输出类型（np或pil）的差异。

### 量化策略 (音频)

将float32音频样本（范围[-1.0, 1.0]）乘以32767转换为int16紧凑格式，用于AAC编码器输入。



## 问题及建议



### 已知问题

-   **类型检查存在风险**：`isinstance(video, list) and isinstance(video[0], PIL.Image.Image)` 在video为空列表时会抛出IndexError
-   **资源未在finally块中释放**：`container.close()` 不在finally块中，如果编码过程中抛出异常，容器可能未正确关闭，导致文件损坏或资源泄漏
-   **音频采样率验证逻辑不完整**：仅在audio不为None时检查audio_sample_rate，但如果audio为None而audio_sample_rate提供了非None值，代码不会给出任何提示
-   **重复计算numpy数组**：每次处理np.ndarray时都计算`np.zeros_like(video)`和`np.ones_like(video)`，这两个数组可以预先定义为常量以避免重复分配内存
-   **音频写入时机可能导致内存问题**：音频在所有视频块编码完成后才写入，对于非常大的视频文件，中间结果可能占用大量内存
-   **硬编码的魔法数字**：如`32767.0`(int16范围)、`24000`(采样率)等数值硬编码在代码中，降低了可维护性
-   **tensor_split后未验证维度**：将视频tensor分割后，直接解包`_`而未验证shape，可能在维度不匹配时产生难以追踪的错误
-   **PyAV依赖处理方式粗糙**：在模块导入时就raise ImportError，无法作为可选功能使用，限制了代码的灵活性
-   **音频重采样器未复用**：每帧都创建新的`AudioResampler`实例，而不是复用，降低了效率
-   **视频流编码循环中的tqdm迭代器问题**：使用`chain([first_chunk], video)`但total设置为了`video_chunks_number`，当video_chunks_number与实际迭代次数不匹配时进度条显示不准确

### 优化建议

-   将`container.close()`包装在try-finally或使用context manager (`with av.open(...) as container:`) 确保资源释放
-   在类型检查前增加空列表判断：`if isinstance(video, list) and video and isinstance(video[0], PIL.Image.Image)`
-   将`np.zeros_like(video)`和`np.ones_like(video)`替换为预定义的`np.zeros(video.shape, dtype=video.dtype)`和`np.ones(video.shape, dtype=video.dtype)`常量
-   添加更严格的对象验证函数，验证输入tensor的维度、范围等是否符合预期
-   将硬编码的数值提取为模块级常量或配置参数，提高可维护性
-   考虑将PyAV依赖改为可选导入，在运行时检查而非模块加载时检查，提供更好的用户体验
-   复用AudioResampler实例而非每次调用都创建新实例
-   在编码完成后立即写入音频流而非等待所有视频处理完成，或采用流式处理策略减少内存峰值
-   增加更详细的日志记录，记录编码进度、参数信息等，便于调试和监控

## 其它




### 设计目标与约束

该模块的核心设计目标是提供一种统一的、高性能的接口，用于将视频帧和音频数据编码为标准的视频文件（MP4格式），支持多种输入格式（PIL Image、numpy数组、PyTorch张量），并能够处理大规模视频的分块编码需求。设计约束包括：依赖PyAV库进行编解码；输入视频帧的像素值必须在[0, 255]范围内（整数）或[0, 1]范围内（浮点数）；音频必须是2通道立体声，采样率需明确指定；仅支持libx264视频编码器和AAC音频编码器。

### 错误处理与异常设计

代码中的错误处理主要包括：ImportError在PyAV不可用时抛出，提示用户安装；ValueError在音频样本不是2通道时抛出；以及当numpy数组值不在[0,1]范围时发出警告并假定为像素值。此外，audio_sample_rate为None但audio不为None时抛出ValueError。当前错误处理粒度较粗，建议增加更多具体的异常类型定义，如VideoEncodingError、AudioEncodingError、UnsupportedFormatError等，以便调用方进行更精确的错误捕获和处理。

### 数据流与状态机

数据流处理流程如下：1) 输入验证阶段检查video和audio的类型与有效性；2) 格式转换阶段将不同输入格式统一转换为torch.Tensor；3) 视频分块阶段按video_chunks_number沿帧维度切分；4) 编码阶段逐块遍历视频帧，转换为AVFrame并编码为H.264流；5) 音频编码阶段将音频张量转换为AVFrame并编码为AAC流；6) 输出阶段将编码后的数据包复用（mux）到容器并写入文件。无复杂状态机设计，流程为单向线性处理。

### 外部依赖与接口契约

核心依赖为PyAV库（av），用于视频/音频的封装、编码和解码。此外依赖torch（张量处理）、numpy（数组操作）、PIL（图像处理）、tqdm（进度条显示）。接口契约规定：video输入可以是List[PIL.Image.Image]、np.ndarray、torch.Tensor或Iterator[torch.Tensor]，形状为[frames, height, width, channels]；fps必须为正整数；audio必须为torch.Tensor，形状为[channels, samples]；output_path必须是有效的文件路径且目录可写；video_chunks_number必须为正整数。

### 性能考虑

当前实现的主要性能瓶颈包括：视频帧逐帧编码效率较低，未利用批量编码；每次编码都需要GIL锁和Python解释器开销；音频重采样每次调用都会创建新的AudioResampler实例。建议优化方向：1) 使用av.VideoFrame的批量编码接口；2) 将AudioResampler实例化移到循环外部；3) 对于大规模视频，考虑使用多进程并行编码不同chunk；4) 预先分配内存缓冲区避免频繁分配。

### 资源管理

当前资源管理存在潜在问题：container.close()在正常流程中调用，但如果编码过程中发生异常，容器可能未正确关闭。建议使用上下文管理器（with语句）包装container，确保资源在任何情况下都能正确释放。此外，编码过程中的临时张量和数组未显式释放，建议在处理完大型视频chunk后显式调用del并清理GPU内存（如果使用GPU）。

### 兼容性考虑

代码兼容Python 3.8+（基于类型注解语法）。PyAV版本需支持指定的API（AudioResampler、container.mux等）。视频编码器libx264需要x264库可用，音频编码器aac需要fdk-aac或内置AAC编码器支持。跨平台性良好，依赖均为跨平台库。音频重采样默认目标格式为"fltp"，某些不支持浮点编码的硬件可能需要调整。

### 测试考虑

建议添加以下测试用例：1) 各种输入格式（PIL、numpy、tensor、iterator）的编码正确性验证；2) 音频视频同步性验证；3) 分块编码的完整性验证；4) 异常输入（错误形状、错误类型）的错误处理验证；5) 大规模视频编码的内存使用和性能基准测试；6) 不同fps、分辨率、音频采样率的组合测试。当前缺少单元测试和集成测试。

    