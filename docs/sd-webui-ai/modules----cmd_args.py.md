# `stable-diffusion-webui\modules\cmd_args.py`

```
# 导入必要的模块
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 数据
import os  # 提供与操作系统交互的功能
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir, sd_default_config, sd_model_file  # noqa: F401

# 创建 ArgumentParser 对象，用于解析命令行参数
parser = argparse.ArgumentParser()

# 添加命令行参数选项
parser.add_argument("-f", action='store_true', help=argparse.SUPPRESS)  # 允许以 root 用户身份运行；在 webui 外部实现
parser.add_argument("--update-all-extensions", action='store_true', help="launch.py argument: download updates for all extensions when starting the program")
parser.add_argument("--skip-python-version-check", action='store_true', help="launch.py argument: do not check python version")
parser.add_argument("--skip-torch-cuda-test", action='store_true', help="launch.py argument: do not check if CUDA is able to work properly")
parser.add_argument("--reinstall-xformers", action='store_true', help="launch.py argument: install the appropriate version of xformers even if you have some version already installed")
parser.add_argument("--reinstall-torch", action='store_true', help="launch.py argument: install the appropriate version of torch even if you have some version already installed")
parser.add_argument("--update-check", action='store_true', help="launch.py argument: check for updates at startup")
parser.add_argument("--test-server", action='store_true', help="launch.py argument: configure server for testing")
parser.add_argument("--log-startup", action='store_true', help="launch.py argument: print a detailed log of what's happening at startup")
parser.add_argument("--skip-prepare-environment", action='store_true', help="launch.py argument: skip all environment preparation")
parser.add_argument("--skip-install", action='store_true', help="launch.py argument: skip installation of packages")
parser.add_argument("--dump-sysinfo", action='store_true', help="launch.py argument: dump limited sysinfo file (without information about extensions, options) to disk and quit")
# 添加命令行参数 --loglevel，指定日志级别，默认为 None
parser.add_argument("--loglevel", type=str, help="log level; one of: CRITICAL, ERROR, WARNING, INFO, DEBUG", default=None)
# 添加命令行参数 --do-not-download-clip，如果设置则不下载 CLIP 模型
parser.add_argument("--do-not-download-clip", action='store_true', help="do not download CLIP model even if it's not included in the checkpoint")
# 添加命令行参数 --data-dir，指定用户数据存储的基本路径，默认为当前文件的上一级目录
parser.add_argument("--data-dir", type=str, default=os.path.dirname(os.path.dirname(os.path.realpath(__file__))), help="base path where all user data is stored")
# 添加命令行参数 --config，指定构建模型的配置文件路径，默认为 sd_default_config
parser.add_argument("--config", type=str, default=sd_default_config, help="path to config which constructs model",)
# 添加命令行参数 --ckpt，指定稳定扩散模型的检查点路径，默认为 sd_model_file
parser.add_argument("--ckpt", type=str, default=sd_model_file, help="path to checkpoint of stable diffusion model; if specified, this checkpoint will be added to the list of checkpoints and loaded",)
# 添加命令行参数 --ckpt-dir，指定稳定扩散检查点的目录路径，默认为 None
parser.add_argument("--ckpt-dir", type=str, default=None, help="Path to directory with stable diffusion checkpoints")
# 添加命令行参数 --vae-dir，指定 VAE 文件的目录路径，默认为 None
parser.add_argument("--vae-dir", type=str, default=None, help="Path to directory with VAE files")
# 添加命令行参数 --gfpgan-dir，指定 GFPGAN 的目录路径，默认为 './src/gfpgan' 或 './GFPGAN'（根据路径是否存在决定）
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN'))
# 添加命令行参数 --gfpgan-model，指定 GFPGAN 模型文件名，默认为 None
parser.add_argument("--gfpgan-model", type=str, help="GFPGAN model file name", default=None)
# 添加命令行参数 --no-half，如果设置则不将模型切换为 16 位浮点数
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats")
# 添加命令行参数 --no-half-vae，如果设置则不将 VAE 模型切换为 16 位浮点数
parser.add_argument("--no-half-vae", action='store_true', help="do not switch the VAE model to 16-bit floats")
# 添加命令行参数 --no-progressbar-hiding，如果设置则不在 gradio UI 中隐藏进度条（因为在浏览器中启用硬件加速会减慢 ML 运行速度）
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware acceleration in browser)")
# 添加命令行参数 --max-batch-count，指定 UI 的最大批次计数值，默认为 16
parser.add_argument("--max-batch-count", type=int, default=16, help="maximum batch count value for the UI")
# 添加命令行参数 --embeddings-dir，指定文本反演的嵌入目录路径，默认为 data_path 下的 'embeddings' 目录
parser.add_argument("--embeddings-dir", type=str, default=os.path.join(data_path, 'embeddings'), help="embeddings directory for textual inversion (default: embeddings)")
# 添加一个参数，指定文本反转模板的目录，默认为脚本路径下的'textual_inversion_templates'目录
parser.add_argument("--textual-inversion-templates-dir", type=str, default=os.path.join(script_path, 'textual_inversion_templates'), help="directory with textual inversion templates")
# 添加一个参数，指定超网络的目录，默认为模型路径下的'hypernetworks'目录
parser.add_argument("--hypernetwork-dir", type=str, default=os.path.join(models_path, 'hypernetworks'), help="hypernetwork directory")
# 添加一个参数，指定本地化目录，默认为脚本路径下的'localizations'目录
parser.add_argument("--localizations-dir", type=str, default=os.path.join(script_path, 'localizations'), help="localizations directory")
# 添加一个参数，允许从 webui 执行自定义脚本
parser.add_argument("--allow-code", action='store_true', help="allow custom script execution from webui")
# 添加一个参数，启用稳定扩散模型优化，牺牲一点速度以降低 VRM 使用
parser.add_argument("--medvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a little speed for low VRM usage")
# 添加一个参数，仅为 SDXL 模型启用 --medvram 优化
parser.add_argument("--medvram-sdxl", action='store_true', help="enable --medvram optimization just for SDXL models")
# 添加一个参数，启用稳定扩散模型优化，牺牲大量速度以降低 VRM 使用
parser.add_argument("--lowvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a lot of speed for very low VRM usage")
# 添加一个参数，将稳定扩散检查点权重加载到 VRAM 而不是 RAM
parser.add_argument("--lowram", action='store_true', help="load stable diffusion checkpoint weights to VRAM instead of RAM")
# 添加一个参数，不执行任何操作
parser.add_argument("--always-batch-cond-uncond", action='store_true', help="does not do anything")
# 添加一个参数，不执行任何操作
parser.add_argument("--unload-gfpgan", action='store_true', help="does not do anything.")
# 添加一个参数，评估在指定精度下，默认为'autocast'，可选值为'full'和'autocast'
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
# 添加一个参数，启用上转换采样，与 --no-half 一起使用时无效，通常产生与 --no-half 类似的结果，性能更好，内存占用更少
parser.add_argument("--upcast-sampling", action='store_true', help="upcast sampling. No effect with --no-half. Usually produces similar results to --no-half with better performance while using less memory.")
# 添加一个参数，使用 share=True 为 gradio，并通过他们的网站使 UI 可访问
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
# 添加一个参数，ngrok 的 authtoken，与 gradio --share 的替代方案，默认为 None
parser.add_argument("--ngrok", type=str, help="ngrok authtoken, alternative to gradio --share", default=None)
# 添加一个参数，不执行任何操作
parser.add_argument("--ngrok-region", type=str, help="does not do anything.", default="")
# 添加一个参数，用于传递给 ngrok 的选项，以 JSON 格式解析，例如：'{"authtoken_from_env":true, "basic_auth":"user:password", "oauth_provider":"google", "oauth_allow_emails":"user@asdf.com"}'
parser.add_argument("--ngrok-options", type=json.loads, help='The options to pass to ngrok in JSON format, e.g.: \'{"authtoken_from_env":true, "basic_auth":"user:password", "oauth_provider":"google", "oauth_allow_emails":"user@asdf.com"}\'', default=dict())

# 添加一个参数，启用不安全的扩展访问，无论其他选项如何
parser.add_argument("--enable-insecure-extension-access", action='store_true', help="enable extensions tab regardless of other options")

# 添加一个参数，指定 Codeformer 模型文件的目录路径
parser.add_argument("--codeformer-models-path", type=str, help="Path to directory with codeformer model file(s).", default=os.path.join(models_path, 'Codeformer'))

# 添加一个参数，指定 GFPGAN 模型文件的目录路径
parser.add_argument("--gfpgan-models-path", type=str, help="Path to directory with GFPGAN model file(s).", default=os.path.join(models_path, 'GFPGAN'))

# 添加一个参数，指定 ESRGAN 模型文件的目录路径
parser.add_argument("--esrgan-models-path", type=str, help="Path to directory with ESRGAN model file(s).", default=os.path.join(models_path, 'ESRGAN'))

# 添加一个参数，指定 BSRGAN 模型文件的目录路径
parser.add_argument("--bsrgan-models-path", type=str, help="Path to directory with BSRGAN model file(s).", default=os.path.join(models_path, 'BSRGAN'))

# 添加一个参数，指定 RealESRGAN 模型文件的目录路径
parser.add_argument("--realesrgan-models-path", type=str, help="Path to directory with RealESRGAN model file(s).", default=os.path.join(models_path, 'RealESRGAN'))

# 添加一个参数，指定 CLIP 模型文件的目录路径
parser.add_argument("--clip-models-path", type=str, help="Path to directory with CLIP model file(s).", default=None)

# 添加一个参数，启用 xformers 用于交叉注意力层
parser.add_argument("--xformers", action='store_true', help="enable xformers for cross attention layers")

# 添加一个参数，强制启用 xformers 用于交叉注意力层，不管检查代码认为是否可以运行；如果此功能无法正常工作，请不要提交错误报告
parser.add_argument("--force-enable-xformers", action='store_true', help="enable xformers for cross attention layers regardless of whether the checking code thinks you can run it; do not make bug reports if this fails to work")

# 添加一个参数，启用具有 Flash Attention 的 xformers 以提高可重现性（仅支持 SD2.x 或其变体）
parser.add_argument("--xformers-flash-attention", action='store_true', help="enable xformers with Flash Attention to improve reproducibility (supported for SD2.x or variant only)")

# 添加一个参数，不执行任何操作
parser.add_argument("--deepdanbooru", action='store_true', help="does not do anything")
# 添加一个命令行参数，用于启用 split attention 优化，自动选择优化方式
parser.add_argument("--opt-split-attention", action='store_true', help="prefer Doggettx's cross-attention layer optimization for automatic choice of optimization")
# 添加一个命令行参数，用于启用 sub-quadratic attention 优化，自动选择优化方式
parser.add_argument("--opt-sub-quad-attention", action='store_true', help="prefer memory efficient sub-quadratic cross-attention layer optimization for automatic choice of optimization")
# 添加一个命令行参数，设置 sub-quadratic attention 优化中的查询块大小，默认为 1024
parser.add_argument("--sub-quad-q-chunk-size", type=int, help="query chunk size for the sub-quadratic cross-attention layer optimization to use", default=1024)
# 添加一个命令行参数，设置 sub-quadratic attention 优化中的键值块大小，默认为 None
parser.add_argument("--sub-quad-kv-chunk-size", type=int, help="kv chunk size for the sub-quadratic cross-attention layer optimization to use", default=None)
# 添加一个命令行参数，设置 sub-quadratic attention 优化中的内存阈值百分比，默认为 None
parser.add_argument("--sub-quad-chunk-threshold", type=int, help="the percentage of VRAM threshold for the sub-quadratic cross-attention layer optimization to use chunking", default=None)
# 添加一个命令行参数，用于启用 InvokeAI 的 split attention 优化，自动选择优化方式
parser.add_argument("--opt-split-attention-invokeai", action='store_true', help="prefer InvokeAI's cross-attention layer optimization for automatic choice of optimization")
# 添加一个命令行参数，用于启用旧版本的 split attention 优化，自动选择优化方式
parser.add_argument("--opt-split-attention-v1", action='store_true', help="prefer older version of split attention optimization for automatic choice of optimization")
# 添加一个命令行参数，用于启用 sdp attention 优化，自动选择优化方式；需要 PyTorch 2.*
parser.add_argument("--opt-sdp-attention", action='store_true', help="prefer scaled dot product cross-attention layer optimization for automatic choice of optimization; requires PyTorch 2.*")
# 添加一个命令行参数，用于启用无内存高效 attention 的 sdp attention 优化，自动选择优化方式，使图像生成确定性；需要 PyTorch 2.*
parser.add_argument("--opt-sdp-no-mem-attention", action='store_true', help="prefer scaled dot product cross-attention layer optimization without memory efficient attention for automatic choice of optimization, makes image generation deterministic; requires PyTorch 2.*")
# 添加一个命令行参数，用于禁用 split attention 优化，自动选择优化方式
parser.add_argument("--disable-opt-split-attention", action='store_true', help="prefer no cross-attention layer optimization for automatic choice of optimization")
# 添加一个命令行参数，用于禁用检查生成的图像/潜在空间是否有 NaN；在 CI 中运行时很有用
parser.add_argument("--disable-nan-check", action='store_true', help="do not check if produced images/latent spaces have nans; useful for running without a checkpoint in CI")
# 添加一个名为"use-cpu"的命令行参数，用于指定要在CPU上使用的torch设备模块
parser.add_argument("--use-cpu", nargs='+', help="use CPU as torch device for specified modules", default=[], type=str.lower)

# 添加一个名为"use-ipex"的命令行参数，用于指定是否使用Intel XPU作为torch设备
parser.add_argument("--use-ipex", action="store_true", help="use Intel XPU as torch device")

# 添加一个名为"disable-model-loading-ram-optimization"的命令行参数，用于指定是否禁用加载模型时减少RAM使用的优化
parser.add_argument("--disable-model-loading-ram-optimization", action='store_true', help="disable an optimization that reduces RAM use when loading a model")

# 添加一个名为"listen"的命令行参数，用于指定是否以0.0.0.0作为服务器名称启动gradio，允许响应网络请求
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")

# 添加一个名为"port"的命令行参数，用于指定以给定服务器端口启动gradio，如果可用，则默认为7860，需要root/admin权限才能使用小于1024的端口
parser.add_argument("--port", type=int, help="launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available", default=None)

# 添加一个名为"show-negative-prompt"的命令行参数，用于指定是否显示负面提示
parser.add_argument("--show-negative-prompt", action='store_true', help="does not do anything", default=False)

# 添加一个名为"ui-config-file"的命令行参数，用于指定UI配置的文件名，默认为"data_path"目录下的"ui-config.json"
parser.add_argument("--ui-config-file", type=str, help="filename to use for ui configuration", default=os.path.join(data_path, 'ui-config.json'))

# 添加一个名为"hide-ui-dir-config"的命令行参数，用于指定是否隐藏webui中的目录配置
parser.add_argument("--hide-ui-dir-config", action='store_true', help="hide directory configuration from webui", default=False)

# 添加一个名为"freeze-settings"的命令行参数，用于指定是否禁用编辑设置
parser.add_argument("--freeze-settings", action='store_true', help="disable editing settings", default=False)

# 添加一个名为"ui-settings-file"的命令行参数，用于指定UI设置的文件名，默认为"data_path"目录下的"config.json"
parser.add_argument("--ui-settings-file", type=str, help="filename to use for ui settings", default=os.path.join(data_path, 'config.json'))

# 添加一个名为"gradio-debug"的命令行参数，用于指定是否以--debug选项启动gradio
parser.add_argument("--gradio-debug",  action='store_true', help="launch gradio with --debug option")

# 添加一个名为"gradio-auth"的命令行参数，用于设置gradio的身份验证，格式为"username:password"，或者用逗号分隔多个，如"u1:p1,u2:p2,u3:p3"
parser.add_argument("--gradio-auth", type=str, help='set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"', default=None)

# 添加一个名为"gradio-auth-path"的命令行参数，用于设置gradio的身份验证文件路径，格式与--gradio-auth相同
parser.add_argument("--gradio-auth-path", type=str, help='set gradio authentication file path ex. "/path/to/auth/file" same auth format as --gradio-auth', default=None)

# 添加一个名为"gradio-img2img-tool"的命令行参数，用于指定不执行任何操作
parser.add_argument("--gradio-img2img-tool", type=str, help='does not do anything')

# 添加一个名为"gradio-inpaint-tool"的命令行参数，用于指定不执行任何操作
parser.add_argument("--gradio-inpaint-tool", type=str, help="does not do anything")
# 添加一个参数到命令行解析器，用于指定 Gradio 允许的路径，可以从这些路径中提供文件服务
parser.add_argument("--gradio-allowed-path", action='append', help="add path to gradio's allowed_paths, make it possible to serve files from it", default=[data_path])
# 添加一个参数到命令行解析器，用于指定是否将内存类型更改为 channels last 以稳定扩散
parser.add_argument("--opt-channelslast", action='store_true', help="change memory type for stable diffusion to channels last")
# 添加一个参数到命令行解析器，用于指定要使用的样式文件的文件名
parser.add_argument("--styles-file", type=str, help="filename to use for styles", default=os.path.join(data_path, 'styles.csv'))
# 添加一个参数到命令行解析器，用于指定是否在启动时在系统默认浏览器中打开 webui URL
parser.add_argument("--autolaunch", action='store_true', help="open the webui URL in the system's default browser upon launch", default=False)
# 添加一个参数到命令行解析器，用于指定以浅色或深色主题启动 UI
parser.add_argument("--theme", type=str, help="launches the UI with light or dark theme", default=None)
# 添加一个参数到命令行解析器，用于指定是否在 UI 中使用文本框作为种子输入框
parser.add_argument("--use-textbox-seed", action='store_true', help="use textbox for seeds in UI (no up/down, but possible to input long seeds)", default=False)
# 添加一个参数到命令行解析器，用于指定是否禁用在控制台输出进度条
parser.add_argument("--disable-console-progressbars", action='store_true', help="do not output progressbars to console", default=False)
# 添加一个参数到命令行解析器，用于指定是否启用在控制台输出提示
parser.add_argument("--enable-console-prompts", action='store_true', help="does not do anything", default=False)  # Legacy compatibility, use as default value shared.opts.enable_console_prompts
# 添加一个参数到命令行解析器，用于指定要用作 VAE 的检查点；设置此参数将禁用与 VAE 相关的所有设置
parser.add_argument('--vae-path', type=str, help='Checkpoint to use as VAE; setting this argument disables all settings related to VAE', default=None)
# 添加一个参数到命令行解析器，用于指定是否禁用检查 PyTorch 模型是否包含恶意代码
parser.add_argument("--disable-safe-unpickle", action='store_true', help="disable checking pytorch models for malicious code", default=False)
# 添加一个参数到命令行解析器，用于指定是否同时启动 API 与 webui（使用 --nowebui 代替只启动 API）
parser.add_argument("--api", action='store_true', help="use api=True to launch the API together with the webui (use --nowebui instead for only the API)")
# 添加一个参数到命令行解析器，用于设置 API 的身份验证，格式为 "username:password"；或使用逗号分隔多个，如 "u1:p1,u2:p2,u3:p3"
parser.add_argument("--api-auth", type=str, help='Set authentication for API like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"', default=None)
# 添加一个参数到命令行解析器，用于指定是否启用记录所有 API 请求
parser.add_argument("--api-log", action='store_true', help="use api-log=True to enable logging of all API requests")
# 添加一个参数，当设置时使用 API 模式而不是 WebUI 模式
parser.add_argument("--nowebui", action='store_true', help="use api=True to launch the API instead of the webui")
# 添加一个参数，用于在调试模式下快速启动 UI 而不加载模型
parser.add_argument("--ui-debug-mode", action='store_true', help="Don't load model to quickly launch UI")
# 添加一个参数，用于选择要使用的默认 CUDA 设备
parser.add_argument("--device-id", type=str, help="Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc might be needed before)", default=None)
# 添加一个参数，用于指定是否具有管理员权限
parser.add_argument("--administrator", action='store_true', help="Administrator rights", default=False)
# 添加一个参数，用于设置允许的 CORS 来源列表，以逗号分隔
parser.add_argument("--cors-allow-origins", type=str, help="Allowed CORS origin(s) in the form of a comma-separated list (no spaces)", default=None)
# 添加一个参数，用于设置允许的 CORS 来源列表的正则表达式形式
parser.add_argument("--cors-allow-origins-regex", type=str, help="Allowed CORS origin(s) in the form of a single regular expression", default=None)
# 添加一个参数，用于部分启用 TLS，需要 --tls-certfile 参数才能完全生效
parser.add_argument("--tls-keyfile", type=str, help="Partially enables TLS, requires --tls-certfile to fully function", default=None)
# 添加一个参数，用于部分启用 TLS，需要 --tls-keyfile 参数才能完全生效
parser.add_argument("--tls-certfile", type=str, help="Partially enables TLS, requires --tls-keyfile to fully function", default=None)
# 添加一个参数，用于禁用 TLS 验证，当设置时启用使用自签名证书
parser.add_argument("--disable-tls-verify", action="store_false", help="When passed, enables the use of self-signed certificates.", default=None)
# 添加一个参数，用于设置服务器的主机名
parser.add_argument("--server-name", type=str, help="Sets hostname of server", default=None)
# 添加一个参数，用于启用 Gradio 队列，但实际上不执行任何操作
parser.add_argument("--gradio-queue", action='store_true', help="does not do anything", default=True)
# 添加一个参数，用于禁用 Gradio 队列，导致网页使用 HTTP 请求而不是 WebSockets；在早期版本中是默认设置
parser.add_argument("--no-gradio-queue", action='store_true', help="Disables gradio queue; causes the webpage to use http requests instead of websockets; was the default in earlier versions")
# 添加一个参数，用于跳过 torch 和 xformers 版本检查
parser.add_argument("--skip-version-check", action='store_true', help="Do not check versions of torch and xformers")
# 添加一个参数，用于禁用检查点的 sha256 哈希以提高加载性能
parser.add_argument("--no-hashing", action='store_true', help="disable sha256 hashing of checkpoints to help loading performance", default=False)
# 添加一个参数，如果设置了该参数，则不下载SD1.5模型，即使在--ckpt-dir中找不到模型
parser.add_argument("--no-download-sd-model", action='store_true', help="don't download SD1.5 model even if no model is found in --ckpt-dir", default=False)

# 添加一个参数，用于自定义gradio的子路径，与反向代理一起使用
parser.add_argument('--subpath', type=str, help='customize the subpath for gradio, use with reverse proxy')

# 添加一个参数，不执行任何操作
parser.add_argument('--add-stop-route', action='store_true', help='does not do anything')

# 添加一个参数，启用通过API停止/重新启动/终止服务器
parser.add_argument('--api-server-stop', action='store_true', help='enable server stop/restart/kill via api')

# 添加一个参数，设置uvicorn的timeout_keep_alive
parser.add_argument('--timeout-keep-alive', type=int, default=30, help='set timeout_keep_alive for uvicorn')

# 添加一个参数，如果设置了该参数，则阻止所有扩展运行，不考虑其他设置
parser.add_argument("--disable-all-extensions", action='store_true', help="prevent all extensions from running regardless of any other settings", default=False)

# 添加一个参数，如果设置了该参数，则阻止除内置扩展之外的所有扩展运行，不考虑其他设置
parser.add_argument("--disable-extra-extensions", action='store_true', help="prevent all extensions except built-in from running regardless of any other settings", default=False)

# 添加一个参数，如果设置了该参数，则在Web启动时不加载模型，仅在--nowebui时生效
parser.add_argument("--skip-load-model-at-start", action='store_true', help="if load a model at web start, only take effect when --nowebui", )
```