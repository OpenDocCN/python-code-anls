# `stable-diffusion-webui\modules\api\api.py`

```py
# 导入必要的库
import base64
import io
import os
import time
import datetime
import uvicorn
import ipaddress
import requests
import gradio as gr
from threading import Lock
from io import BytesIO
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from secrets import compare_digest

# 导入自定义模块
import modules.shared as shared
from modules import sd_samplers, deepbooru, sd_hijack, images, scripts, ui, postprocessing, errors, restart, shared_items, script_callbacks, generation_parameters_copypaste, sd_models
from modules.api import models
from modules.shared import opts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.textual_inversion.textual_inversion import create_embedding, train_embedding
from modules.hypernetworks.hypernetwork import create_hypernetwork, train_hypernetwork
from PIL import PngImagePlugin, Image
from modules.sd_models_config import find_checkpoint_config_near_filename
from modules.realesrgan_model import get_realesrgan_models
from modules import devices
from typing import Any
import piexif
import piexif.helper
from contextlib import closing

# 定义函数，根据脚本名称和脚本列表返回对应脚本的索引
def script_name_to_index(name, scripts):
    try:
        return [script.title().lower() for script in scripts].index(name.lower())
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Script '{name}' not found") from e

# 验证采样器名称是否存在
def validate_sampler_name(name):
    config = sd_samplers.all_samplers_map.get(name, None)
    if config is None:
        raise HTTPException(status_code=404, detail="Sampler not found")

    return name

# 设置上采样器
def setUpscalers(req: dict):
    reqDict = vars(req)
    reqDict['extras_upscaler_1'] = reqDict.pop('upscaler_1', None)
    reqDict['extras_upscaler_2'] = reqDict.pop('upscaler_2', None)
    # 返回 reqDict 字典作为函数的输出结果
    return reqDict
# 验证 URL 是否指向全局资源，如果是返回 True
def verify_url(url):
    # 导入必要的模块
    import socket
    from urllib.parse import urlparse
    try:
        # 解析 URL
        parsed_url = urlparse(url)
        # 获取域名
        domain_name = parsed_url.netloc
        # 获取主机的 IP 地址
        host = socket.gethostbyname_ex(domain_name)
        # 遍历主机的 IP 地址列表
        for ip in host[2]:
            # 创建 IP 地址对象
            ip_addr = ipaddress.ip_address(ip)
            # 如果 IP 地址不是全局地址，则返回 False
            if not ip_addr.is_global:
                return False
    except Exception:
        return False

    return True


# 将 base64 编码的数据解码为图像
def decode_base64_to_image(encoding):
    # 如果编码以 "http://" 或 "https://" 开头
    if encoding.startswith("http://") or encoding.startswith("https://"):
        # 如果不允许使用 requests 模块，则抛出异常
        if not opts.api_enable_requests:
            raise HTTPException(status_code=500, detail="Requests not allowed")
        
        # 如果禁止请求本地资源且 URL 不是全局资源，则抛出异常
        if opts.api_forbid_local_requests and not verify_url(encoding):
            raise HTTPException(status_code=500, detail="Request to local resource not allowed")
        
        # 设置请求头
        headers = {'user-agent': opts.api_useragent} if opts.api_useragent else {}
        # 发起 GET 请求获取响应
        response = requests.get(encoding, timeout=30, headers=headers)
        try:
            # 尝试打开图像
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            # 如果出现异常，则抛出异常
            raise HTTPException(status_code=500, detail="Invalid image url") from e

    # 如果编码以 "data:image/" 开头
    if encoding.startswith("data:image/"):
        # 提取编码数据部分
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        # 尝试解码 base64 编码的数据并打开图像
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        # 如果出现异常，则抛出异常
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e


# 将 PIL 图像编码为 base64
def encode_pil_to_base64(image):
    # 使用 BytesIO 创建一个内存中的二进制流对象，并赋值给 output_bytes
    with io.BytesIO() as output_bytes:
        # 如果 image 是字符串类型，则直接返回该字符串
        if isinstance(image, str):
            return image
        # 如果 opts.samples_format 是 'png'，则执行以下代码块
        if opts.samples_format.lower() == 'png':
            # 初始化 use_metadata 为 False，创建一个空的 PNG 元数据对象
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            # 遍历 image 的元数据，将元数据添加到 PNG 元数据对象中
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            # 将 image 保存为 PNG 格式到 output_bytes 中，包括元数据信息
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=opts.jpeg_quality)

        # 如果 opts.samples_format 是 'jpg', 'jpeg' 或 'webp'，则执行以下代码块
        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            # 如果 image 的模式是 'RGBA'，则转换为 'RGB'
            if image.mode == "RGBA":
                image = image.convert("RGB")
            # 获取 image 的参数信息，将参数信息转换为 Exif 格式的字节流
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
            })
            # 根据 opts.samples_format 的值保存 image 到 output_bytes 中，包括 Exif 信息
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif = exif_bytes, quality=opts.jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif = exif_bytes, quality=opts.jpeg_quality)

        # 如果 opts.samples_format 不是 'png', 'jpg', 'jpeg' 或 'webp'，则抛出异常
        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        # 获取 output_bytes 中的所有数据，并赋值给 bytes_data
        bytes_data = output_bytes.getvalue()

    # 返回 base64 编码后的 bytes_data
    return base64.b64encode(bytes_data)
# 定义一个 API 中间件函数，接受一个 FastAPI 应用对象作为参数
def api_middleware(app: FastAPI):
    # 初始化一个变量，用于标记是否 rich 库可用
    rich_available = False
    try:
        # 尝试获取环境变量 'WEBUI_RICH_EXCEPTIONS'，如果存在则表示 rich 库可用
        if os.environ.get('WEBUI_RICH_EXCEPTIONS', None) is not None:
            # 导入 anyio 和 starlette 库，仅仅是为了将其放在 silent list 中
            import anyio
            import starlette
            # 导入 rich 库中的 Console 类
            from rich.console import Console
            # 创建一个 Console 对象
            console = Console()
            # 设置 rich 库可用标记为 True
            rich_available = True
    except Exception:
        pass

    # 定义一个 HTTP 中间件函数，用于记录请求日志和处理时间
    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        # 记录请求处理开始时间
        ts = time.time()
        # 调用下一个中间件或路由处理函数，并获取返回的响应对象
        res: Response = await call_next(req)
        # 计算请求处理时间
        duration = str(round(time.time() - ts, 4))
        # 将处理时间添加到响应头中
        res.headers["X-Process-Time"] = duration
        # 获取请求的端点路径
        endpoint = req.scope.get('path', 'err')
        # 如果启用了 API 日志，并且端点路径以 '/sdapi' 开头，则打印请求日志
        if shared.cmd_opts.api_log and endpoint.startswith('/sdapi'):
            print('API {t} {code} {prot}/{ver} {method} {endpoint} {cli} {duration}'.format(
                t=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                code=res.status_code,
                ver=req.scope.get('http_version', '0.0'),
                cli=req.scope.get('client', ('0:0.0.0', 0))[0],
                prot=req.scope.get('scheme', 'err'),
                method=req.scope.get('method', 'err'),
                endpoint=endpoint,
                duration=duration,
            ))
        # 返回响应对象
        return res
    # 处理异常的函数，接受请求对象和异常对象作为参数
    def handle_exception(request: Request, e: Exception):
        # 构建异常信息字典
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get('detail', ''),
            "body": vars(e).get('body', ''),
            "errors": str(e),
        }
        # 如果异常不是 HTTPException 类型，则打印错误信息
        if not isinstance(e, HTTPException):  # do not print backtrace on known httpexceptions
            message = f"API error: {request.method}: {request.url} {err}"
            # 如果有 rich 库可用，则使用 console 打印异常信息
            if rich_available:
                print(message)
                console.print_exception(show_locals=True, max_frames=2, extra_lines=1, suppress=[anyio, starlette], word_wrap=False, width=min([console.width, 200]))
            # 否则使用 errors 模块报告异常信息
            else:
                errors.report(message, exc_info=True)
        # 返回 JSONResponse 对象，包含异常状态码和异常信息
        return JSONResponse(status_code=vars(e).get('status_code', 500), content=jsonable_encoder(err))

    # 注册一个中间件，用于处理异常
    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    # 注册一个异常处理函数，用于处理所有异常
    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    # 注册一个异常处理函数，用于处理 HTTPException 类型的异常
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)
# 定义一个名为 Api 的类
class Api:
    # 添加 API 路由的方法，接受路径、端点和其他参数
    def add_api_route(self, path: str, endpoint, **kwargs):
        # 如果启用了 API 认证
        if shared.cmd_opts.api_auth:
            # 添加 API 路由并指定依赖于认证函数
            return self.app.add_api_route(path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        # 否则添加 API 路由
        return self.app.add_api_route(path, endpoint, **kwargs)

    # 认证函数，接受 HTTPBasicCredentials 作为参数
    def auth(self, credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
        # 如果用户名在凭证列表中且密码匹配
        if credentials.username in self.credentials and compare_digest(credentials.password, self.credentials[credentials.username]):
            return True

        # 否则抛出 HTTP 异常，提示用户名或密码不正确
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})

    # 获取可选择脚本的方法，接受脚本名称和脚本运行器作为参数
    def get_selectable_script(self, script_name, script_runner):
        # 如果脚本名称为空，则返回 None
        if script_name is None or script_name == "":
            return None, None

        # 根据脚本名称获取脚本索引，并获取对应的脚本
        script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
        script = script_runner.selectable_scripts[script_idx]
        return script, script_idx

    # 获取脚本列表的方法
    def get_scripts_list(self):
        # 获取文本到图像脚本和图像到图像脚本的名称列表
        t2ilist = [script.name for script in scripts.scripts_txt2img.scripts if script.name is not None]
        i2ilist = [script.name for script in scripts.scripts_img2img.scripts if script.name is not None]

        # 返回脚本列表模型对象
        return models.ScriptsList(txt2img=t2ilist, img2img=i2ilist)

    # 获取脚本信息的方法
    def get_script_info(self):
        res = []

        # 遍历文本到图像脚本和图像到图像脚本列表，获取 API 信息
        for script_list in [scripts.scripts_txt2img.scripts, scripts.scripts_img2img.scripts]:
            res += [script.api_info for script in script_list if script.api_info is not None]

        return res

    # 获取特定脚本的方法，接受脚本名称和脚本运行器作为参数
    def get_script(self, script_name, script_runner):
        # 如果脚本名称为空，则返回 None
        if script_name is None or script_name == "":
            return None, None

        # 根据脚本名称获取脚本索引，并返回对应的脚本
        script_idx = script_name_to_index(script_name, script_runner.scripts)
        return script_runner.scripts[script_idx]
    # 初始化默认的脚本参数，根据脚本运行器中的脚本找到最大的索引，生成一个空数组来初始化脚本参数
    def init_default_script_args(self, script_runner):
        last_arg_index = 1
        for script in script_runner.scripts:
            if last_arg_index < script.args_to:
                last_arg_index = script.args_to
        # 在除了位置0之外的所有位置初始化脚本参数为 None
        script_args = [None]*last_arg_index
        script_args[0] = 0

        # 获取默认值
        with gr.Blocks(): # 如果没有这个会在调用 UI 函数时抛出错误
            for script in script_runner.scripts:
                if script.ui(script.is_img2img):
                    ui_default_values = []
                    for elem in script.ui(script.is_img2img):
                        ui_default_values.append(elem.value)
                    script_args[script.args_from:script.args_to] = ui_default_values
        # 返回脚本参数
        return script_args
    # 初始化脚本参数，根据请求、默认脚本参数、可选择脚本、可选择脚本索引和脚本运行器
    def init_script_args(self, request, default_script_args, selectable_scripts, selectable_idx, script_runner):
        # 复制默认脚本参数
        script_args = default_script_args.copy()
        # 如果存在可选择脚本
        if selectable_scripts:
            # 将请求中的脚本参数替换到指定位置
            script_args[selectable_scripts.args_from:selectable_scripts.args_to] = request.script_args
            # 设置可选择脚本的索引
            script_args[0] = selectable_idx + 1

        # 检查始终运行的脚本
        if request.alwayson_scripts:
            # 遍历始终运行的脚本
            for alwayson_script_name in request.alwayson_scripts.keys():
                # 获取始终运行的脚本
                alwayson_script = self.get_script(alwayson_script_name, script_runner)
                # 如果始终运行的脚本不存在，则抛出异常
                if alwayson_script is None:
                    raise HTTPException(status_code=422, detail=f"always on script {alwayson_script_name} not found")
                # 检查始终运行的脚本中是否包含可选择脚本
                if alwayson_script.alwayson is False:
                    raise HTTPException(status_code=422, detail="Cannot have a selectable script in the always on scripts params")
                # 如果始终运行的脚本没有参数，则无需添加到请求中
                if "args" in request.alwayson_scripts[alwayson_script_name]:
                    # 在脚本运行器参数长度和请求参数长度中取较小值
                    for idx in range(0, min((alwayson_script.args_to - alwayson_script.args_from), len(request.alwayson_scripts[alwayson_script_name]["args"]))):
                        script_args[alwayson_script.args_from + idx] = request.alwayson_scripts[alwayson_script_name]["args"][idx]
        # 返回更新后的脚本参数
        return script_args
    # 处理单张图片的额外信息API请求
    def extras_single_image_api(self, req: models.ExtrasSingleImageRequest):
        # 将请求转换为字典形式
        reqDict = setUpscalers(req)

        # 解码base64编码的图片数据
        reqDict['image'] = decode_base64_to_image(reqDict['image'])

        # 使用队列锁保护，运行额外信息处理函数
        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=0, image_folder="", input_dir="", output_dir="", save_output=False, **reqDict)

        # 返回处理结果
        return models.ExtrasSingleImageResponse(image=encode_pil_to_base64(result[0][0]), html_info=result[1])

    # 处理批量图片的额外信息API请求
    def extras_batch_images_api(self, req: models.ExtrasBatchImagesRequest):
        # 将请求转换为字典形式
        reqDict = setUpscalers(req)

        # 获取图片列表并解码base64编码的图片数据
        image_list = reqDict.pop('imageList', [])
        image_folder = [decode_base64_to_image(x.data) for x in image_list]

        # 使用队列锁保护，运行批量图片额外信息处理函数
        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=1, image_folder=image_folder, image="", input_dir="", output_dir="", save_output=False, **reqDict)

        # 返回处理结果
        return models.ExtrasBatchImagesResponse(images=list(map(encode_pil_to_base64, result[0])), html_info=result[1])

    # 处理PNG图片信息API请求
    def pnginfoapi(self, req: models.PNGInfoRequest):
        # 解码base64编码的图片数据
        image = decode_base64_to_image(req.image.strip())
        if image is None:
            return models.PNGInfoResponse(info="")

        # 从图片中读取通用信息和项目信息
        geninfo, items = images.read_info_from_image(image)
        if geninfo is None:
            geninfo = ""

        # 解析生成参数
        params = generation_parameters_copypaste.parse_generation_parameters(geninfo)
        script_callbacks.infotext_pasted_callback(geninfo, params)

        # 返回信息结果
        return models.PNGInfoResponse(info=geninfo, items=items, parameters=params)
    # 定义一个名为progressapi的方法，接受一个ProgressRequest类型的参数req
    def progressapi(self, req: models.ProgressRequest = Depends()):
        # 从ui.py的check_progress_call中复制代码

        # 如果当前没有任务在进行，则返回进度为0的响应
        if shared.state.job_count == 0:
            return models.ProgressResponse(progress=0, eta_relative=0, state=shared.state.dict(), textinfo=shared.state.textinfo)

        # 避免除以零
        progress = 0.01

        # 如果当前有任务在进行，则计算进度
        if shared.state.job_count > 0:
            progress += shared.state.job_no / shared.state.job_count
        if shared.state.sampling_steps > 0:
            progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

        # 计算预计完成时间
        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start/progress)
        eta_relative = eta-time_since_start

        # 将进度限制在0到1之间
        progress = min(progress, 1)

        # 设置当前图片
        shared.state.set_current_image()

        current_image = None
        # 如果当前图片存在且请求中未跳过当前图片，则将当前图片编码为base64格式
        if shared.state.current_image and not req.skip_current_image:
            current_image = encode_pil_to_base64(shared.state.current_image)

        # 返回进度响应
        return models.ProgressResponse(progress=progress, eta_relative=eta_relative, state=shared.state.dict(), current_image=current_image, textinfo=shared.state.textinfo)

    # 定义一个名为interrogateapi的方法，接受一个InterrogateRequest类型的参数interrogatereq
    def interrogateapi(self, interrogatereq: models.InterrogateRequest):
        # 获取请求中的base64格式图片数据
        image_b64 = interrogatereq.image
        # 如果图片数据为空，则抛出404错误
        if image_b64 is None:
            raise HTTPException(status_code=404, detail="Image not found")

        # 将base64格式图片数据解码为图像，并转换为RGB格式
        img = decode_base64_to_image(image_b64)
        img = img.convert('RGB')

        # 根据请求中的模型参数进行不同的处理
        with self.queue_lock:
            if interrogatereq.model == "clip":
                processed = shared.interrogator.interrogate(img)
            elif interrogatereq.model == "deepdanbooru":
                processed = deepbooru.model.tag(img)
            else:
                raise HTTPException(status_code=404, detail="Model not found")

        # 返回处理后的结果
        return models.InterrogateResponse(caption=processed)
    # 中断 API 操作
    def interruptapi(self):
        # 调用共享状态中的中断方法
        shared.state.interrupt()

        # 返回空字典
        return {}

    # 卸载 API
    def unloadapi(self):
        # 调用模型加载模块中的卸载模型权重方法
        sd_models.unload_model_weights()

        # 返回空字典
        return {}

    # 重新加载 API
    def reloadapi(self):
        # 将共享的模型发送到设备
        sd_models.send_model_to_device(shared.sd_model)

        # 返回空字典
        return {}

    # 跳过操作
    def skip(self):
        # 调用共享状态中的跳过方法
        shared.state.skip()

    # 获取配置信息
    def get_config(self):
        # 初始化选项字典
        options = {}
        # 遍历共享选项数据的键
        for key in shared.opts.data.keys():
            # 获取键对应的元数据
            metadata = shared.opts.data_labels.get(key)
            # 如果元数据不为空
            if(metadata is not None):
                # 更新选项字典，使用默认值填充缺失的键
                options.update({key: shared.opts.data.get(key, shared.opts.data_labels.get(key).default)})
            else:
                # 更新选项字典，使用 None 填充缺失的键
                options.update({key: shared.opts.data.get(key, None)})

        # 返回选项字典
        return options

    # 设置配置信息
    def set_config(self, req: dict[str, Any]):
        # 获取请求中的模型检查点名称
        checkpoint_name = req.get("sd_model_checkpoint", None)
        # 如果模型检查点名称不为空且不在检查点别名中
        if checkpoint_name is not None and checkpoint_name not in sd_models.checkpoint_aliases:
            # 抛出运行时错误
            raise RuntimeError(f"model {checkpoint_name!r} not found")

        # 遍历请求中的键值对
        for k, v in req.items():
            # 设置共享选项中的键值对，标记为 API 操作
            shared.opts.set(k, v, is_api=True)

        # 保存共享选项到配置文件
        shared.opts.save(shared.config_filename)
        return

    # 获取命令行标志
    def get_cmd_flags(self):
        # 返回共享命令行选项的变量
        return vars(shared.cmd_opts)

    # 获取采样器信息
    def get_samplers(self):
        # 返回所有采样器的名称、别名和选项列表
        return [{"name": sampler[0], "aliases":sampler[2], "options":sampler[3]} for sampler in sd_samplers.all_samplers]

    # 获取上采样器信息
    def get_upscalers(self):
        # 返回所有上采样器的名称、模型名称、模型路径、模型 URL 和缩放比例
        return [
            {
                "name": upscaler.name,
                "model_name": upscaler.scaler.model_name,
                "model_path": upscaler.data_path,
                "model_url": None,
                "scale": upscaler.scale,
            }
            for upscaler in shared.sd_upscalers
        ]

    # 获取潜在上采样模式
    def get_latent_upscale_modes(self):
        # 返回所有潜在上采样模式的名称
        return [
            {
                "name": upscale_mode,
            }
            for upscale_mode in [*(shared.latent_upscale_modes or {})]
        ]
    # 获取所有的 SD 模型信息
    def get_sd_models(self):
        # 导入 sd_models 模块
        import modules.sd_models as sd_models
        # 返回包含模型信息的列表
        return [{"title": x.title, "model_name": x.model_name, "hash": x.shorthash, "sha256": x.sha256, "filename": x.filename, "config": find_checkpoint_config_near_filename(x)} for x in sd_models.checkpoints_list.values()]

    # 获取所有的 SD VAE 模型信息
    def get_sd_vaes(self):
        # 导入 sd_vae 模块
        import modules.sd_vae as sd_vae
        # 返回包含 VAE 模型信息的列表
        return [{"model_name": x, "filename": sd_vae.vae_dict[x]} for x in sd_vae.vae_dict.keys()]

    # 获取所有的超网络信息
    def get_hypernetworks(self):
        # 返回包含超网络信息的列表
        return [{"name": name, "path": shared.hypernetworks[name]} for name in shared.hypernetworks]

    # 获取所有的人脸修复器信息
    def get_face_restorers(self):
        # 返回包含人脸修复器信息的列表
        return [{"name":x.name(), "cmd_dir": getattr(x, "cmd_dir", None)} for x in shared.face_restorers]

    # 获取所有的 RealesrGAN 模型信息
    def get_realesrgan_models(self):
        # 返回包含 RealesrGAN 模型信息的列表
        return [{"name":x.name,"path":x.data_path, "scale":x.scale} for x in get_realesrgan_models(None)]

    # 获取所有的提示样式信息
    def get_prompt_styles(self):
        # 初始化样式列表
        styleList = []
        # 遍历提示样式字典，将样式信息添加到列表中
        for k in shared.prompt_styles.styles:
            style = shared.prompt_styles.styles[k]
            styleList.append({"name":style[0], "prompt": style[1], "negative_prompt": style[2]})
        # 返回样式列表
        return styleList

    # 获取所有的嵌入信息
    def get_embeddings(self):
        # 获取嵌入数据库
        db = sd_hijack.model_hijack.embedding_db

        # 将单个嵌入对象转换为字典格式
        def convert_embedding(embedding):
            return {
                "step": embedding.step,
                "sd_checkpoint": embedding.sd_checkpoint,
                "sd_checkpoint_name": embedding.sd_checkpoint_name,
                "shape": embedding.shape,
                "vectors": embedding.vectors,
            }

        # 将所有嵌入对象转换为字典格式
        def convert_embeddings(embeddings):
            return {embedding.name: convert_embedding(embedding) for embedding in embeddings.values()}

        # 返回包含加载和跳过嵌入信息的字典
        return {
            "loaded": convert_embeddings(db.word_embeddings),
            "skipped": convert_embeddings(db.skipped_embeddings),
        }

    # 刷新检查点信息
    def refresh_checkpoints(self):
        # 使用队列锁确保线程安全
        with self.queue_lock:
            shared.refresh_checkpoints()
    # 刷新 VAE 列表
    def refresh_vae(self):
        # 使用队列锁，确保线程安全
        with self.queue_lock:
            # 刷新 VAE 列表
            shared_items.refresh_vae_list()

    # 创建嵌入
    def create_embedding(self, args: dict):
        try:
            # 开始创建嵌入任务
            shared.state.begin(job="create_embedding")
            # 创建空的嵌入文件
            filename = create_embedding(**args)
            # 重新加载嵌入，以便新的嵌入可以立即使用
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
            # 返回创建响应
            return models.CreateResponse(info=f"create embedding filename: {filename}")
        except AssertionError as e:
            # 返回错误响应
            return models.TrainResponse(info=f"create embedding error: {e}")
        finally:
            # 结束任务
            shared.state.end()

    # 创建超网络
    def create_hypernetwork(self, args: dict):
        try:
            # 开始创建超网络任务
            shared.state.begin(job="create_hypernetwork")
            # 创建空的超网络文件
            filename = create_hypernetwork(**args)
            # 返回创建响应
            return models.CreateResponse(info=f"create hypernetwork filename: {filename}")
        except AssertionError as e:
            # 返回错误响应
            return models.TrainResponse(info=f"create hypernetwork error: {e}")
        finally:
            # 结束任务
            shared.state.end()
    # 训练嵌入层模型
    def train_embedding(self, args: dict):
        # 开始训练嵌入层模型任务
        try:
            shared.state.begin(job="train_embedding")
            # 获取是否应用优化的标志
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ''
            # 如果不应用优化，则撤销优化
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                # 训练嵌入层模型，可能需要很长时间
                embedding, filename = train_embedding(**args)
            except Exception as e:
                error = e
            finally:
                # 如果不应用优化，则重新应用优化
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
            # 返回训练结果信息
            return models.TrainResponse(info=f"train embedding complete: filename: {filename} error: {error}")
        except Exception as msg:
            # 返回训练错误信息
            return models.TrainResponse(info=f"train embedding error: {msg}")
        finally:
            # 结束训练任务
            shared.state.end()

    # 训练超网络模型
    def train_hypernetwork(self, args: dict):
        # 开始训练超网络模型任务
        try:
            shared.state.begin(job="train_hypernetwork")
            # 初始化已加载的超网络列表
            shared.loaded_hypernetworks = []
            # 获取是否应用优化的标志
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ''
            # 如果不应用优化，则撤销优化
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                # 训练超网络模型
                hypernetwork, filename = train_hypernetwork(**args)
            except Exception as e:
                error = e
            finally:
                # 将条件阶段模型和第一阶段模型转移到指定设备
                shared.sd_model.cond_stage_model.to(devices.device)
                shared.sd_model.first_stage_model.to(devices.device)
                # 如果不应用优化，则重新应用优化
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
                # 结束训练任务
                shared.state.end()
            # 返回训练结果信息
            return models.TrainResponse(info=f"train embedding complete: filename: {filename} error: {error}")
        except Exception as exc:
            # 返回训练错误信息
            return models.TrainResponse(info=f"train embedding error: {exc}")
        finally:
            # 结束训练任务
            shared.state.end()
    # 获取当前进程的内存信息，包括 RAM 和 CUDA
    def get_memory(self):
        try:
            # 导入必要的库
            import os
            import psutil
            # 获取当前进程的 PID
            process = psutil.Process(os.getpid())
            # 获取进程的内存信息，只使用 rss 是跨平台保证的，因此不依赖其他值
            res = process.memory_info()
            # 计算 RAM 总量，实际值不跨平台安全，因此使用计算值
            ram_total = 100 * res.rss / process.memory_percent()
            # 构建 RAM 字典，包括空闲、已使用和总量
            ram = { 'free': ram_total - res.rss, 'used': res.rss, 'total': ram_total }
        except Exception as err:
            # 如果出现异常，返回错误信息
            ram = { 'error': f'{err}' }
        try:
            # 导入 torch 库
            import torch
            # 检查是否有可用的 CUDA
            if torch.cuda.is_available():
                # 获取 CUDA 的内存信息
                s = torch.cuda.mem_get_info()
                # 构建系统内存字典，包括空闲、已使用和总量
                system = { 'free': s[0], 'used': s[1] - s[0], 'total': s[1] }
                # 获取 CUDA 内存统计信息
                s = dict(torch.cuda.memory_stats(shared.device))
                # 构建已分配内存字典
                allocated = { 'current': s['allocated_bytes.all.current'], 'peak': s['allocated_bytes.all.peak'] }
                # 构建保留内存字典
                reserved = { 'current': s['reserved_bytes.all.current'], 'peak': s['reserved_bytes.all.peak'] }
                # 构建活跃内存字典
                active = { 'current': s['active_bytes.all.current'], 'peak': s['active_bytes.all.peak'] }
                # 构建非活跃内存字典
                inactive = { 'current': s['inactive_split_bytes.all.current'], 'peak': s['inactive_split_bytes.all.peak'] }
                # 构建警告信息字典
                warnings = { 'retries': s['num_alloc_retries'], 'oom': s['num_ooms'] }
                # 构建 CUDA 字典，包括系统、活跃、已分配、保留、非活跃和事件信息
                cuda = {
                    'system': system,
                    'active': active,
                    'allocated': allocated,
                    'reserved': reserved,
                    'inactive': inactive,
                    'events': warnings,
                }
            else:
                # 如果 CUDA 不可用，返回错误信息
                cuda = {'error': 'unavailable'}
        except Exception as err:
            # 如果出现异常，返回错误信息
            cuda = {'error': f'{err}'}
        # 返回内存信息的模型响应对象
        return models.MemoryResponse(ram=ram, cuda=cuda)
    # 获取已安装扩展的列表
    def get_extensions_list(self):
        # 从模块中导入扩展列表
        from modules import extensions
        # 调用扩展模块中的函数列出已安装的扩展
        extensions.list_extensions()
        # 初始化空的扩展列表
        ext_list = []
        # 遍历每个扩展对象
        for ext in extensions.extensions:
            # 指定ext为extensions.Extension类型
            ext: extensions.Extension
            # 从仓库中读取扩展信息
            ext.read_info_from_repo()
            # 如果扩展有远程信息，则添加到扩展列表中
            if ext.remote is not None:
                ext_list.append({
                    "name": ext.name,
                    "remote": ext.remote,
                    "branch": ext.branch,
                    "commit_hash":ext.commit_hash,
                    "commit_date":ext.commit_date,
                    "version":ext.version,
                    "enabled":ext.enabled
                })
        # 返回扩展列表
        return ext_list

    # 启动WebUI
    def launch(self, server_name, port, root_path):
        # 将路由器包含到应用中
        self.app.include_router(self.router)
        # 运行应用
        uvicorn.run(self.app, host=server_name, port=port, timeout_keep_alive=shared.cmd_opts.timeout_keep_alive, root_path=root_path)

    # 关闭WebUI
    def kill_webui(self):
        # 停止程序
        restart.stop_program()

    # 重启WebUI
    def restart_webui(self):
        # 如果可以重启，则重启程序
        if restart.is_restartable():
            restart.restart_program()
        # 返回状态码501
        return Response(status_code=501)

    # 停止WebUI
    def stop_webui(request):
        # 设置服务器命令为停止
        shared.state.server_command = "stop"
        # 返回"Stopping."
        return Response("Stopping.")
```