# `stable-diffusion-webui\modules\gfpgan_model.py`

```py
# 导入所需的库
import os
# 导入 facexlib 库
import facexlib
# 导入 gfpgan 库
import gfpgan
# 导入 face_restoration 模块
import modules.face_restoration
# 导入 paths, shared, devices, modelloader, errors 模块
from modules import paths, shared, devices, modelloader, errors

# 设置模型目录
model_dir = "GFPGAN"
# 用户路径为空
user_path = None
# 模型路径为 models_path 下的 model_dir
model_path = os.path.join(paths.models_path, model_dir)
# 模型文件路径为空
model_file_path = None
# 模型 URL
model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
# 是否有 GFPGAN 模型
have_gfpgan = False
# 加载的 GFPGAN 模型为空
loaded_gfpgan_model = None

# 定义 gfpgann 函数
def gfpgann():
    global loaded_gfpgan_model
    global model_path
    global model_file_path
    # 如果已加载 GFPGAN 模型，则将其转移到指定设备并返回
    if loaded_gfpgan_model is not None:
        loaded_gfpgan_model.gfpgan.to(devices.device_gfpgan)
        return loaded_gfpgan_model

    # 如果 gfpgan_constructor 为空，则返回 None
    if gfpgan_constructor is None:
        return None

    # 加载模型
    models = modelloader.load_models(model_path, model_url, user_path, ext_filter=['.pth'])

    # 根据加载的模型数量和类型确定模型文件路径
    if len(models) == 1 and models[0].startswith("http"):
        model_file = models[0]
    elif len(models) != 0:
        gfp_models = []
        for item in models:
            if 'GFPGAN' in os.path.basename(item):
                gfp_models.append(item)
        latest_file = max(gfp_models, key=os.path.getctime)
        model_file = latest_file
    else:
        print("Unable to load gfpgan model!")
        return None

    # 设置 facexlib.detection.retinaface.device 为指定设备
    if hasattr(facexlib.detection.retinaface, 'device'):
        facexlib.detection.retinaface.device = devices.device_gfpgan
    # 设置模型文件路径
    model_file_path = model_file
    # 使用 gfpgan_constructor 加载模型
    model = gfpgan_constructor(model_path=model_file, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=devices.device_gfpgan)
    loaded_gfpgan_model = model

    return model

# 将模型发送到指定设备
def send_model_to(model, device):
    model.gfpgan.to(device)
    model.face_helper.face_det.to(device)
    model.face_helper.face_parse.to(device)

# 修复图像中的人脸
def gfpgan_fix_faces(np_image):
    # 加载 GFPGAN 模型
    model = gfpgann()
    if model is None:
        return np_image

    # 将模型发送到指定设备
    send_model_to(model, devices.device_gfpgan)

    # 将图像转换为 BGR 格式
    np_image_bgr = np_image[:, :, ::-1]
    # 使用模型对图像进行增强，得到裁剪后的人脸、恢复后的人脸和 GFPGAN 输出的 BGR 图像
    cropped_faces, restored_faces, gfpgan_output_bgr = model.enhance(np_image_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    # 将 GFPGAN 输出的 BGR 图像转换为 RGB 图像
    np_image = gfpgan_output_bgr[:, :, ::-1]

    # 清除模型中的所有人脸信息
    model.face_helper.clean_all()

    # 如果设置了人脸恢复卸载选项，则将模型发送到 CPU
    if shared.opts.face_restoration_unload:
        send_model_to(model, devices.cpu)

    # 返回 RGB 图像
    return np_image
# 初始化 GFPGAN 构造函数为 None
gfpgan_constructor = None

# 设置模型的函数，接受一个目录名参数
def setup_model(dirname):
    try:
        # 创建模型路径，如果路径已存在则不做任何操作
        os.makedirs(model_path, exist_ok=True)
        # 导入 GFPGANer 类和 facexlib 模块中的 detection 和 parsing 模块
        from gfpgan import GFPGANer
        from facexlib import detection, parsing  # noqa: F401
        # 声明全局变量
        global user_path
        global have_gfpgan
        global gfpgan_constructor
        global model_file_path

        # 将 facexlib_path 初始化为 model_path，如果传入了 dirname 参数，则使用传入的 dirname
        facexlib_path = model_path
        if dirname is not None:
            facexlib_path = dirname

        # 备份原始的 load_file_from_url 函数
        load_file_from_url_orig = gfpgan.utils.load_file_from_url
        facex_load_file_from_url_orig = facexlib.detection.load_file_from_url
        facex_load_file_from_url_orig2 = facexlib.parsing.load_file_from_url

        # 定义新的 load_file_from_url 函数，用于加载模型文件
        def my_load_file_from_url(**kwargs):
            return load_file_from_url_orig(**dict(kwargs, model_dir=model_file_path))

        # 定义新的 facex_load_file_from_url 函数，用于加载 facexlib 模块中的模型文件
        def facex_load_file_from_url(**kwargs):
            return facex_load_file_from_url_orig(**dict(kwargs, save_dir=facexlib_path, model_dir=None))

        # 定义新的 facex_load_file_from_url2 函数，用于加载 facexlib 模块中的模型文件
        def facex_load_file_from_url2(**kwargs):
            return facex_load_file_from_url_orig2(**dict(kwargs, save_dir=facexlib_path, model_dir=None))

        # 替换原始的 load_file_from_url 函数
        gfpgan.utils.load_file_from_url = my_load_file_from_url
        facexlib.detection.load_file_from_url = facex_load_file_from_url
        facexlib.parsing.load_file_from_url = facex_load_file_from_url2

        # 设置 user_path 为传入的 dirname
        user_path = dirname
        # 设置 have_gfpgan 为 True
        have_gfpgan = True
        # 初始化 gfpgan_constructor 为 GFPGANer 类

        gfpgan_constructor = GFPGANer

        # 定义 FaceRestorerGFPGAN 类，继承自 modules.face_restoration.FaceRestoration
        class FaceRestorerGFPGAN(modules.face_restoration.FaceRestoration):
            # 返回类名
            def name(self):
                return "GFPGAN"

            # 重写 restore 方法，调用 gfpgan_fix_faces 函数修复人脸
            def restore(self, np_image):
                return gfpgan_fix_faces(np_image)

        # 将 FaceRestorerGFPGAN 实例添加到 shared.face_restorers 列表中
        shared.face_restorers.append(FaceRestorerGFPGAN())
    except Exception:
        # 报告设置 GFPGAN 时出现的错误
        errors.report("Error setting up GFPGAN", exc_info=True)
```