# `.\lucidrains\TPDNE\TPDNE_utils\tpdne.py`

```
# 导入必要的库
import os
import sys
import numpy as np
from time import time, sleep
from pathlib import Path
from functools import wraps
from PIL import Image

# 导入第三方库
from beartype import beartype
from beartype.typing import Callable, Optional
from einops import rearrange, repeat
from jinja2 import Environment, FileSystemLoader

# 获取当前脚本路径和父目录
script_path = Path(__file__)
current_dir = script_path.parents[0]
# 设置模板环境
environment = Environment(loader = FileSystemLoader(str(current_dir)))

# 获取模板文件
nginx_template = environment.get_template('nginx.conf.tmpl')
systemd_service_template = environment.get_template('tpdne.service.tmpl')

# 定义辅助函数
def exists(val):
    return val is not None

# 处理图像张量的函数
def auto_handle_image_tensor(t):
    if t.ndim == 4:
        t = t[0]  # 假设批次是第一个维度并取第一个样本

    if t.ndim == 2:
        t = rearrange(t, 'h w -> h w 1')  # 假设是灰度图像

    if t.shape[0] <= 3:
        t = rearrange(t, 'c h w -> h w c')  # 通道在前

    assert t.shape[-1] <= 3, 'image tensor must be returned in the shape (height, width, channels), where channels is either 3 or 1'

    if t.shape[-1] == 1:
        t = repeat(t, 'h w 1 -> h w c', c = 3)  # 处理单通道图像

    # 处理缩放
    if t.dtype == np.float:
        has_negatives = np.any(t < 0)

        if has_negatives:
            t = t * 127.5 + 128
        else:
            t = t * 255

        t = t.astype(np.uint8)

    return t.clip(0, 255)

# 主函数
@beartype
def sample_image_and_save_repeatedly(
    fn: Callable[..., np.ndarray],         # 返回形状为 (3, <width>, <height>) 的数组的函数
    output_path: str = './out/random',     # 输出图像的路径，不包括扩展名（将保存为 webp 格式）
    *,
    call_every_ms: int = 250,              # 采样频率
    tmp_dir: str = '/tmp',                 # 存储临时图像的目录
    num_rotated_tmp_images: int = 10,
    image_format: str = 'jpeg',
    verbose: bool = True,
    quality = 99,
    resize_image_to: Optional[int] = None,
    generate_favicon: bool = True,
    favicon_size: int = 32,
    generate_nginx_conf: bool = True,
    symbolic_link_nginx_conf: bool = True,
    nginx_sites_available_path: str = '/etc/nginx/sites-available',
    nginx_conf_filename = 'default',
    generate_systemd_service_conf: bool = False,
    systemd_service_path: str = '/etc/systemd/system',
    systemd_service_name = 'tpdne',
    domain_name = '_'
):
    assert 0 < quality <= 100
    assert favicon_size in {16, 32}
    assert image_format in {'jpeg', 'png', 'webp'}

    tmp_dir = Path(tmp_dir)
    output_path = Path(output_path)

    assert output_path.suffix == '', 'output path suffix will be automatically determined by `image_format` keyword arg'

    output_path = output_path.with_suffix(f'.{image_format}')

    call_every_seconds = call_every_ms / 1000

    assert tmp_dir.is_dir()
    root = output_path.parents[0]
    root.mkdir(parents = True, exist_ok = True)

    tmp_image_index = 0

    # 链接 nginx
    if generate_nginx_conf:
        nginx_sites_path = Path(nginx_sites_available_path)
        nginx_sites_conf_path = nginx_sites_path / nginx_conf_filename

        assert nginx_sites_path.is_dir()

        nginx_conf_text = nginx_template.render(
            root = str(root.resolve()),
            index = output_path.name,
            server_name = domain_name
        )

        tmp_conf_path = Path(tmp_dir / 'nginx.server.conf')
        tmp_conf_path.write_text(nginx_conf_text)

        print(f'nginx server conf generated at {str(tmp_conf_path)}')

        if symbolic_link_nginx_conf:
            os.system(f'ln -nfs {str(tmp_conf_path)} {nginx_sites_conf_path}')

            print(f'nginx conf linked to {nginx_sites_conf_path}\nrun `systemctl reload nginx` for it to be in effect')
    # 如果需要生成 systemd 服务配置文件，并且当前不是在 systemd 中启动
    if generate_systemd_service_conf and not exists(os.getenv('LAUNCHED_FROM_SYSTEMD', None)):

        # 设置 systemd 服务路径
        systemd_service_path = Path(systemd_service_path)
        # 设置 systemd 服务配置文件路径
        systemd_service_conf_path = systemd_service_path / f'{systemd_service_name}.service'

        # 断言 systemd 服务路径是一个目录
        assert systemd_service_path.is_dir()

        # 使用 systemd 服务模板渲染 systemd 配置文本
        systemd_conf_text = systemd_service_template.render(
            working_directory = str(current_dir.resolve()),
            python_executable = sys.executable,
            script_path = str(script_path.resolve())
        )

        # 创建临时服务路径，写入 systemd 配置文本
        tmp_service_path = Path(tmp_dir / 'tpdne.services')
        tmp_service_path.write_text(systemd_conf_text)

        # 创建符号链接，将临时服务路径链接到 systemd 服务配置文件路径
        os.system(f'ln -nfs {str(tmp_service_path)} {str(systemd_service_conf_path)}')

        # 打印提示信息
        print(f'service {systemd_service_name}.service created at {str(systemd_service_conf_path)}')
        print(f'run `systemctl enable {systemd_service_name}.service` to start this script')
        print(f'then run `systemctl status {systemd_service_name}.service` to see the status')
        # 退出程序
        exit()

    # 在一个无限循环中调用函数 `fn`
    while True:
        start = time()
        # 调用函数 `fn` 获取图像张量
        image_tensor = fn()

        # 对图像张量进行处理
        image_tensor = auto_handle_image_tensor(image_tensor)

        # 计算临时图像索引
        tmp_image_index = (tmp_image_index + 1) % num_rotated_tmp_images
        tmp_path = str(tmp_dir / f'{tmp_image_index}.{image_format}')

        # 使用 PIL 创建图像对象
        pil_image = Image.fromarray(image_tensor, 'RGB')

        # 如果存在 resize_image_to 参数，对图像进行缩放
        if exists(resize_image_to):
            pil_image = pil_image.resize((resize_image_to, resize_image_to))

        # 根据图像格式设置不同的参数
        image_save_kwargs = dict()

        if image_format == 'jpeg':
            image_save_kwargs = dict(optimize = True, progressive = True)
        elif image_format == 'webp':
            image_save_kwargs = dict(format = 'webp')

        # 保存图像到临时路径
        pil_image.save(tmp_path, quality = quality, **image_save_kwargs)

        # 创建符号链接，将临时图像路径链接到输出路径
        os.system(f'ln -nfs {tmp_path} {output_path}')

        # 如果需要生成 favicon
        if generate_favicon:
            tmp_favicon_path = str(tmp_dir / f'favicon_{tmp_image_index}.png')
            output_favicon_path = output_path.parents[0] / 'favicon.png'

            # 缩小图像为 favicon 大小
            small_pil_image = pil_image.resize((favicon_size, favicon_size))
            small_pil_image.save(tmp_favicon_path)
            os.system(f'ln -nfs {tmp_favicon_path} {output_favicon_path}')

        # 计算执行时间
        elapsed = time() - start

        # 如果 verbose 为 True，打印执行时间和路径信息
        if verbose:
            print(f'{elapsed:.3f}s - tmp image at {tmp_path}, output image at {output_path}')

        # 确保至少每隔 `call_every_seconds` 秒生成一次图像
        if elapsed >= call_every_seconds:
            continue

        # 休眠直到下一次生成图像的时间点
        sleep(call_every_seconds - elapsed)
```