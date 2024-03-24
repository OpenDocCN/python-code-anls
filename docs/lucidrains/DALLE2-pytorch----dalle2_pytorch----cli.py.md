# `.\lucidrains\DALLE2-pytorch\dalle2_pytorch\cli.py`

```
# 导入需要的库
import click
import torch
import torchvision.transforms as T
from functools import reduce
from pathlib import Path

# 导入自定义模块
from dalle2_pytorch import DALLE2, Decoder, DiffusionPrior

# 定义函数，根据键路径获取字典中的值
def safeget(dictionary, keys, default = None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split('.'), dictionary)

# 简单的文本转换函数，将特殊字符替换为下划线
def simple_slugify(text, max_length = 255):
    return text.replace("-", "_").replace(",", "").replace(" ", "_").replace("|", "--").strip('-_')[:max_length]

# 获取包的版本号
def get_pkg_version():
    from pkg_resources import get_distribution
    return get_distribution('dalle2_pytorch').version

# 主函数
def main():
    pass

# 命令行参数设置
@click.command()
@click.option('--model', default = './dalle2.pt', help = 'path to trained DALL-E2 model')
@click.option('--cond_scale', default = 2, help = 'conditioning scale (classifier free guidance) in decoder')
@click.argument('text')
def dream(
    model,
    cond_scale,
    text
):
    # 获取模型路径
    model_path = Path(model)
    full_model_path = str(model_path.resolve())
    # 检查模型是否存在
    assert model_path.exists(), f'model not found at {full_model_path}'
    # 加载模型
    loaded = torch.load(str(model_path))

    # 获取模型版本号
    version = safeget(loaded, 'version')
    print(f'loading DALL-E2 from {full_model_path}, saved at version {version} - current package version is {get_pkg_version()}')

    # 获取初始化参数
    prior_init_params = safeget(loaded, 'init_params.prior')
    decoder_init_params = safeget(loaded, 'init_params.decoder')
    model_params = safeget(loaded, 'model_params')

    # 初始化 DiffusionPrior 和 Decoder
    prior = DiffusionPrior(**prior_init_params)
    decoder = Decoder(**decoder_init_params)

    # 初始化 DALLE2 模型
    dalle2 = DALLE2(prior, decoder)
    dalle2.load_state_dict(model_params)

    # 生成图像
    image = dalle2(text, cond_scale = cond_scale)

    # 转换为 PIL 图像并保存
    pil_image = T.ToPILImage()(image)
    return pil_image.save(f'./{simple_slugify(text)}.png')
```