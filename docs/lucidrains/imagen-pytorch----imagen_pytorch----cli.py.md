# `.\lucidrains\imagen-pytorch\imagen_pytorch\cli.py`

```py
import click
import torch
from pathlib import Path
import pkgutil

from imagen_pytorch import load_imagen_from_checkpoint
from imagen_pytorch.version import __version__
from imagen_pytorch.data import Collator
from imagen_pytorch.utils import safeget
from imagen_pytorch import ImagenTrainer, ElucidatedImagenConfig, ImagenConfig
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import json

# 定义一个函数，用于检查值是否存在
def exists(val):
    return val is not None

# 定义一个简单的字符串处理函数，将特殊字符替换为下划线，并截取指定长度
def simple_slugify(text: str, max_length = 255):
    return text.replace('-', '_').replace(',', '').replace(' ', '_').replace('|', '--').strip('-_./\\')[:max_length]

# 主函数
def main():
    pass

# 创建一个命令组
@click.group()
def imagen():
    pass

# 创建一个命令，用于从 Imagen 模型检查点中进行采样
@imagen.command(help = 'Sample from the Imagen model checkpoint')
@click.option('--model', default = './imagen.pt', help = 'path to trained Imagen model')
@click.option('--cond_scale', default = 5, help = 'conditioning scale (classifier free guidance) in decoder')
@click.option('--load_ema', default = True, help = 'load EMA version of unets if available')
@click.argument('text')
def sample(
    model,
    cond_scale,
    load_ema,
    text
):
    model_path = Path(model)
    full_model_path = str(model_path.resolve())
    assert model_path.exists(), f'model not found at {full_model_path}'
    loaded = torch.load(str(model_path))

    # 获取版本信息
    version = safeget(loaded, 'version')
    print(f'loading Imagen from {full_model_path}, saved at version {version} - current package version is {__version__}')

    # 获取 Imagen 参数和类型
    imagen = load_imagen_from_checkpoint(str(model_path), load_ema_if_available = load_ema)
    imagen.cuda()

    # 生成图像
    pil_image = imagen.sample([text], cond_scale = cond_scale, return_pil_images = True)

    image_path = f'./{simple_slugify(text)}.png'
    pil_image[0].save(image_path)

    print(f'image saved to {str(image_path)}')
    return

# 创建一个命令，用于生成 Imagen 模型的配置
@imagen.command(help = 'Generate a config for the Imagen model')
@click.option('--path', default = './imagen_config.json', help = 'Path to the Imagen model config')
def config(
    path
):
    data = pkgutil.get_data(__name__, 'default_config.json').decode("utf-8") 
    with open(path, 'w') as f:
        f.write(data)

# 创建一个命令，用于训练 Imagen 模型
@imagen.command(help = 'Train the Imagen model')
@click.option('--config', default = './imagen_config.json', help = 'Path to the Imagen model config')
@click.option('--unet', default = 1, help = 'Unet to train', type = click.IntRange(1, 3, False, True, True))
@click.option('--epoches', default = 50, help = 'Amount of epoches to train for')
def train(
    config,
    unet,
    epoches,
):
    # 检查配置文件路径
    config_path = Path(config)
    full_config_path = str(config_path.resolve())
    assert config_path.exists(), f'config not found at {full_config_path}'
    
    with open(config_path, 'r') as f:
        config_data = json.loads(f.read())

    assert 'checkpoint_path' in config_data, 'checkpoint path not found in config'
    
    model_path = Path(config_data['checkpoint_path'])
    full_model_path = str(model_path.resolve())
    
    # 设置 Imagen 配置
    imagen_config_klass = ElucidatedImagenConfig if config_data['type'] == 'elucidated' else ImagenConfig
    imagen = imagen_config_klass(**config_data['imagen']).create()

    trainer = ImagenTrainer(
    imagen = imagen,
        **config_data['trainer']
    )

    # 加载模型
    if model_path.exists():
        loaded = torch.load(str(model_path))
        version = safeget(loaded, 'version')
        print(f'loading Imagen from {full_model_path}, saved at version {version} - current package version is {__version__}')
        trainer.load(model_path)
        
    if torch.cuda.is_available():
        trainer = trainer.cuda()

    size = config_data['imagen']['image_sizes'][unet-1]

    max_batch_size = config_data['max_batch_size'] if 'max_batch_size' in config_data else 1

    channels = 'RGB'
    # 检查配置数据中是否包含 'channels' 键
    if 'channels' in config_data['imagen']:
        # 断言通道数在 1 到 4 之间，否则抛出异常
        assert config_data['imagen']['channels'] > 0 and config_data['imagen']['channels'] < 5, 'Imagen only support 1 to 4 channels L, LA, RGB, RGBA'
        # 根据通道数设置 channels 变量
        if config_data['imagen']['channels'] == 4:
            channels = 'RGBA' # Color with alpha
        elif config_data['imagen']['channels'] == 2:
            channels == 'LA' # Luminance (Greyscale) with alpha
        elif config_data['imagen']['channels'] == 1:
            channels = 'L' # Luminance (Greyscale)

    # 断言配置数据中包含 'batch_size' 键
    assert 'batch_size' in config_data['dataset'], 'A batch_size is required in the config file'
    
    # 加载并添加训练数据集和验证数据集
    ds = load_dataset(config_data['dataset_name'])
    
    train_ds = None
    
    # 如果有训练和验证数据集，则将它们合并成一个数据集，以便训练器处理拆分
    if 'train' in ds and 'valid' in ds:
        train_ds = concatenate_datasets([ds['train'], ds['valid']])
    elif 'train' in ds:
        train_ds = ds['train']
    elif 'valid' in ds:
        train_ds = ds['valid']
    else:
        train_ds = ds
        
    # 断言训练数据集不为空
    assert train_ds is not None, 'No train dataset could be fetched from the dataset name provided'
    
    # 添加训练数据集到训练器
    trainer.add_train_dataset(
        ds = train_ds,
        collate_fn = Collator(
            image_size = size,
            image_label = config_data['image_label'],
            text_label = config_data['text_label'],
            url_label = config_data['url_label'],
            name = imagen.text_encoder_name,
            channels = channels
        ),
        **config_data['dataset']
    )
    
    # 检查是否需要验证、采样和保存
    should_validate = trainer.split_valid_from_train and 'validate_at_every' in config_data
    should_sample = 'sample_texts' in config_data and 'sample_at_every' in config_data
    should_save = 'save_at_every' in config_data
    
    # 根据配置设置验证、采样和保存的频率
    valid_at_every = config_data['validate_at_every'] if should_validate else 0
    assert isinstance(valid_at_every, int), 'validate_at_every must be an integer'
    sample_at_every = config_data['sample_at_every'] if should_sample else 0
    assert isinstance(sample_at_every, int), 'sample_at_every must be an integer'
    save_at_every = config_data['save_at_every'] if should_save else 0
    assert isinstance(save_at_every, int), 'save_at_every must be an integer'
    sample_texts = config_data['sample_texts'] if should_sample else []
    assert isinstance(sample_texts, list), 'sample_texts must be a list'
    
    # 当 should_sample 为真时，检查 sample_texts 不为空
    assert not should_sample or len(sample_texts) > 0, 'sample_texts must not be empty when sample_at_every is set'
    
    # 循环训练模型
    for i in range(epoches):
        for _ in tqdm(range(len(trainer.train_dl)):
            # 训练模型并获取损失
            loss = trainer.train_step(unet_number = unet, max_batch_size = max_batch_size)
            print(f'loss: {loss}')

        # 在指定的验证频率进行验证
        if not (i % valid_at_every) and i > 0 and trainer.is_main and should_validate:
            valid_loss = trainer.valid_step(unet_number = unet, max_batch_size = max_batch_size)
            print(f'valid loss: {valid_loss}')

        # 在指定的采样频率进行采样并保存图片
        if not (i % save_at_every) and i > 0 and trainer.is_main and should_sample:
            images = trainer.sample(texts = [sample_texts], batch_size = 1, return_pil_images = True, stop_at_unet_number = unet)
            images[0].save(f'./sample-{i // 100}.png')
            
        # 在指定的保存频率保存模型
        if not (i % save_at_every) and i > 0 and trainer.is_main and should_save:
            trainer.save(model_path)

    # 最终保存模型
    trainer.save(model_path)
```