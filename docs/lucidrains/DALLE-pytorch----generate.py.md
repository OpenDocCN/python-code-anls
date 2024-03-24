# `.\lucidrains\DALLE-pytorch\generate.py`

```py
# 导入必要的库
import argparse
from pathlib import Path
from tqdm import tqdm

# 导入 torch 库
import torch

# 导入 einops 库中的 repeat 函数
from einops import repeat

# 导入 vision 相关库
from PIL import Image
from torchvision.utils import make_grid, save_image

# 导入 dalle_pytorch 库中的类和工具
from dalle_pytorch import __version__
from dalle_pytorch import DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, YttmTokenizer, ChineseTokenizer

# 参数解析
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--dalle_path', type = str, required = True,
                    help='path to your trained DALL-E')

parser.add_argument('--vqgan_model_path', type=str, default = None,
                   help='path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)')

parser.add_argument('--vqgan_config_path', type=str, default = None,
                   help='path to your trained VQGAN config. This should be a .yaml file.  (only valid when taming option is enabled)')

parser.add_argument('--text', type = str, required = True,
                    help='your text prompt')

parser.add_argument('--num_images', type = int, default = 128, required = False,
                    help='number of images')

parser.add_argument('--batch_size', type = int, default = 4, required = False,
                    help='batch size')

parser.add_argument('--top_k', type = float, default = 0.9, required = False,
                    help='top k filter threshold')

parser.add_argument('--outputs_dir', type = str, default = './outputs', required = False,
                    help='output directory')

parser.add_argument('--bpe_path', type = str,
                    help='path to your huggingface BPE json file')

parser.add_argument('--hug', dest='hug', action = 'store_true')

parser.add_argument('--chinese', dest='chinese', action = 'store_true')

parser.add_argument('--taming', dest='taming', action='store_true')

parser.add_argument('--gentxt', dest='gentxt', action='store_true')

# 解析参数
args = parser.parse_args()

# 辅助函数
def exists(val):
    return val is not None

# 根据参数设置 tokenizer
if exists(args.bpe_path):
    klass = HugTokenizer if args.hug else YttmTokenizer
    tokenizer = klass(args.bpe_path)
elif args.chinese:
    tokenizer = ChineseTokenizer()

# 加载 DALL-E 模型
dalle_path = Path(args.dalle_path)
assert dalle_path.exists(), 'trained DALL-E must exist'

load_obj = torch.load(str(dalle_path))
dalle_params, vae_params, weights, vae_class_name, version = load_obj.pop('hparams'), load_obj.pop('vae_params'), load_obj.pop('weights'), load_obj.pop('vae_class_name', None), load_obj.pop('version', None)

# 友好打印
if exists(version):
    print(f'Loading a model trained with DALLE-pytorch version {version}')
else:
    print('You are loading a model trained on an older version of DALL-E pytorch - it may not be compatible with the most recent version')

# 加载 VAE 模型
if args.taming:
    vae = VQGanVAE(args.vqgan_model_path, args.vqgan_config_path)
elif vae_params is not None:
    vae = DiscreteVAE(**vae_params)
else:
    vae = OpenAIDiscreteVAE()

assert not (exists(vae_class_name) and vae.__class__.__name__ != vae_class_name), f'you trained DALL-E using {vae_class_name} but are trying to generate with {vae.__class__.__name__} - please make sure you are passing in the correct paths and settings for the VAE to use for generation'

# 重建 DALL-E 模型
dalle = DALLE(vae = vae, **dalle_params).cuda()
dalle.load_state_dict(weights)

# 生成图片
image_size = vae.image_size
texts = args.text.split('|')

for j, text in tqdm(enumerate(texts)):
    if args.gentxt:
        text_tokens, gen_texts = dalle.generate_texts(tokenizer, text=text, filter_thres = args.top_k)
        text = gen_texts[0]
    else:
        text_tokens = tokenizer.tokenize([text], dalle.text_seq_len).cuda()

    text_tokens = repeat(text_tokens, '() n -> b n', b = args.num_images)

    outputs = []
    # 使用 tqdm 分块处理文本标记，每块大小为 args.batch_size，显示进度条描述为生成图像的文本
    for text_chunk in tqdm(text_tokens.split(args.batch_size), desc = f'generating images for - {text}'):
        # 生成图像，根据文本块和筛选阈值 args.top_k
        output = dalle.generate_images(text_chunk, filter_thres = args.top_k)
        # 将生成的图像添加到输出列表中
        outputs.append(output)

    # 将所有输出图像拼接成一个张量
    outputs = torch.cat(outputs)

    # 保存所有图像

    # 定义文件名为文本
    file_name = text 
    # 定义输出目录为 args.outputs_dir 下的文件名替换空格为下划线后的前100个字符
    outputs_dir = Path(args.outputs_dir) / file_name.replace(' ', '_')[:(100)]
    # 创建输出目录，如果不存在则创建，存在则忽略
    outputs_dir.mkdir(parents = True, exist_ok = True)

    # 遍历输出图像，保存为 PNG 格式
    for i, image in tqdm(enumerate(outputs), desc = 'saving images'):
        # 保存图像为 PNG 格式，文件名为序号.png，进行归一化
        save_image(image, outputs_dir / f'{i}.png', normalize=True)
        # 将文本写入 caption.txt 文件
        with open(outputs_dir / 'caption.txt', 'w') as f:
            f.write(file_name)

    # 打印生成的图像数量和输出目录路径
    print(f'created {args.num_images} images at "{str(outputs_dir)}"')
```