# `.\lucidrains\ddpm-proteins\train.py`

```py
import os
import torch
import sidechainnet as scn

from PIL import Image
from random import randrange

import torch
import torch.nn.functional as F
from torch import optim

from ddpm_proteins import Unet, GaussianDiffusion
from ddpm_proteins.utils import save_heatmap, broadcat, get_msa_attention_embeddings, symmetrize, get_msa_transformer, pad_image_to

from einops import rearrange

os.makedirs('./.tmps', exist_ok = True)

# 定义常量

NUM_ITERATIONS = int(2e6)
IMAGE_SIZE = 256
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 8
LEARNING_RATE = 2e-5
SAMPLE_EVERY = 200
SCALE_DISTANCE_BY = 1e2

# 实验追踪器

import wandb
wandb.init(project = 'ddpm-proteins')
wandb.run.name = f'proteins of length {IMAGE_SIZE} or less'
wandb.run.save()

# 定义模型

model = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8),
    channels = 1,
    condition_dim = 1 + 144  # mask (1) + attention embedding size (144)
)

diffusion = GaussianDiffusion(
    model,
    image_size = IMAGE_SIZE,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

def cycle(loader, thres = 256):
    while True:
        for data in loader:
            if data.seqs.shape[1] <= thres:
                yield data

data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = BATCH_SIZE,
    dynamic_batching = False
)

model, batch_converter = get_msa_transformer()
model = model.cuda(1) # 将 msa transformer 放在 cuda 设备 1 上

opt = optim.Adam(diffusion.parameters(), lr = LEARNING_RATE)

train_dl = cycle(data['train'], thres = IMAGE_SIZE)
valid_dl = cycle(data['test'], thres = IMAGE_SIZE)

diffusion = diffusion.cuda()

upper_triangular_mask = torch.ones(IMAGE_SIZE, IMAGE_SIZE).triu_(1).bool().cuda()

# 迭代训练模型

for ind in range(NUM_ITERATIONS):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(train_dl)
        ids, seqs, coords, masks = batch.pids, batch.seqs, batch.crds, batch.msks
        seqs = seqs.argmax(dim = -1)

        coords = coords.reshape(BATCH_SIZE, -1, 14, 3)
        coords = coords[:, :, 1].cuda() # 选择 alpha 碳

        dist = torch.cdist(coords, coords)
        data = dist[:, None, :, :]

        crossed_mask = (masks[:, None, :, None] * masks[:, None, None, :]).cuda()
        data.masked_fill_(~crossed_mask.bool(), 0.)

        data = pad_image_to(data, IMAGE_SIZE, value = 0.)
        crossed_mask = pad_image_to(crossed_mask, IMAGE_SIZE, value = 0.)

        data = (data / SCALE_DISTANCE_BY).clamp(0., 1.)

        data = data * upper_triangular_mask[None, None, :, :]

        msa_attention_embeds = get_msa_attention_embeddings(model, batch_converter, seqs, ids)
        msa_attention_embeds = pad_image_to(msa_attention_embeds, IMAGE_SIZE)

        condition_tensor = broadcat((msa_attention_embeds.cuda(0), crossed_mask.float()), dim = 1)

        loss = diffusion(data, condition_tensor = condition_tensor)
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print(loss.item())
    wandb.log({'loss': loss.item()})
    opt.step()
    opt.zero_grad()
    # 检查是否满足采样条件
    if (ind % SAMPLE_EVERY) == 0:
        # 从验证数据加载一个批次
        batch = next(valid_dl)
        # 获取批次中的蛋白质 ID、序列、坐标和掩码
        ids, seqs, coords, masks = batch.pids, batch.seqs, batch.crds, batch.msks
        # 将序列转换为 one-hot 编码
        seqs = seqs.argmax(dim=-1)

        # 重新整形坐标数据以便提取 alpha 碳原子
        coords = coords.reshape(BATCH_SIZE, -1, 14, 3)
        coords = coords[:, :, 1].cuda()

        # 计算坐标之间的距离
        dist = torch.cdist(coords, coords)
        data = dist[:, None, :, :]

        # 创建交叉掩码
        crossed_mask = (masks[:, None, :, None] * masks[:, None, None, :]).cuda()
        # 将数据中未交叉的部分填充为 0
        data.masked_fill_(~crossed_mask.bool(), 0.)

        # 将数据填充到指定大小，并缩放到指定范围
        data = pad_image_to(data, IMAGE_SIZE, value=0.)
        valid_data = (data / SCALE_DISTANCE_BY).clamp(0., 1.)

        # 将交叉掩码填充到指定大小
        crossed_mask = pad_image_to(crossed_mask, IMAGE_SIZE, value=0.)[:1].float()

        # 获取 MSA 注意力嵌入
        msa_attention_embeds = get_msa_attention_embeddings(model, batch_converter, seqs[:1], ids[:1])
        msa_attention_embeds = pad_image_to(msa_attention_embeds, IMAGE_SIZE)

        # 创建条件张量
        condition_tensor = broadcat((msa_attention_embeds.cuda(0), crossed_mask.float()), dim=1)

        # 从扩散过程中采样生成图像
        sampled = diffusion.sample(batch_size=1, condition_tensor=condition_tensor)[0][0]

        # 将采样结果限制在 0 到 1 之间，并根据上三角掩码进行修正
        sampled = sampled.clamp(0., 1.) * upper_triangular_mask
        sampled = symmetrize(sampled)

        # 保存生成的图像和相关信息
        img = save_heatmap(sampled, './.tmps/validation.png', dpi=100, return_image=True)
        crossed_mask_img = save_heatmap(crossed_mask[0][0], './.tmps/mask.png', dpi=100, return_image=True)
        truth_img = save_heatmap(valid_data[0][0], './.tmps/truth.png', dpi=100, return_image=True)

        # 将图像上传到 wandb
        wandb.log({'sample': wandb.Image(img), 'mask': wandb.Image(crossed_mask_img), 'truth': wandb.Image(truth_img)})
```