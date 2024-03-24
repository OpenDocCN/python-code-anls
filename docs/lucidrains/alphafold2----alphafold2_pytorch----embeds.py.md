# `.\lucidrains\alphafold2\alphafold2_pytorch\embeds.py`

```
# 导入 torch 库
import torch
# 导入 torch 中的函数库
import torch.nn.functional as F
# 从 torch 中导入 nn 模块
from torch import nn

# 从 alphafold2_pytorch.utils 中导入 get_msa_embedd, get_esm_embedd, get_prottran_embedd, exists 函数
from alphafold2_pytorch.utils import get_msa_embedd, get_esm_embedd, get_prottran_embedd, exists
# 从 alphafold2_pytorch.constants 中导入 MSA_MODEL_PATH, MSA_EMBED_DIM, ESM_MODEL_PATH, ESM_EMBED_DIM, PROTTRAN_EMBED_DIM 常量
from alphafold2_pytorch.constants import MSA_MODEL_PATH, MSA_EMBED_DIM, ESM_MODEL_PATH, ESM_EMBED_DIM, PROTTRAN_EMBED_DIM

# 从 einops 中导入 rearrange 函数
from einops import rearrange

# 定义 ProtTranEmbedWrapper 类，继承自 nn.Module
class ProtTranEmbedWrapper(nn.Module):
    # 初始化函数
    def __init__(self, *, alphafold2):
        super().__init__()
        # 从 transformers 中导入 AutoTokenizer, AutoModel
        from transformers import AutoTokenizer, AutoModel

        # 初始化属性 alphafold2
        self.alphafold2 = alphafold2
        # 创建线性层，用于将 PROTTRAN_EMBED_DIM 维度的数据映射到 alphafold2.dim 维度
        self.project_embed = nn.Linear(PROTTRAN_EMBED_DIM, alphafold2.dim)
        # 使用 'Rostlab/prot_bert' 模型初始化 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
        # 使用 'Rostlab/prot_bert' 模型初始化 model
        self.model = AutoModel.from_pretrained('Rostlab/prot_bert')

    # 前向传播函数
    def forward(self, seq, msa, msa_mask = None, **kwargs):
        # 获取设备信息
        device = seq.device
        # 获取 msa 的数量
        num_msa = msa.shape[1]
        # 将 msa 展平
        msa_flat = rearrange(msa, 'b m n -> (b m) n')

        # 获取序列的 PROTTRAN 嵌入
        seq_embed = get_prottran_embedd(seq, self.model, self.tokenizer, device = device)
        # 获取 msa 的 PROTTRAN 嵌入
        msa_embed = get_prottran_embedd(msa_flat, self.model, self.tokenizer, device = device)

        # 将序列和 msa 的嵌入映射到指定维度
        seq_embed, msa_embed = map(self.project_embed, (seq_embed, msa_embed))
        # 重新排列 msa_embed 的维度
        msa_embed = rearrange(msa_embed, '(b m) n d -> b m n d', m = num_msa)

        # 调用 alphafold2 模型进行预测
        return self.alphafold2(seq, msa, seq_embed = seq_embed, msa_embed = msa_embed, msa_mask = msa_mask, **kwargs)

# 定义 MSAEmbedWrapper 类，继承自 nn.Module
class MSAEmbedWrapper(nn.Module):
    # 初始化函数
    def __init__(self, *, alphafold2):
        super().__init__()
        # 初始化属性 alphafold2
        self.alphafold2 = alphafold2

        # 加载 MSA 模型和字母表
        model, alphabet = torch.hub.load(*MSA_MODEL_PATH) 
        batch_converter = alphabet.get_batch_converter()

        # 初始化 model, batch_converter, project_embed 属性
        self.model = model
        self.batch_converter = batch_converter
        self.project_embed = nn.Linear(MSA_EMBED_DIM, alphafold2.dim) if MSA_EMBED_DIM != alphafold2.dim else nn.Identity()

    # 前向传播函数
    def forward(self, seq, msa, msa_mask = None, **kwargs):
        # 断言序列和 msa 的长度相同
        assert seq.shape[-1] == msa.shape[-1], 'sequence and msa must have the same length if you wish to use MSA transformer embeddings'
        # 获取 model, batch_converter, device 信息
        model, batch_converter, device = self.model, self.batch_converter, seq.device

        # 将序列和 msa 连接
        seq_and_msa = torch.cat((seq.unsqueeze(1), msa), dim = 1)

        if exists(msa_mask):
            # 处理 MSA 中完全填充的行
            num_msa = msa_mask.any(dim = -1).sum(dim = -1).tolist()
            seq_and_msa_list = seq_and_msa.unbind(dim = 0)
            num_rows = seq_and_msa.shape[1]

            embeds = []
            for num, batch_el in zip(num_msa, seq_and_msa_list):
                batch_el = rearrange(batch_el, '... -> () ...')
                batch_el = batch_el[:, :num]
                embed = get_msa_embedd(batch_el, model, batch_converter, device = device)
                embed = F.pad(embed, (0, 0, 0, 0, 0, num_rows - num), value = 0.)
                embeds.append(embed)

            embeds = torch.cat(embeds, dim = 0)
        else:
            embeds = get_msa_embedd(seq_and_msa, model, batch_converter, device = device)

        # 映射嵌入到指定维度
        embeds = self.project_embed(embeds)
        seq_embed, msa_embed = embeds[:, 0], embeds[:, 1:]

        # 调用 alphafold2 模型进行预测
        return self.alphafold2(seq, msa, seq_embed = seq_embed, msa_embed = msa_embed, msa_mask = msa_mask, **kwargs)

# 定义 ESMEmbedWrapper 类，继承自 nn.Module
class ESMEmbedWrapper(nn.Module):
    # 初始化函数
    def __init__(self, *, alphafold2):
        super().__init__()
        # 初始化属性 alphafold2
        self.alphafold2 = alphafold2

        # 加载 ESM 模型和字母表
        model, alphabet = torch.hub.load(*ESM_MODEL_PATH) 
        batch_converter = alphabet.get_batch_converter()

        # 初始化 model, batch_converter, project_embed 属性
        self.model = model
        self.batch_converter = batch_converter
        self.project_embed = nn.Linear(ESM_EMBED_DIM, alphafold2.dim) if ESM_EMBED_DIM != alphafold2.dim else nn.Identity()

    # 前向传播函数
    def forward(self, seq, msa=None, **kwargs):
        # 获取 model, batch_converter, device 信息
        model, batch_converter, device = self.model, self.batch_converter, seq.device

        # 获取序列的 ESM 嵌入
        seq_embeds = get_esm_embedd(seq, model, batch_converter, device = device)
        seq_embeds = self.project_embed(seq_embeds)

        if msa is not None:
            # 将 msa 展平
            flat_msa = rearrange(msa, 'b m n -> (b m) n')
            # 获取 msa 的 ESM 嵌入
            msa_embeds = get_esm_embedd(flat_msa, model, batch_converter, device = device)
            msa_embeds = rearrange(msa_embeds, '(b m) n d -> b m n d')
            msa_embeds = self.project_embed(msa_embeds)
        else: 
            msa_embeds = None

        # 调用 alphafold2 模型进行预测
        return self.alphafold2(seq, msa, seq_embed = seq_embeds, msa_embed = msa_embeds, **kwargs)
```