# `.\lucidrains\ddpm-proteins\cache.py`

```
# 导入必要的库
from tqdm import tqdm
import sidechainnet as scn
from ddpm_proteins.utils import get_msa_attention_embeddings, get_msa_transformer

# 加载 sidechainnet 数据集
data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = 1,
    dynamic_batching = False
)

# 设置常量
LENGTH_THRES = 256

# 定义获取 MSAs 的函数，根据你的设置填写
def fetch_msas_fn(aa_str):
    """
    给定一个氨基酸序列作为字符串
    填写一个返回 MSAs（作为字符串列表）的函数
    （默认情况下，它将返回空列表，只有主要序列被馈送到 MSA Transformer）
    """
    return []

# 缓存循环

# 获取 MSA Transformer 模型和批处理转换器
model, batch_converter = get_msa_transformer()

# 遍历训练数据集中的批次
for batch in tqdm(data['train']):
    # 如果序列长度大于阈值，则跳过当前批次
    if batch.seqs.shape[1] > LENGTH_THRES:
        continue

    # 获取批次中的蛋白质 ID 和序列
    pids = batch.pids
    seqs = batch.seqs.argmax(dim = -1)

    # 获取 MSA 注意力嵌入
    _ = get_msa_attention_embeddings(
        model,
        batch_converter,
        seqs,
        batch.pids,
        fetch_msas_fn
    )

# 输出缓存完成信息
print('caching complete')
```