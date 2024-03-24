# `.\lucidrains\tf-bind-transformer\precache_proteins.py`

```
# 导入需要的库
import click  # 用于创建命令行接口
from tqdm import tqdm  # 用于显示进度条
from pathlib import Path  # 用于处理文件路径
from Bio import SeqIO  # 用于处理生物信息学数据
from tf_bind_transformer.protein_utils import get_protein_embedder  # 从自定义模块中导入函数

# 创建命令行接口
@click.command()
@click.option('--model-name', default = 'protalbert', help = 'Protein model name')  # 添加命令行参数，指定蛋白质模型名称
@click.option('--fasta-folder', help = 'Path to factor fastas', required = True)  # 添加命令行参数，指定FASTA文件夹路径
def cache_embeddings(
    model_name,  # 指定蛋白质模型名称
    fasta_folder  # 指定FASTA文件夹路径
):
    # 获取指定蛋白质模型的函数
    fn = get_protein_embedder(model_name)['fn']
    # 获取FASTA文件夹下所有的FASTA文件路径
    fastas = [*Path(fasta_folder).glob('**/*.fasta')]

    # 断言确保至少找到一个FASTA文件
    assert len(fastas) > 0, f'no fasta files found at {fasta_folder}'

    # 遍历所有FASTA文件并处理
    for fasta in tqdm(fastas):
        # 读取FASTA文件中的序列数据
        seq = SeqIO.read(fasta, 'fasta')
        # 将序列数据转换为字符串
        seq_str = str(seq.seq)
        # 使用指定的函数处理序列数据
        fn([seq_str], device = 'cpu')

# 如果作为脚本直接运行，则调用cache_embeddings函数
if __name__ == '__main__':
    cache_embeddings()
```