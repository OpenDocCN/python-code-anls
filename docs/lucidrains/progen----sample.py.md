# `.\lucidrains\progen\sample.py`

```py
# 导入 load_dotenv 函数，用于加载环境变量
from dotenv import load_dotenv
# 调用 load_dotenv 函数加载环境变量

# 导入 click 模块，用于创建命令行接口
import click
# 导入 humanize 模块，用于处理人类可读的数据格式

# 导入 jax 模块及其子模块
import jax
from jax import nn, random, jit, tree_util, numpy as np

# 导入 haiku 模块中的 PRNGSequence 类
from haiku import PRNGSequence

# 导入 progen_transformer 模块及其子模块
from progen_transformer import ProGen
from progen_transformer.data import decode_tokens, encode_tokens
from progen_transformer.utils import sample, set_hardware_rng_
from progen_transformer.checkpoint import get_checkpoint_fns

# 调用 set_hardware_rng_ 函数，加速随机数生成器

# 定义主函数
@click.command()
# 定义命令行参数
@click.option('--seed', default = 42)
@click.option('--checkpoint_path', default = './ckpts')
@click.option('--prime', default = '')
def main(
    seed,
    checkpoint_path,
    prime,
):
    # 准备文件夹

    # 获取最后一个检查点
    _, get_last_checkpoint, _ = get_checkpoint_fns(checkpoint_path)
    last_checkpoint = get_last_checkpoint()

    # 如果没有找到最后一个检查点，则退出程序
    if last_checkpoint is None:
        exit(f'no checkpoints found at {checkpoint_path}')

    # 获取参数和序列数
    params = last_checkpoint['params']
    num_seqs = max(last_checkpoint['next_seq_index'], 0)

    # 设置模型和参数
    model_kwargs = last_checkpoint['model_config']
    model = ProGen(**model_kwargs)
    model_apply = jit(model.apply)
    rng = PRNGSequence(seed)

    # 初始化所有状态，或从检查点加载

    seq_len = model_kwargs['seq_len']
    num_params = tree_util.tree_reduce(lambda acc, el: acc + el.size, params, 0)
    num_params_readable = humanize.naturalsize(num_params)

    # 打印参数、序列长度和训练序列数
    print(f'params: {num_params_readable}')
    print(f'sequence length: {seq_len}')
    print(f'trained for {num_seqs} sequences')

    # 使用 prime 进行采样
    prime_tokens = encode_tokens(prime)
    prime_length = len(prime_tokens) + 1
    prime_tensor = np.array(prime_tokens, dtype = np.uint16)

    sampled = sample(rng, jit(model_apply), params, prime_tensor, seq_len, top_k = 25, add_bos = True)
    sampled_str = decode_tokens(sampled[prime_length:])

    # 打印采样结果
    print("\n", prime, "\n", "*" * 40, "\n", sampled_str)

# 如果当前脚本被直接执行，则调用主函数
if __name__ == '__main__':
    main()
```