# `.\lucidrains\progen\train.py`

```py
# 导入 load_dotenv 函数，用于加载环境变量
from dotenv import load_dotenv
# 调用 load_dotenv 函数加载环境变量

# 导入 click、humanize、Template、Path、tqdm、numpy 等模块
import click
import humanize
from jinja2 import Template
from pathlib import Path
import tqdm
import numpy as np

# 导入 toml 模块
import toml

# 导入 jax 相关模块和函数
import jax
from jax import nn, random, jit, tree_util, tree_map
from optax import adamw, clip_by_global_norm, chain, apply_updates, apply_every

# 导入 haiku 模块中的 PRNGSequence 类
from haiku import PRNGSequence

# 导入 progen_transformer 模块及其子模块
from progen_transformer import ProGen
from progen_transformer.data import decode_tokens, iterator_from_tfrecords_folder
from progen_transformer.utils import sample, get_loss_fn, set_hardware_rng_, confirm, exists
from progen_transformer.checkpoint import get_checkpoint_fns

# 导入 wandb 模块
import wandb

# 创建模板对象 sample_tmpl，用于生成 HTML 样式
sample_tmpl = Template("""<i>{{prime_str}}</i><br/><br/><div style="overflow-wrap: break-word;">{{sampled_str}}</div>""")

# 设置硬件随机数生成器
set_hardware_rng_(jax)

# 主函数定义，接收多个命令行参数
@click.command()
@click.option('--seed', default = 42)
@click.option('--batch_size', default = 4)
@click.option('--grad_accum_every', default = 4)
@click.option('--learning_rate', default = 2e-4)
@click.option('--weight_decay', default = 1e-3)
@click.option('--data_parallel', default = False, is_flag = True)
@click.option('--max_grad_norm', default = 0.5)
@click.option('--validate_every', default = 100)
@click.option('--sample_every', default = 500)
@click.option('--checkpoint_every', default = 1000)
@click.option('--checkpoint_path', default = './ckpts')
@click.option('--checkpoint_keep_n', default = 500)
@click.option('--config_path', default = './configs/model')
@click.option('--model_name', default = 'default')
@click.option('--prime_length', default = 25)
@click.option('--seq_len', default = 1024)
@click.option('--mixed_precision', default = False, is_flag = True)
@click.option('--data_path', default = './train_data')
@click.option('--wandb_off', default = False, is_flag = True)
@click.option('--wandb_project_name', default = 'progen-training')
@click.option('--new', default = False, is_flag = True)
def main(
    seed,
    batch_size,
    grad_accum_every,
    learning_rate,
    weight_decay,
    data_parallel,
    max_grad_norm,
    validate_every,
    sample_every,
    checkpoint_every,
    checkpoint_path,
    checkpoint_keep_n,
    config_path,
    model_name,
    prime_length,
    seq_len,
    mixed_precision,
    data_path,
    wandb_off,
    wandb_project_name,
    new
):
    # 准备文件夹

    # 获取重置、获取最新、保存检查点的函数
    reset_checkpoint, get_last_checkpoint, save_checkpoint = get_checkpoint_fns(checkpoint_path)

    # 如果设置了 new 参数，清除所有检查点并重新开始训练
    if new:
        if not confirm('are you sure you want to clear all your checkpoints and restart training?'):
            exit()
        reset_checkpoint()

    # 初始化所有状态，或从检查点加载

    # 获取最新的检查点
    last_checkpoint = get_last_checkpoint()

    # 如果最新的检查点不存在
    if not exists(last_checkpoint):
        # 获取模型配置文件路径
        config_folder_path = Path(config_path)
        config_path = config_folder_path / f'{model_name}.toml'
        # 检查模型配置文件是否存在
        assert config_path.exists(), f'path to your model config {str(config_path)} does not exist'
        # 加载模型参数
        model_kwargs = toml.loads(config_path.read_text())
    else:
        # 使用最新的检查点中的模型配置
        model_kwargs = last_checkpoint['model_config']

    # 设置模型和参数

    # 创建 ProGen 模型实例
    model = ProGen(**{
        **model_kwargs,
        'mixed_precision': mixed_precision
    })

    # 编译模型应用函数
    model_apply = jit(model.apply)
    # 创建随机数生成器
    rng = PRNGSequence(seed)
    # 获取损失函数
    loss_fn = get_loss_fn(model, data_parallel = data_parallel)

    # 优化器

    # 定义排除规范和偏置参数的函数
    exclude_norm_and_bias_params = lambda p: tree_map(lambda x: x.ndim > 1, p)

    # 构建优化器链
    optim = chain(
        clip_by_global_norm(max_grad_norm),
        adamw(learning_rate, weight_decay = weight_decay, mask = exclude_norm_and_bias_params),
        apply_every(grad_accum_every)
    )

    # 获取参数和优化器状态

    if exists(last_checkpoint):
        params = last_checkpoint['params']
        optim_state = last_checkpoint['optim_state']
        start_seq_index = last_checkpoint['next_seq_index']
    else:
        # 如果不是第一次训练，则创建一个全零数组作为模拟数据
        mock_data = np.zeros((model_kwargs['seq_len'],), dtype = np.uint8)
        # 使用模拟数据初始化模型参数
        params = model.init(next(rng), mock_data)
        # 使用初始化的参数初始化优化器状态
        optim_state = optim.init(params)
        # 设置起始序列索引为0
        start_seq_index = 0

    # 实验追踪器

    # 获取模型序列长度
    seq_len = model_kwargs['seq_len']
    # 计算模型参数的数量
    num_params = tree_util.tree_reduce(lambda acc, el: acc + el.size, params, 0)
    # 将参数数量转换为可读的格式
    num_params_readable = humanize.naturalsize(num_params)

    # 设置wandb配置中的参数数量
    wandb.config.num_params = num_params

    # 根据wandb_off参数决定是否禁用wandb
    wandb_kwargs = {'mode': 'disabled'} if wandb_off else {}

    # 如果存在上次的检查点信息，则恢复运行ID和恢复模式
    if exists(last_checkpoint) and exists(last_checkpoint['run_id']):
        run_id = last_checkpoint['run_id']
        wandb_kwargs = {**wandb_kwargs, 'id': run_id, 'resume': 'allow'}

    # 初始化wandb
    wandb.init(project = wandb_project_name, **wandb_kwargs)
    wandb_run_id = wandb.run.id if not wandb_off else None

    # 获取tf数据集

    # 从tfrecords文件夹中获取训练数据集
    total_train_seqs, get_train_dataset = iterator_from_tfrecords_folder(data_path, data_type = 'train')
    # 从tfrecords文件夹中获取验证数据集
    total_valid_seqs, get_valid_dataset = iterator_from_tfrecords_folder(data_path, data_type = 'valid',)

    # 断言训练数据集和验证数据集的序列数量大于0
    assert total_train_seqs > 0, 'no protein sequences found for training'
    assert total_valid_seqs > 0, 'no protein sequences found for validation'

    # 获取训练数据集和验证数据集
    train_dataset = get_train_dataset(
        seq_len = seq_len,
        batch_size = batch_size,
        skip = start_seq_index
    )

    valid_dataset = get_valid_dataset(
        seq_len = seq_len,
        batch_size = batch_size,
        loop = True
    )

    # 打印信息

    print(f'params: {num_params_readable}')
    print(f'sequence length: {seq_len}')
    print(f'num sequences: {total_train_seqs}')
    print(f'starting from sequence {start_seq_index}')

    # 训练

    # 计算有效批次大小
    effective_batch_size = batch_size * grad_accum_every
    # 计算序列索引范围
    seq_index_ranges = range(start_seq_index, total_train_seqs, effective_batch_size)    

    # 遍历序列索引范围
    for i, seq_index in tqdm.tqdm(enumerate(seq_index_ranges), mininterval = 10., desc = 'training', total = len(seq_index_ranges)):
        # 根据梯度累积次数进行训练
        for _ in range(grad_accum_every):
            data = next(train_dataset)

            # 计算损失和梯度
            loss, grads = loss_fn(params, next(rng), data)
            # 更新参数和优化器状态
            updates, optim_state = optim.update(grads, optim_state, params)
            params = apply_updates(params, updates)

        print(f'loss: {loss.item()}')
        wandb.log({'loss': loss.item()})

        if i % checkpoint_every == 0:
            # 保存检查点信息
            package = {
                'next_seq_index': seq_index + effective_batch_size,
                'params': params,
                'optim_state': optim_state,
                'model_config': model_kwargs,
                'run_id': wandb_run_id
            }

            save_checkpoint(package, checkpoint_keep_n)
            print(f"checkpoint to start at sequence index of {package['next_seq_index']}")

        if i % validate_every == 0:
            # 验证模型
            valid_data = next(valid_dataset)
            loss, _ = loss_fn(params, next(rng), valid_data)
            print(f'valid_loss: {loss.item()}')
            wandb.log({'valid_loss': loss.item()})

        if i % sample_every == 0:
            # 生成样本
            valid_data = next(valid_dataset)[0]
            prime = valid_data[:prime_length]
            prime_str = decode_tokens(prime)

            sampled = sample(rng, model_apply, params, prime, seq_len, top_k = 25)
            sampled_str = decode_tokens(sampled[prime_length:])

            print(prime_str, "\n", "*" * 40, "\n", sampled_str)
            wandb.log({'samples': wandb.Html(sample_tmpl.render(prime_str = prime_str, sampled_str = sampled_str))})
# 如果当前脚本被直接执行，则调用主函数
if __name__ == '__main__':
    main()
```