# `.\lucidrains\tf-bind-transformer\tf_bind_transformer\training_utils.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 tf_bind_transformer.optimizer 模块中导入 get_optimizer 函数
from tf_bind_transformer.optimizer import get_optimizer
# 从 tf_bind_transformer.data 模块中导入 read_bed, collate_dl_outputs, get_dataloader, remap_df_add_experiment_target_cell 函数
from tf_bind_transformer.data import read_bed, collate_dl_outputs, get_dataloader, remap_df_add_experiment_target_cell
# 从 tf_bind_transformer.data 模块中导入 RemapAllPeakDataset, NegativePeakDataset, ScopedNegativePeakDataset 类

# 定义 exists 函数，用于判断值是否存在
def exists(val):
    return val is not None

# 定义 default 函数，用于返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义 accum_log 函数，用于记录和累积梯度步骤中的值
def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# 定义简单的 Trainer 类
class Trainer(nn.Module):
    def __init__(
        self,
        model,
        *,
        remap_bed_file,
        negative_bed_file,
        factor_fasta_folder,
        fasta_file,
        train_chromosome_ids,
        valid_chromosome_ids,
        batch_size,
        context_length,
        lr = 3e-4,
        wd = 0.1,
        validate_every = 250,
        grad_clip_norm = None,
        grad_accum_every = 1,
        held_out_targets = [],
        held_out_cell_types = [],
        exclude_targets = [],
        exclude_cell_types = [],
        shuffle = False,
        train_sample_frac = 1.,
        valid_sample_frac = 1.,
        remap_sample_frac = 1.,
        shift_aug_range = (-2, 2),
        rc_aug = False,
        experiments_json_path = None,
        read_value_aux_loss = False,
        checkpoint_filename = './checkpoint.pt',
        include_scoped_negs = False,
        scoped_negs_remap_bed_path = None,
        scoped_negs_path = None,
        scoped_negs_exts = '.bed.bool.npy',
        include_biotypes_metadata_in_context = False,
        biotypes_metadata_path = None,
        include_biotypes_metadata_columns = ['germ_layer', 'cellline_cat'],
        biotypes_metadata_delimiter = ' | ',
        balance_sampling_by_target = True,
        valid_balance_sampling_by_target = None,
    # 定义 forward 方法，用于前向传播
    def forward(
        self,
        finetune_enformer_ln_only = True,
        **kwargs
        ):
            # 获取当前的梯度累积步数
            grad_accum_every = self.grad_accum_every
            # 获取当前步数
            curr_step = int(self.steps.item())
            # 设置模型为训练模式
            self.model.train()

            # 初始化日志字典
            log = {}

            # 循环执行梯度累积步数次
            for _ in range(self.grad_accum_every):
                # 从数据加载器中获取数据
                dl_outputs = [next(self.dl), next(self.neg_dl)]

                # 如果包含了作用域负样本，则继续获取数据
                if self.include_scoped_negs:
                    dl_outputs.append(next(self.scoped_neg_dl))

                # 将数据整理成模型所需的格式
                seq, tf_aa, contextual_texts, peaks_nr, read_value, binary_target = collate_dl_outputs(*dl_outputs)
                seq, binary_target, read_value, peaks_nr = seq.cuda(), binary_target.cuda(), read_value.cuda(), peaks_nr.cuda()

                # 计算模型的损失
                loss, aux_loss = self.model(
                    seq,
                    target = binary_target,
                    aa = tf_aa,
                    contextual_free_text = contextual_texts,
                    finetune_enformer_ln_only = finetune_enformer_ln_only,
                    read_value = read_value,
                    peaks_nr = peaks_nr,
                    **kwargs
                )

                # 计算总损失
                total_loss = self.model.combine_losses(loss, aux_loss)

                # 更新日志
                log = accum_log(log, {
                    'loss': loss.item() / grad_accum_every,
                    'aux_loss': aux_loss.item() / grad_accum_every,
                    'total_loss': total_loss.item() / grad_accum_every
                })

                # 反向传播
                (total_loss / self.grad_accum_every).backward()

            # 打印当前步数的总损失
            print(f'{curr_step} loss: {log["total_loss"]}')

            # 如果设置了梯度裁剪阈值，则进行梯度裁剪
            if exists(self.grad_clip_norm):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            # 更新优化器
            self.optim.step()
            self.optim.zero_grad()

            # 每隔一定步数进行验证
            if (curr_step % self.validate_every) == 0:
                # 设置模型为评估模式
                self.model.eval()

                # 循环执行梯度累积步数次验证
                for _ in range(self.grad_accum_every):
                    # 从验证数据加载器中获取数据
                    seq, tf_aa, contextual_texts, peaks_nr, read_value, binary_target = collate_dl_outputs(next(self.valid_dl), next(self.valid_neg_dl))
                    seq, binary_target = seq.cuda(), binary_target.cuda()

                    # 获取验证集的预测结果
                    valid_logits = self.model(
                        seq,
                        aa = tf_aa,
                        contextual_free_text = contextual_texts,
                    )

                    # 计算验证集的损失和准确率
                    valid_loss = self.model.loss_fn(valid_logits, binary_target.float())
                    valid_accuracy = ((valid_logits.sigmoid() > 0.5).int() == binary_target).sum() / (binary_target.numel())

                    # 更新日志
                    log = accum_log(log, {
                        'valid_loss': valid_loss.item() / grad_accum_every,
                        'valid_accuracy': valid_accuracy.item() / grad_accum_every
                    })

                # 打印验证集的损失和准确率
                print(f'{curr_step} valid loss: {log["valid_loss"]}')
                print(f'{curr_step} valid accuracy: {log["valid_accuracy"]}')

                # 如果当前步数大于0，则保存模型参数
                if curr_step > 0:
                    torch.save(self.model.state_dict(), self.checkpoint_filename)

            # 更新步数
            self.steps += 1
            # 返回日志
            return log
```