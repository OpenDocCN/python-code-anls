# `.\VisualGLM-6B\finetune_visualglm.py`

```
# 导入必要的库
import os
import torch
import argparse

# 从 sat 模块中导入 mpu, get_args, get_tokenizer 函数
from sat import mpu, get_args, get_tokenizer
# 从 sat.training.deepspeed_training 模块中导入 training_main 函数
from sat.training.deepspeed_training import training_main
# 从 model 模块中导入 VisualGLMModel 类
from model import VisualGLMModel
# 从 sat.model.finetune 模块中导入 PTuningV2Mixin 类
from sat.model.finetune import PTuningV2Mixin
# 从 sat.model.finetune.lora2 模块中导入 LoraMixin 类

class FineTuneVisualGLMModel(VisualGLMModel):
    # 初始化 FineTuneVisualGLMModel 类
    def __init__(self, args, transformer=None, parallel_output=True, **kw_args):
        # 调用父类 VisualGLMModel 的初始化方法
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, **kw_args)
        # 如果 args 中包含 use_ptuning 标志，则添加 PTuningV2Mixin 混合类
        if args.use_ptuning:
            self.add_mixin("ptuning", PTuningV2Mixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.pre_seq_len))
        # 如果 args 中包含 use_lora 标志，则添加 LoraMixin 混合类
        if args.use_lora:
            self.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range), reinit=True)
            # self.get_mixin("eva").model.glm_proj = replace_linear_with_lora(self.get_mixin("eva").model.glm_proj, LoraLinear, args.lora_rank)
        # 如果 args 中包含 use_qlora 标志，则添加带有 qlora 标志的 LoraMixin 混合类
        elif args.use_qlora:
            self.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range, qlora=True), reinit=True)
        # 将 args 存储在对象中
        self.args = args
        
    # 类方法，用于添加模型特定的参数
    @classmethod
    def add_model_specific_args(cls, parser):
        # 创建一个参数组 'VisualGLM-finetune'，包含 finetune 配置
        group = parser.add_argument_group('VisualGLM-finetune', 'VisualGLM finetune Configurations')
        # 添加参数 pre_seq_len，默认值为 8
        group.add_argument('--pre_seq_len', type=int, default=8)
        # 添加参数 lora_rank，默认值为 10
        group.add_argument('--lora_rank', type=int, default=10)
        # 添加参数 use_ptuning，用于启用 PTuning
        group.add_argument('--use_ptuning', action="store_true")
        # 添加参数 use_lora，用于启用 Lora
        group.add_argument('--use_lora', action="store_true")
        # 添加参数 use_qlora，用于启用 QLora
        group.add_argument('--use_qlora', action="store_true")
        # 添加参数 layer_range，用于指定层范围
        group.add_argument('--layer_range', nargs='+', type=int, default=None)
        # 调用父类的 add_model_specific_args 方法
        return super().add_model_specific_args(parser)
    # 定义一个方法用于禁用不可训练的参数
    def disable_untrainable_params(self):
        # 初始化一个空列表用于存储需要启用的参数
        enable = []
        # 如果使用参数调整（ptuning），则添加相关参数到列表中
        if self.args.use_ptuning:
            enable.extend(['ptuning'])
        # 如果使用 lora 或 qlora，则添加相关参数到列表中
        if self.args.use_lora or self.args.use_qlora:
            enable.extend(['matrix_A', 'matrix_B'])
        # 遍历模型的所有参数及其名称
        for n, p in self.named_parameters():
            # 初始化一个标志位
            flag = False
            # 遍历需要启用的参数列表
            for e in enable:
                # 如果参数名称中包含需要启用的参数名，则将标志位设为 True
                if e.lower() in n.lower():
                    flag = True
                    break
            # 如果标志位为 False，则将该参数设置为不可训练
            if not flag:
                p.requires_grad_(False)
            else:
                # 否则打印参数名称
                print(n)
# 定义一个函数，用于获取数据批次
def get_batch(data_iterator, args, timers):
    # 定义数据项和它们的类型
    keys = ['input_ids', 'labels']
    datatype = torch.int64

    # 广播数据
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    data_i = mpu.broadcast_data(['image'], data, torch.float32)
    # 解包数据
    tokens = data_b['input_ids'].long()
    labels = data_b['labels'].long()
    img = data_i['image']
    if args.fp16:
        img = img.half()
    
    return tokens, labels, img, data['pre_image']


from torch.nn import CrossEntropyLoss

def forward_step(data_iterator, model, args, timers):
    """前向步骤。"""

    # 获取批次数据
    timers('batch generator').start()
    tokens, labels, image, pre_image = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    logits = model(input_ids=tokens, image=image, pre_image=pre_image)[0]
    dtype = logits.dtype
    lm_logits = logits.to(torch.float32)

    # 移位，使得 tokens < n 预测 n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # 展平 tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    lm_logits = lm_logits.to(dtype)
    loss = loss.to(dtype)
    return loss, {'loss': loss}


from model.blip2 import BlipImageEvalProcessor
from torch.utils.data import Dataset
import json
from PIL import Image

class FewShotDataset(Dataset):
    # 初始化函数，接受路径、处理器、分词器和参数作为输入
    def __init__(self, path, processor, tokenizer, args):
        # 计算最大序列长度
        max_seq_length = args.max_source_length + args.max_target_length
        # 打开文件并加载 JSON 数据
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 初始化空列表用于存储图像、输入序列和标签
        self.images = []
        self.input_ids = []
        self.labels = []
        # 遍历数据中的每个项目
        for item in data:
            # 处理图像
            image = processor(Image.open(item['img']).convert('RGB'))
            # 编码特殊标记
            input0 = tokenizer.encode("<img>", add_special_tokens=False)
            input1 = [tokenizer.pad_token_id] * args.image_length
            input2 = tokenizer.encode("</img>问："+item['prompt']+"\n答：", add_special_tokens=False)
            # 合并编码后的输入序列
            a_ids = sum([input0, input1, input2], [])
            b_ids = tokenizer.encode(text=item['label'], add_special_tokens=False)
            # 截断序列长度
            if len(a_ids) > args.max_source_length - 1:
                a_ids = a_ids[: args.max_source_length - 1]
            if len(b_ids) > args.max_target_length - 2:
                b_ids = b_ids[: args.max_target_length - 2]
            pre_image = len(input0)
            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
            # 计算上下文长度和掩码位置
            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position+1:]
            # 填充序列至最大长度
            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            # 如果忽略填充标记，则将填充标记替换为-100
            if args.ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            # 将处理后的图像、输入序列和标签添加到相应列表中
            self.images.append(image)
            self.input_ids.append(input_ids)
            self.labels.append(labels)
        # 保存处理前的图像长度
        self.pre_image = pre_image

    # 返回数据集的长度
    def __len__(self):
        return len(self.images)
    # 定义一个特殊方法，用于获取对象的索引值对应的数据
    def __getitem__(self, idx):
        # 返回一个包含图像、输入ID、标签和前一个图像的字典
        return {
            "image": self.images[idx],  # 获取索引值对应的图像数据
            "input_ids": self.input_ids[idx],  # 获取索引值对应的输入ID数据
            "labels": self.labels[idx],  # 获取索引值对应的标签数据
            "pre_image": self.pre_image  # 获取前一个图像数据
        }
# 创建数据集函数，接受路径和参数作为输入
def create_dataset_function(path, args):
    # 获取分词器
    tokenizer = get_tokenizer(args)
    # 创建图像处理器
    image_processor = BlipImageEvalProcessor(224)

    # 创建少样本数据集
    dataset = FewShotDataset(path, image_processor, tokenizer, args)
    # 返回数据集
    return dataset


if __name__ == '__main__':
    # 创建参数解析器
    py_parser = argparse.ArgumentParser(add_help=False)
    # 添加最大源长度参数
    py_parser.add_argument('--max_source_length', type=int)
    # 添加最大目标长度参数
    py_parser.add_argument('--max_target_length', type=int)
    # 添加是否忽略填充标记用于损失的参数
    py_parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True)
    # 添加源前缀参数
    py_parser.add_argument('--source_prefix', type=str, default="")
    # 添加模型特定参数
    py_parser = FineTuneVisualGLMModel.add_model_specific_args(py_parser)
    # 解析已知参数和参数列表
    known, args_list = py_parser.parse_known_args()
    # 获取参数
    args = get_args(args_list)
    # 将已知参数和参数合并
    args = argparse.Namespace(**vars(args), **vars(known))
    # 设置设备为 CPU
    args.device = 'cpu'

    # 模型类型为 'visualglm-6b'，从预训练模型中加载模型和参数
    model_type = 'visualglm-6b'
    model, args = FineTuneVisualGLMModel.from_pretrained(model_type, args)
    # 如果 CUDA 可用，将模型移动到 CUDA
    if torch.cuda.is_available():
        model = model.to('cuda')
    # 获取分词器
    tokenizer = get_tokenizer(args)
    # 如果忽略填充标记用于损失，则标签填充标记为 -100，否则为分词器的填充标记 ID
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # 数据收集器函数，对每个示例进行处理
    def data_collator(examples):
        for example in examples:
            example['input_ids'] = torch.tensor(example['input_ids'], dtype=torch.long)
            example['labels'] = torch.tensor(example['labels'], dtype=torch.long)
        # 返回处理后的数据
        ret = {
            'input_ids': torch.stack([example['input_ids'] for example in examples]),
            'labels': torch.stack([example['labels'] for example in examples]),
            'image': torch.stack([example['image'] for example in examples]),
            'pre_image': example['pre_image']
        }
        return ret
    # 进行训练
    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=create_dataset_function, collate_fn=data_collator)
```