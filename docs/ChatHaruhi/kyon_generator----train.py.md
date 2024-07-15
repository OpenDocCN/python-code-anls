# `.\Chat-Haruhi-Suzumiya\kyon_generator\train.py`

```py
'''
FILE_NAME: Train.py
GLM2-LoRA Edition
Edited by 冷子昂，睡觉鱼, Aug 14 2023
'''

# 导入所需的依赖库
import json  # 导入处理 JSON 格式数据的库
from torch.utils.data import Dataset, DataLoader  # 导入 PyTorch 中用于定义数据集和数据加载器的类
import os  # 导入操作系统相关功能的库
import jsonlines  # 导入处理 JSONlines 格式数据的库
from torch.utils.data import ConcatDataset  # 导入用于合并数据集的类
from transformers import AutoTokenizer, AutoModel  # 导入 Hugging Face Transformers 库中的模型和分词器
from datasets import load_dataset, Dataset  # 导入 Hugging Face datasets 库中的数据集加载函数和类
import torch  # 导入 PyTorch 深度学习库
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
from peft import LoraConfig, get_peft_model  # 导入 peft 库中的配置和模型加载函数
from transformers import Trainer, TrainingArguments  # 导入 Hugging Face Transformers 中的训练器和训练参数
from huggingface_hub import login  # 导入 Hugging Face Hub 的登录函数
from dataset import CharacterDataset, read_jsonl_file, collate_fn  # 导入自定义的数据集类和文件读取函数

# 设置 Hugging Face Hub 的登录 token
# HF_TOKEN = 'hflcAlYNF'
# HF_TOKEN = "nPhmtMVuXy"
# login(token=HF_TOKEN)

# 从预训练模型加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()

'''
这里放你的dataloader
'''
# 设置文件路径和字符数据路径
# jsonl_file_path = '/Users/pufferfish/Downloads/real_train_data/'
# character_path = "/Users/pufferfish/Chat-Haruhi-Suzumiya/characters/"

# 定义需要处理的 JSONL 文件列表
# file_names = ['xiaofeng_test_output_dialogue.jsonl', 'baizhantang_test_output_dialogue.jsonl', 'wangduoyu_test_output_dialogue.jsonl', 'guofurong_test_output_dialogue.jsonl', 'weixiaobao_test_output_dialogue.jsonl', 'haruhi_synthesis_dialogue.jsonl', 'murongfu_test_output_dialogue.jsonl', 'McGonagall_test_output_dialogue.jsonl', 'Ron_test_output_dialogue.jsonl', 'Sheldon_test_output_dialogue.jsonl', 'yuqian_test_output_dialogue.jsonl', 'duanyu_test_output_dialogue.jsonl', 'xuzhu_test_output_dialogue.jsonl', 'jiumozhi_test_output_dialogue.jsonl', 'liyunlong_synthesis_dialogue.jsonl', 'Malfoy_test_output_dialogue.jsonl', 'tongxiangyu_test_output_dialogue.jsonl', 'ayaka_test_output_dialogue.jsonl', 'Raj_test_output_dialogue.jsonl', 'Harry_test_output_dialogue.jsonl', 'Snape_test_output_dialogue.jsonl', 'Penny_test_output_dialogue.jsonl', 'zhongli_test_output_dialogue.jsonl', 'tangshiye_test_output_dialogue.jsonl', 'Luna_test_output_dialogue.jsonl', 'hutao_test_output_dialogue.jsonl', 'Dumbledore_test_output_dialogue.jsonl', 'Hermione_test_output_dialogue.jsonl', 'qiaofeng_test_output_dialogue.jsonl', 'wangyuyan_test_output_dialogue.jsonl', 'wanderer_test_output_dialogue.jsonl', 'raidenShogun_test_output_dialogue.jsonl']

# 初始化数据集列表
# all_datasets = []
# 遍历每个文件名，构建对应的 CharacterDataset 并加入数据集列表
# for file_name in file_names:
#     character_name = file_name.split("_")[0]
#     character = os.path.join(character_path, character_name)
#     jsonl_file = os.path.join(jsonl_file_path, file_name)
#     jsonl_data = read_jsonl_file(jsonl_file)
#     c = CharacterDataset(jsonl_data, character, 8, 2000)
#     all_datasets.append(c)

# 合并所有数据集为一个 ConcatDataset
# combined_dataset = ConcatDataset(all_datasets)

# 设置批量大小
# batch_size = 1
# 加载数据集 'silk-road/Chat_Suzumiya_Fusion'
dataset = load_dataset('silk-road/Chat_Suzumiya_Fusion')
print(dataset)

# 定义预处理对话函数，将上下文和目标文本转换为模型可接受的输入格式
def preprocess_dialogue(example):
    prompt = example["context"]
    target = example["target"]
    # 使用分词器对上下文和目标文本进行编码，添加特殊标记并进行截断
    prompt_ids = tokenizer.encode(prompt, truncation=True, add_special_tokens=True)
    target_ids = tokenizer.encode(target, truncation=True, add_special_tokens=False)
    input_ids = prompt_ids + target_ids
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

# 对训练数据集应用预处理函数，转换为模型输入格式
model_inputs = train_dataset.map(preprocess_dialogue)

# 冻结模型中的参数，稍后训练适配器
for param in model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
        # 将参数转换为 float32 类型以增强稳定性，特别是层归一化等参数
        param.data = param.data.to(torch.float32)

# 开启模型的梯度检查点，减少存储的激活数量
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.is_parallelizable = True
model.model_parallel = True

# 配置 Lora 模型的参数
config = LoraConfig(
    r=16,
    lora_alpha=32,
    inference_mode=False,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# 根据配置获取 PEFT 模型
model = get_peft_model(model, config)

# 定义数据集收集器函数，将特征列表转换为模型所需的输入和标签格式
def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    # 按序列长度降序排序特征，生成输入和标签
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1):] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

# 自定义 Trainer 类，重写 compute_loss 方法以计算模型损失
class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

# 训练参数配置
training_args = TrainingArguments(
    num_train_epochs=2,
    max_steps=-1,
    evaluation_strategy="no",
    gradient_accumulation_steps=1,
    group_by_length=False,
    save_strategy="steps",
    save_steps=500,
    output_dir='output',
    remove_unused_columns=False,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-4,
    fp16=True,
    seed=2023,
)
    # 设定数据种子为 2023
    data_seed=2023
)

# 创建一个修改后的训练器对象，用于训练模型
trainer = ModifiedTrainer(
    model=model,  # 指定要训练的模型
    train_dataset=model_inputs['train'],  # 指定训练数据集
    #eval_dataset=model_inputs['test'],  # 评估数据集（目前被注释掉，不参与训练）
    args=training_args,  # 指定训练参数
    data_collator=data_collator,  # 数据收集器，用于处理批次数据
)
# 开始训练模型
trainer.train()

# 将模型推送到模型中心（已被注释掉，不执行这行代码）
# model.push_to_hub("Jyshen/Chat_Suzumiya_GLM2LoRA", use_auth_token=True)
```