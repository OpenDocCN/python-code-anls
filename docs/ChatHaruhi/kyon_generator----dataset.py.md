# `.\Chat-Haruhi-Suzumiya\kyon_generator\dataset.py`

```py
# 导入 PyTorch 库
import torch
# 导入处理 JSON 数据的库
import json
# 导入 Dataset 类，用于创建自定义数据集
from torch.utils.data import Dataset
# 导入操作系统相关功能的库
import os
# 导入处理 JSONlines 文件的库
import jsonlines
# 导入 ConcatDataset 类，用于合并多个数据集
from torch.utils.data import ConcatDataset
# 导入 DataLoader 类，用于批量加载数据
from torch.utils.data import DataLoader
# 导入 Huggingface Hub 的登录函数
from huggingface_hub import login

# 定义一个函数，用于读取并解析 JSONlines 文件的内容
def read_jsonl_file(file_path):
    data = []
    # 打开文件，逐行读取数据
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去除每行两端的空白字符
            if line:
                try:
                    json_data = json.loads(line)  # 尝试解析 JSON 格式的数据
                    data.append(json_data)  # 将解析后的数据添加到列表中
                except json.JSONDecodeError:
                    # 如果解析失败，跳过该行数据
                    continue
    return data

# 定义一个函数，用于处理数据集中的一个 batch 的数据
def collate_fn(batch):
    # 提取每个样本的输入部分并组成列表
    inputs = [sample["input"] for sample in batch]
    # 提取每个样本的答案部分并组成列表
    targets = [sample["answer"] for sample in batch]

    # 可选：将输入和答案转换为 PyTorch 张量
    # batch_inputs = torch.stack(inputs)
    # batch_targets = torch.stack(targets)

    # 返回输入和答案列表
    return inputs, targets

# 定义一个自定义的 Dataset 类，用于加载角色相关的数据
class CharacterDataset(Dataset):
    def __init__(self, json_data, character_path, memory_number, memory_length):
        # 初始化方法，设置数据、文件路径、记忆数量和记忆长度等属性
        self.data = json_data  # 角色相关的 JSON 数据
        self.character_path = character_path  # 角色文件路径
        self.memory_number = memory_number  # 记忆数量
        self.memory_path = "jsonl/title_text_embed.jsonl"  # 记忆文件路径
        self.system_prompt_name = "system_prompt.txt"  # 系统提示文件名
        # 获取系统提示信息
        self.system_prompt = self.getSystemPrompt()
        # 读取 JSONlines 文件并将其转换为张量表示的记忆嵌入和文本
        self.memory_embed, self.memory_text = self.read_jsonl_and_convert_to_tensor(
            os.path.join(self.character_path, self.memory_path))
        self.memory_length = memory_length  # 记忆长度，可能是一个限制参数

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.data)
    # 定义类方法，实现索引操作，获取指定索引处的样本数据
    def __getitem__(self, idx):
        # 从数据集中获取指定索引的样本
        sample = self.data[idx]
        # 从样本中获取嵌入向量，并转换为 Torch 张量
        query_embed = torch.tensor(sample["embedding"])
        # 调用类中的方法，根据查询嵌入向量获取存储的记忆
        top_k = self.getMemory(query_embed)
        
        # 从存储的记忆中提取信息，这里采用特定的分隔符进行拆分提取
        retrieved_memory = [self.memory_text[i].split("｜｜｜")[1] for i in top_k]
        # 将提取的记忆信息按照指定长度进行连接
        retrieved_memory = self.join_with_limit(retrieved_memory, self.memory_length)
        
        # 将历史对话列表转换为特定格式的字符串
        chat_history = sample["chat_history"]
        chat_history = '###'.join(chat_history)
        chat_history += '###'
        
        # 构造输入文本，包含系统提示、提取的记忆信息、历史对话和查询信息
        query = sample["query"]
        input = self.system_prompt + "###" + retrieved_memory + chat_history + query
        
        # 组装返回的数据字典，包含查询、系统提示、提取的记忆、历史对话、答案、嵌入向量、来源和输入文本
        data = {
            "query": sample["query"],
            "system_prompt": self.system_prompt,
            "retrieved_memory": retrieved_memory,
            "chat_history": chat_history,
            "answer": sample["answer"],
            "embedding": query_embed,
            "source": sample["source"],
            "input": input
        }
        return data

    # 类方法，将给定列表按照最大长度连接为字符串，使用指定的分隔符
    def join_with_limit(self, items, max_length, separator="###"):
        result = ""
        for item in items:
            # 如果加入当前元素不会导致超过上限，就将其添加到结果字符串中
            if len(result) + len(item) + len(separator) <= max_length:
                if result:
                    result += separator
                result += item
            else:
                break  # 如果已经超过上限，就停止添加新元素
        return result

    # 类方法，读取指定路径文件的内容并返回
    def read_file_content(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    # 类方法，读取 JSONL 文件并转换成 Torch 张量和键列表
    def read_jsonl_and_convert_to_tensor(self, file_path):
        embed_list = []
        data_list = []
        with jsonlines.open(file_path) as reader:
            for item in reader:
                embed_list.append(next(iter(item.values())))
                data_list.append(next(iter(item.keys())))
        data_tensor = torch.tensor(embed_list)
        return data_tensor, data_list

    # 类方法，获取系统提示内容，从文件中读取
    def getSystemPrompt(self):
        file_path = os.path.join(self.character_path, self.system_prompt_name)
        file_content = self.read_file_content(file_path)
        return file_content

    # 类方法，根据输入的向量计算与存储的记忆之间的相似度，并返回最相似的索引
    def getMemory(self, vector):
        # 计算存储的记忆与给定向量的相似度
        similarity = torch.matmul(self.memory_embed, vector)
        
        # 找出相似度最高的几个元素的索引
        top_indices = torch.topk(similarity, self.memory_number).indices

        return top_indices
        # print("Top indices:", top_indices)
# 设置 JSONL 文件夹路径
jsonl_file_path = '/Users/pufferfish/Downloads/real_train_data/'
# 设置角色文件夹路径
character_path = "/Users/pufferfish/Chat-Haruhi-Suzumiya/characters/"
# 定义文件名列表
file_names = [
    'xiaofeng_test_output_dialogue.jsonl', 
    'baizhantang_test_output_dialogue.jsonl', 
    'wangduoyu_test_output_dialogue.jsonl', 
    'guofurong_test_output_dialogue.jsonl', 
    'weixiaobao_test_output_dialogue.jsonl', 
    'haruhi_synthesis_dialogue.jsonl', 
    'murongfu_test_output_dialogue.jsonl', 
    'McGonagall_test_output_dialogue.jsonl', 
    'Ron_test_output_dialogue.jsonl', 
    'Sheldon_test_output_dialogue.jsonl', 
    'yuqian_test_output_dialogue.jsonl', 
    'duanyu_test_output_dialogue.jsonl', 
    'xuzhu_test_output_dialogue.jsonl', 
    'jiumozhi_test_output_dialogue.jsonl', 
    'liyunlong_synthesis_dialogue.jsonl', 
    'Malfoy_test_output_dialogue.jsonl', 
    'tongxiangyu_test_output_dialogue.jsonl', 
    'ayaka_test_output_dialogue.jsonl', 
    'Raj_test_output_dialogue.jsonl', 
    'Harry_test_output_dialogue.jsonl', 
    'Snape_test_output_dialogue.jsonl', 
    'Penny_test_output_dialogue.jsonl', 
    'zhongli_test_output_dialogue.jsonl', 
    'tangshiye_test_output_dialogue.jsonl', 
    'Luna_test_output_dialogue.jsonl', 
    'hutao_test_output_dialogue.jsonl', 
    'Dumbledore_test_output_dialogue.jsonl', 
    'Hermione_test_output_dialogue.jsonl', 
    'qiaofeng_test_output_dialogue.jsonl', 
    'wangyuyan_test_output_dialogue.jsonl', 
    'wanderer_test_output_dialogue.jsonl', 
    'raidenShogun_test_output_dialogue.jsonl'
]

# 初始化数据集列表
all_datasets = []

# 遍历每个文件名
for file_name in file_names:
    # 从文件名中提取角色名
    character_name = file_name.split("_")[0]
    # 构建角色文件路径
    character = os.path.join(character_path, character_name)
    # 构建 JSONL 文件的完整路径
    jsonl_file = os.path.join(jsonl_file_path, file_name)
    # 读取 JSONL 文件内容
    jsonl_data = read_jsonl_file(jsonl_file)
    # 创建角色数据集对象
    c = CharacterDataset(jsonl_data, character, 8, 2000)
    # 将数据集对象添加到总数据集列表中
    all_datasets.append(c)
    # 打印角色名
    # 打印系统提示信息
    # 打印第一个查询
    # all_datasets.append(c)

# 将所有数据集合并成一个数据集
combined_dataset = ConcatDataset(all_datasets)

# 打印第一个查询
print(combined_dataset[0]["query"])
# 打印第一个检索到的记忆
print(combined_dataset[0]["retrieved_memory"])
# 打印第3000个查询
print(combined_dataset[3000]["query"])
# 打印第3000个检索到的记忆
print(combined_dataset[3000]["retrieved_memory"])

# 设置文件路径
path = '/Users/pufferfish/Downloads/real_train_data/yuqian_test_output_dialogue.jsonl'
# 逐行读取文件内容并尝试解析为 JSON
with open(path, "r") as file:
    for line in file:
        try:
            json.loads(line)
        except:
            # 打印解析失败的行
            print(line)

# 获取目录下所有文件名
file_names = os.listdir("/Users/pufferfish/Downloads/training_data_b/")
# 设置文件路径
path = "/Users/pufferfish/Downloads/training_data_b/"
# 遍历所有文件名
for file_name in file_names:
    # 打开 JSON 文件
    with open(path+file_name, 'r') as json_file:
        # 遍历文件的每一行
        for line in json_file:
            # 在每个 '}' 后插入换行符
            new_line = line.replace("}", "}\n")
            # 按换行符分割为新的行列表
            new_line_list = new_line.split("\n")
            # 打开输出文件，将处理后的行写入
            with open('/Users/pufferfish/Downloads/real_train_data_b/'+file_name, "w") as output_file:
                output_file.write('\n'.join(new_line_list))

# 定义角色字典映射
dic = {
    "tangshiye":['汤师爷'],
    "murongfu":['慕容复'],
    "liyunlong":['李云龙'],
    "Luna":['Luna'],
    "wangduoyu":['王多鱼'],
    "Ron":['Ron', '罗恩'],
}
# 定义一个字典，将角色名映射为包含其各种别名的列表
dic = {
    "jiumozhi":['鸠摩智'],
    "Snape":['Snape'],
    "haruhi":['春日', '凉宫春日', '涼宮ハルヒ', '涼宮'],
    "Malfoy":['Malfoy'],
    "xuzhu":['虚竹'],
    "xiaofeng":['萧峰'],
    "duanyu":['段誉'],
    "Hermione":['Hermione', '赫敏'],
    "Dumbledore":['Dumbledore', '邓布利多'],
    "wangyuyan":['王语嫣'],
    "Harry":['Harry', '哈利'],
    "McGonagall":['McGonagall', 'Professor McGonagall'],
    "baizhantang":['白展堂', '展堂'],
    "tongxiangyu":['佟湘玉'],
    "guofurong":['郭芙蓉'],
    "wanderer":['旅行者', '流浪者'],
    "zhongli":['钟离'],
    "hutao":['胡桃'],
    "Sheldon":['Sheldon'],
    "Raj":['Raj'],
    "Penny":['Penny'],
    "weixiaobao":['韦小宝'],
    "qiaofeng":['乔峰'],
    "ayaka":['神里绫华'],
    "raidenShogun":['雷电将军'],
    "yuqian":['于谦']
}

# 定义变量 HF_TOKEN，存储字符串 "hf_nPhmtMVuXy"
HF_TOKEN = "hf_nPhmtMVuXy"

# 调用 login 函数，传入 HF_TOKEN 进行登录
login(token=HF_TOKEN)

# 定义变量 jsonl_file_path，存储字符串 '/Users/pufferfish/Downloads/real_train_data_b/'
jsonl_file_path = '/Users/pufferfish/Downloads/real_train_data_b/'

# 定义变量 character_path，存储字符串 "/Users/pufferfish/Chat-Haruhi-Suzumiya/characters/"
character_path = "/Users/pufferfish/Chat-Haruhi-Suzumiya/characters/"

# 使用 os.listdir 函数列出 jsonl_file_path 目录下的所有文件名，存储在变量 file_names 中
file_names = os.listdir("/Users/pufferfish/Downloads/real_train_data_b/")

# 定义空列表 all_datasets，用于存储所有 CharacterDataset 对象
all_datasets = []

# 遍历 file_names 列表中的每个文件名
for filename in file_names:
    # 使用 '_' 分割文件名，存储在 filename_list 列表中
    filename_list = filename.split("_")
    # 初始化 character_name 变量为空字符串
    character_name = ""
    
    # 如果 filename_list 的第一个元素在 dic 字典的键中
    if filename_list[0] in dic.keys():
        # 将 character_name 设置为 filename_list 的第一个元素对应的值
        character_name = filename_list[0]
    
    # 如果 filename_list 的第二个元素在 dic 字典的键中
    if filename_list[1] in dic.keys():
        # 将 character_name 设置为 filename_list 的第二个元素对应的值
        character_name = filename_list[1]
    
    # 将 character_name 与 character_path 拼接，得到角色文件路径，存储在变量 character 中
    character = os.path.join(character_path, character_name)
    
    # 将 jsonl_file_path 与 filename 拼接，得到完整的 JSONL 文件路径，存储在变量 jsonl_file 中
    jsonl_file = os.path.join(jsonl_file_path, filename)
    
    # 调用 read_jsonl_file 函数读取 jsonl_file 中的数据，存储在变量 jsonl_data 中
    jsonl_data = read_jsonl_file(jsonl_file)
    
    # 创建 CharacterDataset 对象 c，传入 jsonl_data、character、8 和 2000 作为参数
    c = CharacterDataset(jsonl_data, character, 8, 2000)
    
    # 将创建的 CharacterDataset 对象 c 添加到 all_datasets 列表中
    all_datasets.append(c)
# 导入必要的库和模块
import json  # 导入处理 JSON 格式数据的模块
from torch.utils.data import Dataset, DataLoader  # 导入 PyTorch 中处理数据集和数据加载的模块
import os  # 导入操作系统相关功能的模块
import jsonlines  # 导入处理 JSONlines 格式数据的模块
from torch.utils.data import ConcatDataset  # 导入合并数据集的模块
from transformers import AutoTokenizer, AutoModel  # 导入 Hugging Face Transformers 库中的 Tokenizer 和 Model
from datasets import load_dataset, concatenate_datasets  # 导入 Hugging Face Datasets 库中的数据集相关功能
import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
from transformers import Trainer, TrainingArguments  # 导入 Hugging Face Transformers 中的训练器和训练参数
from huggingface_hub import login  # 导入 Hugging Face Hub 中的登录功能
from dataset import CharacterDataset, read_jsonl_file, collate_fn  # 导入自定义的数据集和相关函数
from utils import get_embedding, download_models  # 导入自定义的工具函数

# 使用预设的 Hugging Face 令牌登录
HF_TOKEN = "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"
login(token=HF_TOKEN)

# 加载两个数据集
train_dataset_dict_A = load_dataset('silk-road/Chat_Suzumiya_Fusion')  # 加载第一个数据集
train_dataset_dict_B = load_dataset('silk-road/Chat_Suzumiya_Fusion_B')  # 加载第二个数据集

# 从DatasetDict中获取具体的训练数据集
train_dataset_A = train_dataset_dict_A['train']
train_dataset_B = train_dataset_dict_B['train']

# 合并两个数据集
train_dataset = concatenate_datasets([train_dataset_A, train_dataset_B])

# 将合并后的数据集推送到 Hugging Face Hub
from datasets import load_dataset, Dataset  # 导入 Hugging Face Datasets 库中的数据集相关功能
train_dataset.push_to_hub("silk-road/Chat-Haruhi-Fusion-A_B")
```