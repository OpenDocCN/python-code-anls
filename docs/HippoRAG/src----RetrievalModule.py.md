# `.\HippoRAG\src\RetrievalModule.py`

```py
# 导入 sys 模块，用于访问和操作 Python 运行时环境
import sys

# 将当前目录添加到系统路径中，以便可以导入当前目录下的模块
sys.path.append('.')

# 导入 _pickle 模块，并将其别名为 pickle，用于序列化和反序列化对象
import _pickle as pickle

# 导入 argparse 模块，用于处理命令行参数
import argparse

# 从 glob 模块中导入 glob 函数，用于查找符合特定模式的文件路径
from glob import glob

# 导入 os.path 模块，用于处理文件和目录路径
import os.path

# 导入 ipdb 模块，用于调试
import ipdb

# 导入 pandas 模块，并将其别名为 pd，用于数据处理和分析
import pandas as pd

# 导入 pickle 模块，用于序列化和反序列化对象（通常和上面的 _pickle 冲突，建议保持一个）
import pickle

# 导入 numpy 模块，并将其别名为 np，用于科学计算
import numpy as np

# 导入 os 模块，用于处理操作系统相关的功能，例如文件和目录操作
import os

# 导入 tqdm 模块，用于显示循环进度条
from tqdm import tqdm

# 导入 faiss 模块，用于高效的相似性搜索
import faiss

# 导入 gc 模块，用于垃圾回收和内存管理
import gc

# 从 transformers 模块中导入 AutoModel 和 AutoTokenizer，用于加载预训练模型和分词器
from transformers import AutoModel, AutoTokenizer

# 从 processing 模块中导入所有内容（假设 processing.py 中有其他功能）
from processing import *

# TODO: 更改硬编码的向量输出目录
VECTOR_DIR = 'data/lm_vectors'

# 定义 RetrievalModule 类
class RetrievalModule:
    """
    设计用于从一组实体中检索潜在的同义词候选项的类，用于处理 UMLS 术语集合。
    """
    # 初始化方法，用于设置对象的初始状态
        def __init__(self,
                     retriever_name,  # 检索器名称
                     string_filename,  # 字符串文件名
                     pool_method='cls'  # 池化方法，默认为 'cls'
                     ):
            """
            参数:
                retriever_name: 检索器名称可以是以下三种类型之一
                    2) 映射 AUI 到预计算向量的 pickle 文件的名称
                    3) Huggingface 变换器模型
            """
    
            # 保存检索器名称
            self.retriever_name = retriever_name
    
            # 初始化检索名称目录为空
            self.retrieval_name_dir = None
    
            # 输出未找到预计算向量的提示，确认 PLM 模型
            print('No Pre-Computed Vectors. Confirming PLM Model.')
    
            try:
                # 如果检索器名称包含 'ckpt'，则从检查点加载模型
                if 'ckpt' in retriever_name:
                    self.plm = AutoModel.load_from_checkpoint(retriever_name)
                else:
                    # 否则从预训练模型加载
                    self.plm = AutoModel.from_pretrained(retriever_name)
            except:
                # 如果检索器名称无效，则抛出断言错误并打印错误信息
                assert False, print('{} is an invalid retriever name. Check Documentation.'.format(retriever_name))
    
            # 如果没有预计算向量，创建向量目录
            self.retrieval_name_dir = VECTOR_DIR + '/' + self.retriever_name.replace('/', '_').replace('.', '') + '_' + pool_method
    
            # 如果向量目录不存在，则创建该目录
            if not (os.path.exists(self.retrieval_name_dir)):
                os.makedirs(self.retrieval_name_dir)
    
            # 获取先前计算的向量
            precomp_strings, precomp_vectors = self.get_precomputed_plm_vectors(self.retrieval_name_dir)
    
            # 从字符串文件中读取 AUI 字符串，并进行处理
            string_df = pd.read_csv(string_filename, sep='\t')
            string_df.strings = [processing_phrases(str(s)) for s in string_df.strings]
            sorted_df = self.create_sorted_df(string_df.strings.values)
    
            # 识别缺失的字符串
            missing_strings = self.find_missing_strings(sorted_df.strings.unique(), precomp_strings)
    
            # 如果有缺失字符串，则进行编码
            if len(missing_strings) > 0:
                print('Encoding {} Missing Strings'.format(len(missing_strings)))
                new_vectors, new_strings, = self.encode_strings(missing_strings, pool_method)
    
                # 更新预计算的字符串和向量
                precomp_strings = list(precomp_strings)
                precomp_vectors = list(precomp_vectors)
    
                precomp_strings.extend(list(new_strings))
                precomp_vectors.extend(list(new_vectors))
    
                # 将预计算向量转换为 numpy 数组
                precomp_vectors = np.array(precomp_vectors)
    
                # 保存更新后的向量
                self.save_vecs(precomp_strings, precomp_vectors, self.retrieval_name_dir)
    
            # 创建向量字典
            self.vector_dict = self.make_dictionary(sorted_df, precomp_strings, precomp_vectors)
    
            # 输出加载向量的提示
            print('Vectors Loaded.')
    
            # 根据字符串类型分离查询和知识库数据
            queries = string_df[string_df.type == 'query']
            kb = string_df[string_df.type == 'kb']
    
            # 检索 KNN 结果
            nearest_neighbors = self.retrieve_knn(queries.strings.values, kb.strings.values)
            # 将最近邻结果保存到 pickle 文件
            pickle.dump(nearest_neighbors, open(self.retrieval_name_dir + '/nearest_neighbor_{}.p'.format(string_filename.split('/')[1].split('.')[0]), 'wb'))
    # 获取预计算的 PLM 向量和字符串
    def get_precomputed_plm_vectors(self, retrieval_name_dir):
        # 加载或创建一个按短语长度排序的 DataFrame 以便于 PLM 计算
        strings = self.load_precomp_strings(retrieval_name_dir)
        vectors = self.load_plm_vectors(retrieval_name_dir)
    
        # 返回字符串和对应的向量
        return strings, vectors
    
    # 创建一个按字符串长度排序的 DataFrame
    def create_sorted_df(self, strings):
        lengths = []
    
        # 计算每个字符串的长度，并存入列表
        for string in tqdm(strings):
            lengths.append(len(str(string)))
    
        # 将长度和字符串组成 DataFrame
        lengths_df = pd.DataFrame(lengths)
        lengths_df['strings'] = strings
    
        # 按长度排序 DataFrame
        return lengths_df.sort_values(0)
    
    # 保存字符串和向量到文件
    def save_vecs(self, strings, vectors, direc_name, bin_size=50000):
        # 将字符串保存到文件
        with open(direc_name + '/encoded_strings.txt', 'w') as f:
            for string in strings:
                f.write(string + '\n')
    
        # 将向量分割成多个文件保存
        split_vecs = np.array_split(vectors, int(len(vectors) / bin_size) + 1)
    
        # 逐个保存向量文件
        for i, vecs in tqdm(enumerate(split_vecs)):
            pickle.dump(vecs, open(direc_name + '/vecs_{}.p'.format(i), 'wb'))
    
    # 加载预计算的字符串
    def load_precomp_strings(self, retrieval_name_dir):
        filename = retrieval_name_dir + '/encoded_strings.txt'
    
        # 如果文件不存在，返回空列表
        if not (os.path.exists(filename)):
            return []
    
        # 读取字符串文件
        with open(filename, 'r') as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]
    
        # 返回去掉换行符的字符串列表
        return lines
    
    # 加载 PLM 向量
    def load_plm_vectors(self, retrieval_name_dir):
        vectors = []
    
        print('Loading PLM Vectors.')
        files = glob(retrieval_name_dir + '/vecs_*.p')
    
        # 如果没有向量文件，返回空列表
        if len(files) == 0:
            return vectors
    
        # 逐个加载向量文件
        for i in tqdm(range(len(files))):
            i_files = glob(retrieval_name_dir + '/*_{}.p'.format(i))
            if len(i_files) != 1:
                break
            else:
                vectors.append(pickle.load(open(i_files[0], 'rb')))
    
        # 合并所有向量
        vectors = np.vstack(vectors)
    
        return vectors
    
    # 查找缺失的字符串
    def find_missing_strings(self, relevant_strings, precomputed_strings):
        # 返回在相关字符串中但不在预计算字符串中的字符串
        return list(set(relevant_strings).difference(set(precomputed_strings)))
    
    # 创建字符串到向量的字典
    def make_dictionary(self, sorted_df, precomp_strings, precomp_vectors):
        print('Populating Vector Dict')
        precomp_string_ids = {}
    
        # 为每个预计算的字符串分配一个唯一 ID
        for i, string in enumerate(precomp_strings):
            precomp_string_ids[string] = i
    
        vector_dict = {}
    
        # 将按长度排序的字符串与向量关联
        for i, row in tqdm(sorted_df.iterrows(), total=len(sorted_df)):
            string = row.strings
    
            try:
                vector_id = precomp_string_ids[string]
                vector_dict[string] = precomp_vectors[vector_id]
            except:
                ipdb.set_trace()
    
        return vector_dict
    # 定义一个编码字符串的方法，接受要编码的字符串列表和池化方法
        def encode_strings(self, strs_to_encode, pool_method):
            # 将模型加载到 GPU 上
            self.plm.to('cuda')
            # 从预训练模型中加载分词器
            tokenizer = AutoTokenizer.from_pretrained(self.retriever_name)
    
            # 根据字符串的长度排序
            sorted_missing_strings = [len(s) for s in strs_to_encode]
            strs_to_encode = list(np.array(strs_to_encode)[np.argsort(sorted_missing_strings)])
    
            all_cls = []  # 存储所有 CLS 向量
            all_strings = []  # 存储所有处理过的字符串
            num_strings_proc = 0  # 处理过的字符串数量
    
            # 禁用梯度计算以节省内存
            with torch.no_grad():
    
                batch_sizes = []  # 存储每个批次的大小
    
                text_batch = []  # 当前批次的字符串
                max_pad_size = 0  # 当前批次的最大填充大小
    
                # 遍历排序后的字符串列表
                for i, string in tqdm(enumerate(strs_to_encode), total=len(strs_to_encode)):
    
                    # 获取当前字符串的 token 数量
                    length = len(tokenizer.tokenize(string))
    
                    # 将当前字符串添加到批次中
                    text_batch.append(string)
                    num_strings_proc += 1  # 增加处理过的字符串计数
    
                    # 更新最大填充大小
                    if length > max_pad_size:
                        max_pad_size = length
    
                    # 当批次大小超过限制或所有字符串都处理完时
                    if max_pad_size * len(text_batch) > 50000 or num_strings_proc == len(strs_to_encode):
    
                        # 将批次转换为列表
                        text_batch = list(text_batch)
                        # 对当前批次的字符串进行编码
                        encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True,
                                             max_length=self.plm.config.max_length)
                        input_ids = encoding['input_ids']
                        attention_mask = encoding['attention_mask']
    
                        # 将编码的输入和注意力掩码移动到 GPU 上
                        input_ids = input_ids.to('cuda')
                        attention_mask = attention_mask.to('cuda')
    
                        # 通过模型获取输出
                        outputs = self.plm(input_ids, attention_mask=attention_mask)
    
                        # 根据池化方法计算嵌入
                        if pool_method == 'cls':
                            embeddings = outputs[0][:, 0, :]
    
                        elif pool_method == 'mean':
                            embeddings = mean_pooling(outputs[0], attention_mask)
    
                        # 将嵌入和字符串添加到结果列表中
                        all_cls.append(embeddings.cpu().numpy())
                        all_strings.extend(text_batch)
    
                        batch_sizes.append(len(text_batch))
    
                        # 重置批次和最大填充大小
                        text_batch = []
                        max_pad_size = 0
    
            # 将所有 CLS 向量垂直堆叠
            all_cls = np.vstack(all_cls)
    
            # 确保所有 CLS 向量和字符串的数量匹配
            assert len(all_cls) == len(all_strings)
            # 确保所有处理过的字符串与原始字符串列表匹配
            assert all([all_strings[i] == strs_to_encode[i] for i in range(len(all_strings))])
    
            # 返回所有 CLS 向量和所有处理过的字符串
            return all_cls, all_strings
# 当脚本作为主程序运行时执行以下代码
if __name__ == '__main__':
    # 创建一个 ArgumentParser 对象用于解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加一个可选参数 '--retriever_name'，类型为字符串，提供检索模型名称
    parser.add_argument('--retriever_name', type=str, help='retrieval model name, e.g., "facebook/contriever"')
    # 添加一个可选参数 '--string_filename'，类型为字符串，用于输入文件名
    parser.add_argument('--string_filename', type=str)
    # 添加一个可选参数 '--pool_method'，类型为字符串，默认为 'mean'
    parser.add_argument('--pool_method', type=str, default='mean')

    # 解析命令行参数
    args = parser.parse_args()

    # 从解析结果中提取检索模型名称
    retriever_name = args.retriever_name
    # 从解析结果中提取文件名
    string_filename = args.string_filename
    # 从解析结果中提取池化方法，默认为 'mean'
    pool_method = args.pool_method

    # 使用提取的参数初始化 RetrievalModule 对象
    retrieval_module = RetrievalModule(retriever_name, string_filename, pool_method)
```