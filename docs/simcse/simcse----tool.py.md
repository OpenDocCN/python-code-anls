# `.\simcse\tool.py`

```
# 导入日志记录模块
import logging
# 导入进度条显示模块
from tqdm import tqdm
# 导入NumPy库，并引入ndarray类型
import numpy as np
from numpy import ndarray
# 导入PyTorch库
import torch
# 从PyTorch中导入Tensor和device类
from torch import Tensor, device
# 导入transformers库
import transformers
# 从transformers库中导入AutoModel和AutoTokenizer类
from transformers import AutoModel, AutoTokenizer
# 导入用于计算余弦相似度的模块
from sklearn.metrics.pairwise import cosine_similarity
# 导入用于数据归一化的模块
from sklearn.preprocessing import normalize
# 导入类型提示模块，包括List、Dict、Tuple、Type、Union
from typing import List, Dict, Tuple, Type, Union

# 配置日志记录格式和级别
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
# 创建日志记录器对象
logger = logging.getLogger(__name__)

# 定义SimCSE类，用于处理句子的嵌入、计算相似度以及检索句子
class SimCSE(object):
    """
    A class for embedding sentences, calculating similarities, and retrieving sentences by SimCSE.
    """
    # 初始化方法，接受模型名或路径、设备类型、单元格数目等参数
    def __init__(self, model_name_or_path: str, 
                device: str = None,
                num_cells: int = 100,
                num_cells_in_search: int = 10,
                pooler = None):

        # 根据模型名或路径创建自动分词器对象
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # 根据模型名或路径创建自动模型对象
        self.model = AutoModel.from_pretrained(model_name_or_path)
        
        # 如果未指定设备，则根据CUDA是否可用选择设备类型
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # 设置对象的设备属性
        self.device = device

        # 初始化索引对象和相关属性
        self.index = None
        self.is_faiss_index = False
        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search

        # 如果指定了池化策略，则使用指定的池化策略；否则根据模型名判断适合的池化策略
        if pooler is not None:
            self.pooler = pooler
        elif "unsup" in model_name_or_path:
            # 对于无监督模型，建议使用'cls_before_pooler'作为池化策略
            logger.info("Use `cls_before_pooler` for unsupervised models. If you want to use other pooling policy, specify `pooler` argument.")
            self.pooler = "cls_before_pooler"
        else:
            # 对于其他模型，默认使用'cls'作为池化策略
            self.pooler = "cls"
    # 对输入的句子或句子列表进行编码为嵌入向量的操作
    def encode(self, sentence: Union[str, List[str]], 
                device: str = None, 
                return_numpy: bool = False,
                normalize_to_unit: bool = True,
                keepdim: bool = False,
                batch_size: int = 64,
                max_length: int = 128) -> Union[ndarray, Tensor]:

        # 确定目标设备，若未指定则使用类实例的设备
        target_device = self.device if device is None else device
        # 将模型移动到目标设备上
        self.model = self.model.to(target_device)
        
        # 如果输入是单个字符串，转换为字符串列表，并标记为单个句子处理
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        # 初始化嵌入列表
        embedding_list = [] 
        # 使用无梯度计算上下文
        with torch.no_grad():
            # 计算总批次数
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            # 迭代每个批次
            for batch_id in tqdm(range(total_batch)):
                # 使用分词器处理批次的输入句子，设置填充和截断，并返回PyTorch张量
                inputs = self.tokenizer(
                    sentence[batch_id*batch_size:(batch_id+1)*batch_size], 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length, 
                    return_tensors="pt"
                )
                # 将输入张量移动到目标设备
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                # 使用模型进行推理，返回嵌入结果，使用字典形式返回
                outputs = self.model(**inputs, return_dict=True)
                # 根据汇聚器类型选择嵌入结果
                if self.pooler == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooler == "cls_before_pooler":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    raise NotImplementedError
                # 若设置为单位化，对嵌入进行单位化处理
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                # 将结果添加到嵌入列表中（CPU上）
                embedding_list.append(embeddings.cpu())
        # 拼接所有批次的嵌入结果
        embeddings = torch.cat(embedding_list, 0)
        
        # 如果输入是单个句子且不需要保持维度，则返回单个嵌入
        if single_sentence and not keepdim:
            embeddings = embeddings[0]
        
        # 如果需要返回NumPy数组且结果不是NumPy数组，则转换为NumPy数组并返回
        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        # 否则直接返回嵌入张量
        return embeddings
    def similarity(self, queries: Union[str, List[str]], 
                    keys: Union[str, List[str], ndarray], 
                    device: str = None) -> Union[float, ndarray]:
        # 使用编码器对查询进行编码，假设有N个查询，返回结果为numpy数组
        query_vecs = self.encode(queries, device=device, return_numpy=True) # suppose N queries
        
        # 检查keys是否为ndarray类型
        if not isinstance(keys, ndarray):
            # 使用编码器对键进行编码，假设有M个键，返回结果为numpy数组
            key_vecs = self.encode(keys, device=device, return_numpy=True) # suppose M keys
        else:
            key_vecs = keys

        # 检查查询向量是否为单个向量（1维），键向量是否为单个向量（1维）
        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1 
        if single_query:
            # 将单个查询向量reshape为2维数组
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            # 将单个键向量reshape为2维数组
            key_vecs = key_vecs.reshape(1, -1)
        
        # 计算查询向量与键向量之间的余弦相似度，返回一个N*M的相似度数组
        similarities = cosine_similarity(query_vecs, key_vecs)
        
        if single_query:
            # 如果查询向量为单个向量，则将相似度数组转为1维数组
            similarities = similarities[0]
            if single_key:
                # 如果键向量也为单个向量，则将相似度数组的第一个元素转为浮点数
                similarities = float(similarities[0])
        
        # 返回计算得到的相似度结果
        return similarities
    # 定义一个方法，用于构建索引
    def build_index(self, sentences_or_file_path: Union[str, List[str]], 
                        use_faiss: bool = None,
                        faiss_fast: bool = False,
                        device: str = None,
                        batch_size: int = 64):
        
        # 如果 use_faiss 为 None 或者 True，则尝试导入 faiss 库，并检查是否有 IndexFlatIP 属性
        if use_faiss is None or use_faiss:
            try:
                import faiss
                assert hasattr(faiss, "IndexFlatIP")
                use_faiss = True
            except:
                # 如果导入失败，则警告并继续使用暴力搜索
                logger.warning("Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search.")
                use_faiss = False
        
        # 如果输入的句子是字符串，则假设它是存储各种句子的文件路径
        if isinstance(sentences_or_file_path, str):
            sentences = []
            # 从文件中加载句子
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences
        
        # 记录日志，指示正在对句子进行编码生成嵌入向量
        logger.info("Encoding embeddings for sentences...")
        # 使用模型的 encode 方法生成嵌入向量
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True)

        # 记录日志，指示正在构建索引
        logger.info("Building index...")
        # 初始化索引字典，包含句子列表
        self.index = {"sentences": sentences_or_file_path}
        
        # 如果使用 faiss 进行索引构建
        if use_faiss:
            # 创建一个 faiss.IndexFlatIP 类型的量化器
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])
            # 如果选择了 faiss 快速模式
            if faiss_fast:
                # 使用 faiss.IndexIVFFlat 类型的索引器，设置最小单元格数，并指定内积度量
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(self.num_cells, len(sentences_or_file_path)), faiss.METRIC_INNER_PRODUCT)
            else:
                # 否则，直接使用量化器作为索引器
                index = quantizer

            # 如果当前设备是 CUDA，并且指定了使用 GPU 或者设备是 CUDA
            if (self.device == "cuda" and device != "cpu") or device == "cuda":
                # 如果 faiss 支持 GPU 资源
                if hasattr(faiss, "StandardGpuResources"):
                    logger.info("Use GPU-version faiss")
                    # 创建标准 GPU 资源对象
                    res = faiss.StandardGpuResources()
                    # 设置临时内存大小
                    res.setTempMemory(20 * 1024 * 1024 * 1024)
                    # 将索引器移动到 GPU
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                else:
                    # 否则，使用 CPU 版本的 faiss
                    logger.info("Use CPU-version faiss")
            else:
                # 否则，使用 CPU 版本的 faiss
                logger.info("Use CPU-version faiss")

            # 如果选择了 faiss 快速模式
            if faiss_fast:            
                # 对嵌入向量进行训练
                index.train(embeddings.astype(np.float32))
            # 将嵌入向量添加到索引中
            index.add(embeddings.astype(np.float32))
            # 设置探测单元数
            index.nprobe = min(self.num_cells_in_search, len(sentences_or_file_path))
            # 标记使用了 faiss 索引
            self.is_faiss_index = True
        else:
            # 否则，直接使用嵌入向量作为索引
            index = embeddings
            # 标记未使用 faiss 索引
            self.is_faiss_index = False
        
        # 将索引存储在索引字典中
        self.index["index"] = index
        # 记录日志，指示索引构建完成
        logger.info("Finished")
    # 将句子或文件路径添加到索引中
    def add_to_index(self, sentences_or_file_path: Union[str, List[str]],
                     device: str = None,
                     batch_size: int = 64):

        # 如果输入的句子是字符串，则假定它是存储各种句子的文件的路径
        if isinstance(sentences_or_file_path, str):
            sentences = []
            # 从文件中读取句子并加载到列表中
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences
        
        # 记录日志：对句子进行嵌入编码
        logger.info("Encoding embeddings for sentences...")
        # 使用编码函数对句子进行编码，返回嵌入向量
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True)
        
        # 如果使用的是 Faiss 索引，则添加嵌入向量
        if self.is_faiss_index:
            self.index["index"].add(embeddings.astype(np.float32))
        else:
            # 否则，将嵌入向量连接到索引中的现有向量中
            self.index["index"] = np.concatenate((self.index["index"], embeddings))
        # 将新的句子添加到索引的句子列表中
        self.index["sentences"] += sentences_or_file_path
        # 记录日志：添加操作完成
        logger.info("Finished")


    
    # 在索引中搜索查询
    def search(self, queries: Union[str, List[str]], 
               device: str = None, 
               threshold: float = 0.6,
               top_k: int = 5) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        
        # 如果不使用 Faiss 索引
        if not self.is_faiss_index:
            # 如果查询是一个列表，返回每个查询的组合结果
            if isinstance(queries, list):
                combined_results = []
                for query in queries:
                    results = self.search(query, device, threshold, top_k)
                    combined_results.append(results)
                return combined_results
            
            # 计算查询与索引中句子的相似度
            similarities = self.similarity(queries, self.index["index"]).tolist()
            id_and_score = []
            # 筛选出相似度大于等于阈值的结果
            for i, s in enumerate(similarities):
                if s >= threshold:
                    id_and_score.append((i, s))
            # 根据相似度排序并取前 top_k 个结果
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
            # 返回查询结果及其相似度
            results = [(self.index["sentences"][idx], score) for idx, score in id_and_score]
            return results
        else:
            # 对查询向量进行编码
            query_vecs = self.encode(queries, device=device, normalize_to_unit=True, keepdim=True, return_numpy=True)

            # 使用 Faiss 进行向量检索
            distance, idx = self.index["index"].search(query_vecs.astype(np.float32), top_k)
            
            # 定义打包单个查询结果的函数
            def pack_single_result(dist, idx):
                # 返回距离小于阈值的结果及其距离
                results = [(self.index["sentences"][i], s) for i, s in zip(idx, dist) if s >= threshold]
                return results
            
            # 如果查询是一个列表，返回每个查询的组合结果
            if isinstance(queries, list):
                combined_results = []
                for i in range(len(queries)):
                    results = pack_single_result(distance[i], idx[i])
                    combined_results.append(results)
                return combined_results
            else:
                return pack_single_result(distance[0], idx[0])
if __name__=="__main__":
    example_sentences = [
        'An animal is biting a persons finger.',
        'A woman is reading.',
        'A man is lifting weights in a garage.',
        'A man plays the violin.',
        'A man is eating food.',
        'A man plays the piano.',
        'A panda is climbing.',
        'A man plays a guitar.',
        'A woman is slicing a meat.',
        'A woman is taking a picture.'
    ]
    example_queries = [
        'A man is playing music.',
        'A woman is making a photo.'
    ]

    model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
    simcse = SimCSE(model_name)

    # 输出计算查询和句子之间的余弦相似度
    print("\n=========Calculate cosine similarities between queries and sentences============\n")
    similarities = simcse.similarity(example_queries, example_sentences)
    print(similarities)

    # 输出使用朴素的蛮力搜索构建索引的结果
    print("\n=========Naive brute force search============\n")
    simcse.build_index(example_sentences, use_faiss=False)
    results = simcse.search(example_queries)
    for i, result in enumerate(results):
        print("Retrieval results for query: {}".format(example_queries[i]))
        for sentence, score in result:
            print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
        print("")
    
    # 输出使用 Faiss 后端构建索引的搜索结果
    print("\n=========Search with Faiss backend============\n")
    simcse.build_index(example_sentences, use_faiss=True)
    results = simcse.search(example_queries)
    for i, result in enumerate(results):
        print("Retrieval results for query: {}".format(example_queries[i]))
        for sentence, score in result:
            print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
        print("")
```