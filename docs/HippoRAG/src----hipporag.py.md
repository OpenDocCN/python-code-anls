# `.\HippoRAG\src\hipporag.py`

```py
# 导入标准库和第三方库
import json  # 用于处理 JSON 数据
import logging  # 用于记录日志
import os  # 用于操作系统相关功能
import _pickle as pickle  # 用于序列化和反序列化对象
from collections import defaultdict  # 提供默认值的字典
from glob import glob  # 用于文件路径模式匹配

import igraph as ig  # 用于图形和网络分析
import numpy as np  # 提供支持数组和矩阵运算的功能
import pandas as pd  # 用于数据处理和分析
import torch  # 用于深度学习
from colbert import Searcher  # 从 colbert 库中导入 Searcher 类
from colbert.data import Queries  # 从 colbert 库中导入 Queries 类
from colbert.infra import RunConfig, Run, ColBERTConfig  # 从 colbert 库中导入相关配置和运行类
from tqdm import tqdm  # 用于显示循环进度条

from src.colbertv2_indexing import colbertv2_index  # 从 src 模块中导入 colbertv2_index 函数
from src.langchain_util import init_langchain_model, LangChainModel  # 从 src 模块中导入初始化和模型类
from src.lm_wrapper.util import init_embedding_model  # 从 src 模块中导入初始化嵌入模型函数
from src.named_entity_extraction_parallel import named_entity_recognition  # 从 src 模块中导入命名实体识别函数
from src.processing import processing_phrases, min_max_normalize  # 从 src 模块中导入处理短语和归一化函数

# 设置环境变量，禁用并行的标记器
os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'

# 定义 ColBERT 模型的检查点目录
COLBERT_CKPT_DIR = "exp/colbertv2.0"


class HippoRAG:

    def get_passage_by_idx(self, passage_idx):
        """
        根据索引获取段落
        @param passage_idx: 段落的索引
        @return: 段落内容
        """
        # 从数据集中根据给定的索引返回相应的段落
        return self.dataset_df.iloc[passage_idx]['paragraph']

    def get_extraction_by_passage_idx(self, passage_idx, chunk=False):
        """
        获取特定段落的提取结果
        @param passage_idx: 段落索引，即每个段落字典中的 'idx'，而不是语料库中的数组索引
        @param chunk: 是否对语料库进行了分块
        @return: 段落的提取结果
        """
        # 在 self.extracted_triples 中查找与 passage_idx 匹配的条目
        for item in self.extracted_triples:
            if not chunk and item['idx'] == passage_idx:
                return item  # 返回匹配的条目
            elif chunk and (item['idx'] == passage_idx or item['idx'].startswith(passage_idx + '_')):
                return item  # 返回匹配的条目
        return None  # 如果没有找到匹配的条目，返回 None

    def get_shortest_distance_between_nodes(self, node1: str, node2: str):
        """
        获取图中两个节点之间的最短距离
        @param node1: 节点1 的短语
        @param node2: 节点2 的短语
        @return: 两个节点之间的最短距离
        """
        try:
            # 查找节点1 和 节点2 的索引
            node1_id = np.where(self.phrases == node1)[0][0]
            node2_id = np.where(self.phrases == node2)[0][0]

            # 返回两个节点之间的最短路径长度
            return self.g.shortest_paths(node1_id, node2_id)[0][0]
        except Exception as e:
            # 如果出现异常，返回 -1 表示错误
            return -1

    def query_ner(self, query):
        # 根据是否仅使用 DPR 进行初始化
        if self.dpr_only:
            query_ner_list = []
        else:
            # 提取实体
            try:
                if query in self.named_entity_cache:
                    # 如果缓存中已有结果，直接使用缓存
                    query_ner_list = self.named_entity_cache[query]['named_entities']
                else:
                    # 否则调用命名实体识别函数并解析结果
                    query_ner_json, total_tokens = named_entity_recognition(self.client, query)
                    query_ner_list = eval(query_ner_json)['named_entities']

                # 处理短语
                query_ner_list = [processing_phrases(p) for p in query_ner_list]
            except:
                # 如果发生错误，记录日志并返回空列表
                self.logger.error('Error in Query NER')
                query_ner_list = []
        # 返回实体列表
        return query_ner_list
    # 获取指定深度内的邻居节点，并更新概率向量
    def get_neighbors(self, prob_vector, max_depth=1):

        # 获取概率向量中非零元素的索引
        initial_nodes = prob_vector.nonzero()[0]
        # 找到这些非零节点中最小的概率值
        min_prob = np.min(prob_vector[initial_nodes])

        # 遍历所有初始节点
        for initial_node in initial_nodes:
            all_neighborhood = []

            # 初始化当前节点为初始节点
            current_nodes = [initial_node]

            # 根据指定深度遍历邻居节点
            for depth in range(max_depth):
                next_nodes = []

                # 遍历当前节点，获取其所有邻居
                for node in current_nodes:
                    next_nodes.extend(self.g.neighbors(node))
                    all_neighborhood.extend(self.g.neighbors(node))

                # 更新当前节点为去重后的下一层节点
                current_nodes = list(set(next_nodes))

            # 更新所有邻居节点的概率值
            for i in set(all_neighborhood):
                prob_vector[i] += 0.5 * min_prob

        # 返回更新后的概率向量
        return prob_vector

    # 加载语料库
    def load_corpus(self):
        # 如果没有指定语料库路径，则设置默认路径
        if self.corpus_path is None:
            self.corpus_path = 'data/{}_corpus.json'.format(self.corpus_name)
        # 确保语料库文件存在
        assert os.path.isfile(self.corpus_path), 'Corpus file not found'
        # 从文件中加载语料库
        self.corpus = json.load(open(self.corpus_path, 'r'))
        # 创建一个空的数据框
        self.dataset_df = pd.DataFrame()
        # 将语料库中的每个段落合并成一个字符串，并存储到数据框中
        self.dataset_df['paragraph'] = [p['title'] + '\n' + p['text'] for p in self.corpus]

    # 从文档字符串中提取短语
    def get_phrases_in_doc_str(self, doc: str):
        # 尝试从数据框中找到文档 ID
        try:
            doc_id = self.dataset_df[self.dataset_df.paragraph == doc].index[0]
            # 获取该文档 ID 对应的短语 ID 列表
            phrase_ids = self.doc_to_phrases_mat[[doc_id], :].nonzero()[1].tolist()
            # 返回短语 ID 对应的短语
            return [self.phrases[phrase_id] for phrase_id in phrase_ids]
        except:
            # 如果文档 ID 未找到，则返回空列表
            return []

    # 构建图
    def build_graph(self):

        # 初始化边集合
        edges = set()

        # 初始化新图及其邻接列表
        new_graph_plus = {}
        self.kg_adj_list = defaultdict(dict)
        self.kg_inverse_adj_list = defaultdict(dict)

        # 遍历图中的每个边及其权重
        for edge, weight in tqdm(self.graph_plus.items(), total=len(self.graph_plus), desc='Building Graph'):
            edge1 = edge[0]
            edge2 = edge[1]

            # 仅当边不在集合中且边的两个节点不同才处理
            if (edge1, edge2) not in edges and edge1 != edge2:
                new_graph_plus[(edge1, edge2)] = self.graph_plus[(edge[0], edge[1])]
                edges.add((edge1, edge2))
                self.kg_adj_list[edge1][edge2] = self.graph_plus[(edge[0], edge[1])]
                self.kg_inverse_adj_list[edge2][edge1] = self.graph_plus[(edge[0], edge[1])]

        # 更新图的数据
        self.graph_plus = new_graph_plus

        # 将边集合转换为列表
        edges = list(edges)

        # 根据知识库节点-短语映射构建图
        n_vertices = len(self.kb_node_phrase_to_id)
        self.g = ig.Graph(n_vertices, edges)

        # 设置图中每条边的权重
        self.g.es['weight'] = [self.graph_plus[(v1, v3)] for v1, v3 in edges]
        # 记录构建图的日志信息
        self.logger.info(f'Graph built: num vertices: {n_vertices}, num_edges: {len(edges)}')
    # 定义加载节点向量的方法
    def load_node_vectors(self):
        # 根据处理后的链接检索名称构建编码字符串路径
        encoded_string_path = 'data/lm_vectors/{}_mean/encoded_strings.txt'.format(self.linking_retriever_name_processed)
        # 检查编码字符串文件是否存在
        if os.path.isfile(encoded_string_path):
            # 从字符串编码缓存中加载节点向量
            self.load_node_vectors_from_string_encoding_cache(encoded_string_path)
        else:  # 如果编码字符串文件不存在，使用其他方式加载节点向量
            # 如果链接检索名称是 'colbertv2'，直接返回
            if self.linking_retriever_name == 'colbertv2':
                return
            # 根据处理后的链接检索名称和语料库名称构建知识库节点短语嵌入路径
            kb_node_phrase_embeddings_path = 'data/lm_vectors/{}_mean/{}_kb_node_phrase_embeddings.p'.format(self.linking_retriever_name_processed, self.corpus_name)
            # 检查知识库节点短语嵌入文件是否存在
            if os.path.isfile(kb_node_phrase_embeddings_path):
                # 从文件中加载知识库节点短语嵌入
                self.kb_node_phrase_embeddings = pickle.load(open(kb_node_phrase_embeddings_path, 'rb'))
                # 如果嵌入的形状是三维，则在轴 1 上进行压缩
                if len(self.kb_node_phrase_embeddings.shape) == 3:
                    self.kb_node_phrase_embeddings = np.squeeze(self.kb_node_phrase_embeddings, axis=1)
                # 记录加载的短语嵌入的信息
                self.logger.info('Loaded phrase embeddings from: ' + kb_node_phrase_embeddings_path + ', shape: ' + str(self.kb_node_phrase_embeddings.shape))
            else:
                # 如果嵌入文件不存在，计算并保存短语嵌入
                self.kb_node_phrase_embeddings = self.embed_model.encode_text(self.phrases.tolist(), return_cpu=True, return_numpy=True, norm=True)
                pickle.dump(self.kb_node_phrase_embeddings, open(kb_node_phrase_embeddings_path, 'wb'))
                # 记录保存的短语嵌入的信息
                self.logger.info('Saved phrase embeddings to: ' + kb_node_phrase_embeddings_path + ', shape: ' + str(self.kb_node_phrase_embeddings.shape))

    # 定义从字符串编码缓存中加载节点向量的方法
    def load_node_vectors_from_string_encoding_cache(self, string_file_path):
        # 记录从字符串文件加载节点向量的信息
        self.logger.info('Loading node vectors from: ' + string_file_path)
        kb_vectors = []
        # 读取字符串文件的每一行
        self.strings = open(string_file_path, 'r').readlines()
        # 遍历所有的向量文件，加载并追加到 kb_vectors 列表中
        for i in range(len(glob('data/lm_vectors/{}_mean/vecs_*'.format(self.linking_retriever_name_processed)))):
            kb_vectors.append(
                torch.Tensor(pickle.load(
                    open('data/lm_vectors/{}_mean/vecs_{}.p'.format(self.linking_retriever_name_processed, i), 'rb'))))
        # 将所有的向量合并成一个矩阵
        kb_mat = torch.cat(kb_vectors)  # a matrix of phrase vectors
        # 去除字符串中的换行符并构建字符串到 ID 的映射
        self.strings = [s.strip() for s in self.strings]
        self.string_to_id = {string: i for i, string in enumerate(self.strings)}
        # 对 kb_mat 进行标准化，并转移到 GPU
        kb_mat = kb_mat.T.divide(torch.linalg.norm(kb_mat, dim=1)).T
        kb_mat = kb_mat.to('cuda')
        kb_only_indices = []
        num_non_vector_phrases = 0
        # 遍历每个短语，检查其是否在 string_to_id 中
        for i in range(len(self.kb_node_phrase_to_id)):
            phrase = self.phrases[i]
            if phrase not in self.string_to_id:
                num_non_vector_phrases += 1

            phrase_id = self.string_to_id.get(phrase, 0)
            kb_only_indices.append(phrase_id)
        # 根据索引提取相应的短语向量，并转回 CPU 和 NumPy 数组
        self.kb_node_phrase_embeddings = kb_mat[kb_only_indices]  # a matrix of phrase vectors
        self.kb_node_phrase_embeddings = self.kb_node_phrase_embeddings.cpu().numpy()
        # 记录未找到向量的短语数量
        self.logger.info('{} phrases did not have vectors.'.format(num_non_vector_phrases))
    def get_dpr_doc_embedding(self):
        # 构造缓存文件名，包含链接检索器名和语料库名
        cache_filename = 'data/lm_vectors/{}_mean/{}_doc_embeddings.p'.format(self.linking_retriever_name_processed, self.corpus_name)
        # 检查缓存文件是否存在
        if os.path.exists(cache_filename):
            # 从缓存文件中加载文档嵌入矩阵
            self.doc_embedding_mat = pickle.load(open(cache_filename, 'rb'))
            # 记录加载文档嵌入的信息，包括文件名和矩阵形状
            self.logger.info(f'Loaded doc embeddings from {cache_filename}, shape: {self.doc_embedding_mat.shape}')
        else:
            # 如果缓存文件不存在，初始化文档嵌入列表
            self.doc_embeddings = []
            # 使用嵌入模型对数据集中段落进行编码，生成文档嵌入矩阵
            self.doc_embedding_mat = self.embed_model.encode_text(self.dataset_df['paragraph'].tolist(), return_cpu=True, return_numpy=True, norm=True)
            # 将生成的文档嵌入矩阵保存到缓存文件
            pickle.dump(self.doc_embedding_mat, open(cache_filename, 'wb'))
            # 记录保存文档嵌入的信息，包括文件名和矩阵形状
            self.logger.info(f'Saved doc embeddings to {cache_filename}, shape: {self.doc_embedding_mat.shape}')

    def run_pagerank_igraph_chunk(self, reset_prob_chunk):
        """
        Run pagerank on the graph
        :param reset_prob_chunk: 重置概率的块
        :return: PageRank 概率
        """
        # 初始化存储 PageRank 概率的列表
        pageranked_probabilities = []

        # 遍历每个重置概率
        for reset_prob in tqdm(reset_prob_chunk, desc='pagerank chunk'):
            # 计算图中每个节点的 PageRank 概率
            pageranked_probs = self.g.personalized_pagerank(vertices=range(len(self.kb_node_phrase_to_id)), damping=self.damping, directed=False,
                                                            weights='weight', reset=reset_prob, implementation='prpack')

            # 将计算结果转换为 NumPy 数组，并添加到列表中
            pageranked_probabilities.append(np.array(pageranked_probs))

        # 将结果列表转换为 NumPy 数组并返回
        return np.array(pageranked_probabilities)

    def get_colbert_max_score(self, query):
        # 将查询转换为列表形式
        queries_ = [query]
        # 对查询进行编码
        encoded_query = self.phrase_searcher.encode(queries_, full_length_search=False)
        # 从文本中获取文档的编码
        encoded_doc = self.phrase_searcher.checkpoint.docFromText(queries_).float()
        # 计算查询和文档之间的最大分数，并将结果转为 NumPy 数组
        max_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

        return max_score

    def get_colbert_real_score(self, query, doc):
        # 将查询和文档转换为列表形式
        queries_ = [query]
        docs_ = [doc]
        # 对查询进行编码
        encoded_query = self.phrase_searcher.encode(queries_, full_length_search=False)
        # 从文本中获取文档的编码
        encoded_doc = self.phrase_searcher.checkpoint.docFromText(docs_).float()
        # 计算查询和文档之间的实际分数，并将结果转为 NumPy 数组
        real_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

        return real_score
    # 使用 ColBERTv2 模型根据查询实体列表连接节点
    def link_node_by_colbertv2(self, query_ner_list):
        # 初始化存储短语 ID 和最大评分的列表
        phrase_ids = []
        max_scores = []
    
        # 遍历查询实体列表
        for query in query_ner_list:
            # 创建查询对象，路径为 None，数据包含当前查询
            queries = Queries(path=None, data={0: query})
    
            # 将查询封装成列表
            queries_ = [query]
            # 使用 ColBERTv2 的短语搜索器对查询进行编码
            encoded_query = self.phrase_searcher.encode(queries_, full_length_search=False)
    
            # 获取当前查询的 ColBERT 最大评分
            max_score = self.get_colbert_max_score(query)
    
            # 使用 ColBERTv2 进行检索，返回前 k 个（此处为 1）结果
            ranking = self.phrase_searcher.search_all(queries, k=1)
            # 遍历检索结果的短语 ID、排名和评分
            for phrase_id, rank, score in ranking.data[0]:
                # 从短语 ID 获取短语
                phrase = self.phrases[phrase_id]
                # 将短语封装成列表
                phrases_ = [phrase]
                # 从文本生成短语的编码，转换为浮点数
                encoded_doc = self.phrase_searcher.checkpoint.docFromText(phrases_).float()
                # 计算实际评分：查询编码与文档编码的矩阵乘积的最大值
                real_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()
    
                # 将短语 ID 和实际评分添加到列表中
                phrase_ids.append(phrase_id)
                max_scores.append(real_score / max_score)
    
        # 创建一个与短语数量相同的向量，初始化为全零
        top_phrase_vec = np.zeros(len(self.phrases))
    
        # 根据短语 ID 更新向量
        for phrase_id in phrase_ids:
            if self.node_specificity:
                # 根据短语的文档数调整权重
                if self.phrase_to_num_doc[phrase_id] == 0:
                    weight = 1
                else:
                    weight = 1 / self.phrase_to_num_doc[phrase_id]
                top_phrase_vec[phrase_id] = weight
            else:
                # 如果没有节点特异性，权重设为 1.0
                top_phrase_vec[phrase_id] = 1.0
    
        # 返回短语向量和查询与短语评分的映射
        return top_phrase_vec, {(query, self.phrases[phrase_id]): max_score for phrase_id, max_score, query in zip(phrase_ids, max_scores, query_ner_list)}
    def link_node_by_dpr(self, query_
```