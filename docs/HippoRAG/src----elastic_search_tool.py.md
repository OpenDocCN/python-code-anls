# `.\HippoRAG\src\elastic_search_tool.py`

```py
# 导入时间模块，用于处理时间相关操作
import time

# 从 tqdm 模块导入 tqdm 函数，用于显示进度条
from tqdm import tqdm


# 创建并索引 Elasticsearch 索引
def create_and_index(es, index_name, corpus_contents, similarity):
    # 如果指定的索引不存在，则创建它
    if not es.indices.exists(index=index_name):
        # 定义索引创建的设置和映射
        create_index_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index": {
                    "similarity": {
                        "default": {
                            "type": similarity
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text"
                    }
                }
            }
        }
        # 创建索引
        es.indices.create(index=index_name, body=create_index_body)

        # 遍历语料库内容，将每个文档索引到 Elasticsearch 中
        for idx, doc in tqdm(enumerate(corpus_contents), total=len(corpus_contents), desc='indexing'):
            # 如果索引失败（例如网络问题），则重试几次
            for num_attempt in range(10):
                try:
                    # 尝试将文档索引到指定的 Elasticsearch 索引
                    es.index(index=index_name, id=idx, body={"content": doc})
                    break
                except Exception as e:
                    # 打印错误信息
                    print('Error', e)
                    # 刷新索引以确保数据被索引
                    es.indices.refresh(index=index_name)
                    # 等待一段时间后重试
                    time.sleep(num_attempt + 1)
        # 完成索引后刷新索引
        es.indices.refresh(index=index_name)
    else:
        # 如果索引已经存在，则跳过索引过程
        print(f'Index {index_name} already exists, skipping indexing.')


# 搜索指定索引中的文档，返回文档 ID 列表
def search(es, index_name, query, top_k):
    # 构建搜索查询
    search_query = {
        "size": top_k,
        "query": {
            "match": {
                "content": query
            }
        }
    }

    # 执行搜索
    search_results = es.search(index=index_name, body=search_query)
    # 返回文档 ID 列表
    return [hit['_id'] for hit in search_results['hits']['hits']]


# 搜索指定索引中的文档，返回文档 ID 和相关分数
def search_with_score(es, index_name, query, top_k):
    # 构建搜索查询
    search_query = {
        "size": top_k,
        "query": {
            "match": {
                "content": query
            }
        }
    }

    # 执行搜索
    search_results = es.search(index=index_name, body=search_query)
    # 获取搜索结果中的文档及其分数
    hits = search_results['hits']['hits']
    # 返回文档 ID 和分数的元组
    return [(hit['_id'], hit['_score']) for hit in hits]


# 搜索指定索引中的文档，返回文档 ID 和文档内容
def search_with_id_and_content(es, index_name, query, top_k):
    # 构建搜索查询
    res = es.search(index=index_name, body={"query": {"match": {"content": query}}}, size=top_k)
    # 返回文档 ID 和内容的元组
    return [(hit["_id"], hit["_source"]["content"]) for hit in res['hits']['hits']]


# 搜索指定索引中的文档，返回文档 ID、分数和文档内容
def search_with_id_score_and_content(es, index_name, query, top_k):
    # 构建搜索查询
    res = es.search(index=index_name, body={"query": {"match": {"content": query}}}, size=top_k)
    # 返回文档 ID、分数和内容的元组
    return [(hit["_id"], hit["_score"], hit["_source"]["content"]) for hit in res['hits']['hits']]


# 清除指定索引中的所有文档
def clear_index(es, index_name):
    es.delete_by_query(
        index=index_name,
        body={
            "query": {
                "match_all": {}
            }
        }
    )


# 搜索内容的函数尚未完成
def search_content(es, index_name, query, top_k):
    # 创建一个搜索查询字典，包含返回结果的数量和匹配条件
    search_query = {
        "size": top_k,  # 设置返回结果的数量为 top_k
        "query": {  # 定义查询条件
            "match": {  # 使用匹配查询
                "content": query  # 匹配字段为 'content'，查询内容为 query
            }
        }
    }

    # 执行搜索请求，传递查询字典和索引名称
    search_results = es.search(index=index_name, body=search_query)
    # 从搜索结果中提取每个命中的 'content' 字段，生成结果列表
    return [hit['_source']['content'] for hit in search_results['hits']['hits']]
# 定义一个函数，根据给定的查询在 Elasticsearch 中搜索内容，并返回带分数的前 top_k 个结果
def search_content_with_score(es, index_name, query, top_k):
    # 构造搜索查询的 JSON 对象，设置返回结果数量为 top_k
    search_query = {
        "size": top_k,
        "query": {
            "match": {
                "content": query
            }
        }
    }

    # 执行搜索请求，获取搜索结果
    search_results = es.search(index=index_name, body=search_query)
    # 提取结果中的每个命中的内容和对应的分数
    hits = search_results['hits']['hits']
    # 返回包含内容和分数的元组列表
    return [(hit['_source']['content'], hit['_score']) for hit in hits]


# 定义一个函数，根据给定的查询在 Elasticsearch 中滚动搜索所有匹配的内容，并返回带分数的结果
def score_all_with_scroll(es, index_name, query, scroll='2m', size=100):
    # 构造初始搜索查询的 JSON 对象，设置每次返回的结果数量为 size
    search_query = {
        "size": size,
        "query": {
            "match": {
                "content": query
            }
        }
    }

    # 执行初始搜索请求，设置滚动时间为 scroll
    search_results = es.search(index=index_name, body=search_query, scroll=scroll)
    # 提取初始结果中的每个命中的内容和对应的分数
    contents_scores = [(hit['_source']['content'], hit['_score']) for hit in search_results['hits']['hits']]

    # 开始滚动查询，直到没有更多结果
    while True:
        # 执行滚动请求，获取下一批结果
        res = es.scroll(scroll_id=search_results['_scroll_id'], scroll=scroll)
        # 提取当前滚动结果中的每个命中的内容和对应的分数
        hits = res['hits']['hits']
        # 如果没有更多结果，则退出循环
        if not hits:
            break
        # 将新获取的结果追加到列表中
        contents_scores.extend([(hit['_source']['content'], hit['_score']) for hit in hits])

    # 返回所有结果的内容和分数列表
    return contents_scores
```