# `.\HippoRAG\src\analysis\kg_vis.py`

```py
# 导入命令行解析模块和序列化模块
import argparse
import pickle

# 确保只有在直接运行该脚本时才执行以下代码
if __name__ == '__main__':
    # 创建 ArgumentParser 对象用于解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加一个命令行参数 '--kg'，用于指定文件路径，默认为给定路径
    parser.add_argument('--kg', type=str,
                        default='output/musique_facts_and_sim_graph_relation_dict_ents_only_lower_preprocess_ner_colbertv2.v3.subset.p')
    # 解析命令行参数
    args = parser.parse_args()

    # 使用 pickle 加载指定路径的序列化对象
    kg = pickle.load(open(args.kg, 'rb'))
    # 打印加载的对象长度
    print(len(kg))

    # 初始化一个空列表，用于存储过滤后的结果
    res = []
    # 遍历 kg 字典中的每一对 (头，尾) 关系
    for head_tail in kg:
        # 检查头或尾是否包含特定城市名
        if head_tail[0].lower() in ['alhandra', 'vila franca de xira'] or head_tail[1].lower() in ['alhandra', 'vila franca de xira']:
            # 如果匹配，将头、尾和关系存入变量
            head = head_tail[0]
            tail = head_tail[1]
            relation = kg[head_tail]
            # 将 (头，关系，尾) 元组添加到结果列表
            res.append((head, relation, tail))

    # 打印过滤后的结果列表
    print(res)
```