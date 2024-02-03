# `.\PaddleOCR\ppocr\data\imaug\vqa\token\vqa_token_relation.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 定义 VQAReTokenRelation 类
class VQAReTokenRelation(object):
    # 初始化方法，接受任意关键字参数
    def __init__(self, **kwargs):
        # pass 表示不执行任何操作，保持方法的完整性
        pass
    # 定义一个方法，用于构建实体之间的关系
    def __call__(self, data):
        """
        build relations
        """
        # 从数据中获取实体和关系信息
        entities = data['entities']
        relations = data['relations']
        # 从数据中获取并移除 id 到标签的映射关系、空实体和实体 id 到索引的映射关系
        id2label = data.pop('id2label')
        empty_entity = data.pop('empty_entity')
        entity_id_to_index_map = data.pop('entity_id_to_index_map')

        # 去除重复的关系
        relations = list(set(relations))
        # 过滤掉包含空实体的关系
        relations = [
            rel for rel in relations
            if rel[0] not in empty_entity and rel[1] not in empty_entity
        ]
        kv_relations = []
        # 遍历关系，构建键值对关系列表
        for rel in relations:
            pair = [id2label[rel[0]], id2label[rel[1]]]
            if pair == ["question", "answer"]:
                kv_relations.append({
                    "head": entity_id_to_index_map[rel[0]],
                    "tail": entity_id_to_index_map[rel[1]]
                })
            elif pair == ["answer", "question"]:
                kv_relations.append({
                    "head": entity_id_to_index_map[rel[1]],
                    "tail": entity_id_to_index_map[rel[0]]
                })
            else:
                continue
        # 根据头实体索引排序关系列表，并添加关系的起始和结束索引
        relations = sorted(
            [{
                "head": rel["head"],
                "tail": rel["tail"],
                "start_index": self.get_relation_span(rel, entities)[0],
                "end_index": self.get_relation_span(rel, entities)[1],
            } for rel in kv_relations],
            key=lambda x: x["head"], )

        # 更新数据中的关系信息
        data['relations'] = relations
        return data

    # 获取关系的起始和结束索引
    def get_relation_span(self, rel, entities):
        bound = []
        for entity_index in [rel["head"], rel["tail"]]:
            bound.append(entities[entity_index]["start"])
            bound.append(entities[entity_index]["end"])
        return min(bound), max(bound)
```