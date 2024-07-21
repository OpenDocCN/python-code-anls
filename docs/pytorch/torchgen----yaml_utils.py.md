# `.\pytorch\torchgen\yaml_utils.py`

```py
# 安全加载快速的 C Yaml 加载器/转储器（如果可用）
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    # 如果快速加载器不可用，则回退到普通的安全加载器
    from yaml import SafeLoader as Loader  # type: ignore[assignment, misc]

try:
    from yaml import CSafeDumper as Dumper
except ImportError:
    # 如果快速转储器不可用，则回退到普通的安全转储器
    from yaml import SafeDumper as Dumper  # type: ignore[assignment, misc]

# 将 Dumper 赋值给 YamlDumper，便于后续使用
YamlDumper = Dumper

# 一个定制的 YAML 加载器，用于在发现重复键时报错。
# 默认情况下不会报错，请参考 https://github.com/yaml/pyyaml/issues/165
class YamlLoader(Loader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        # 初始化空的映射列表
        mapping = []
        # 遍历节点中的键值对
        for key_node, value_node in node.value:
            # 构造键对象
            key = self.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
            # 检查键是否已经存在于映射中，如果存在则抛出错误
            assert (
                key not in mapping
            ), f"Found a duplicate key in the yaml. key={key}, line={node.start_mark.line}"
            # 将键添加到映射中
            mapping.append(key)
        # 调用父类的构造映射方法
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        return mapping
```