# `.\lucidrains\tf-bind-transformer\tf_bind_transformer\gene_utils.py`

```py
# 用于获取转录因子序列的代码

# 定义基因标识映射，将'RXR'映射为'RXRA'
GENE_IDENTIFIER_MAP = {
    'RXR': 'RXRA'
}

# 包含连字符的基因名称集合
NAMES_WITH_HYPHENS = {
    'NKX3-1',
    'NKX2-1',
    'NKX2-5',
    'SS18-SSX'
}

# 解析基因名称的函数
def parse_gene_name(name):
    # 如果名称中不包含连字符或者名称在NAMES_WITH_HYPHENS中，则直接返回名称
    if '-' not in name or name in NAMES_WITH_HYPHENS:
        name = GENE_IDENTIFIER_MAP.get(name, name)

        # 如果名称中包含下划线，则只搜索下划线左侧的目标因子名称
        if '_' in name:
            name, *_ = name.split('_')

        return (name,)

    # 如果名称中包含连字符，则按照一定规则解析名称
    first, *rest = name.split('-')

    parsed_rest = []

    for name in rest:
        if len(name) == 1:
            name = f'{first[:-1]}{name}'
        parsed_rest.append(name)

    return tuple([first, *parsed_rest])
```