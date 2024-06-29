# `.\numpy\doc\neps\tools\build_index.py`

```
"""
Scan the directory of nep files and extract their metadata.  The
metadata is passed to Jinja for filling out the toctrees for various NEP
categories.
"""

# 导入必要的模块
import os               # 提供与操作系统相关的功能
import jinja2           # 模板引擎
import glob             # 文件名匹配
import re               # 正则表达式操作

# 渲染函数，用于渲染 Jinja 模板
def render(tpl_path, context):
    # 拆分模板路径和文件名
    path, filename = os.path.split(tpl_path)
    # 返回渲染后的模板内容
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(path or './')
    ).get_template(filename).render(context)

# 提取 NEP 元数据的函数
def nep_metadata():
    # 忽略的文件名
    ignore = ('nep-template.rst')
    # 匹配所有符合 nep-*.rst 模式的文件，并按文件名排序
    sources = sorted(glob.glob(r'nep-*.rst'))
    # 过滤掉忽略的文件名
    sources = [s for s in sources if not s in ignore]

    # 元数据的正则表达式模式
    meta_re = r':([a-zA-Z\-]*): (.*)'

    # 是否存在 Provisional 类型的 NEP
    has_provisional = False
    # 存储 NEP 元数据的字典
    neps = {}

    # 打印加载的元数据信息
    print('Loading metadata for:')
    # 遍历每个 NEP 文件
    for source in sources:
        # 打印加载的文件名
        print(f' - {source}')
        # 提取 NEP 编号
        nr = int(re.match(r'nep-([0-9]{4}).*\.rst', source).group(1))

        # 打开 NEP 文件并读取每一行内容
        with open(source) as f:
            lines = f.readlines()
            # 提取每行中的标签信息
            tags = [re.match(meta_re, line) for line in lines]
            tags = [match.groups() for match in tags if match is not None]
            # 将标签信息转化为字典形式
            tags = {tag[0]: tag[1] for tag in tags}

            # 查找 NEP 标题所在行
            for i, line in enumerate(lines[:-1]):
                chars = set(line.rstrip())
                # 判断标题行的特征
                if len(chars) == 1 and ("=" in chars or "*" in chars):
                    break
            else:
                # 如果找不到 NEP 标题行，则引发运行时错误
                raise RuntimeError("Unable to find NEP title.")

            # 将 NEP 标题和文件名添加到标签信息中
            tags['Title'] = lines[i+1].strip()
            tags['Filename'] = source

        # 检查 NEP 标题是否以正确格式开始
        if not tags['Title'].startswith(f'NEP {nr} — '):
            raise RuntimeError(
                f'Title for NEP {nr} does not start with "NEP {nr} — " '
                '(note that — here is a special, elongated dash). Got: '
                f'    {tags["Title"]!r}')

        # 检查已接受、已拒绝或已撤回的 NEP 是否有解决方案标签
        if tags['Status'] in ('Accepted', 'Rejected', 'Withdrawn'):
            if not 'Resolution' in tags:
                raise RuntimeError(
                    f'NEP {nr} is Accepted/Rejected/Withdrawn but '
                    'has no Resolution tag'
                )
        # 如果 NEP 的状态是 Provisional，则设置标志位为 True
        if tags['Status'] == 'Provisional':
            has_provisional = True

        # 将 NEP 数据存入 neps 字典中
        neps[nr] = tags

    # 现在已经获取了所有 NEP 元数据，可以执行一些全局一致性检查
    # 遍历字典 neps，其中 nr 是 NEP 编号，tags 是 NEP 的标签字典
    for nr, tags in neps.items():
        # 检查 NEP 的状态是否为 'Superseded'（已废弃）
        if tags['Status'] == 'Superseded':
            # 如果 NEP 被标记为 'Superseded'，则检查是否存在 'Replaced-By' 标签
            if not 'Replaced-By' in tags:
                # 如果缺少 'Replaced-By' 标签，则抛出运行时错误，指明 NEP 已废弃但未指定替代的 NEP 编号
                raise RuntimeError(
                    f'NEP {nr} has been Superseded, but has no Replaced-By tag'
                )

            # 获取被替代的 NEP 编号，并转换为整数
            replaced_by = int(re.findall(r'\d+', tags['Replaced-By'])[0])
            # 获取替代 NEP 对象的标签信息
            replacement_nep = neps[replaced_by]

            # 检查替代 NEP 是否有 'Replaces' 标签
            if not 'Replaces' in replacement_nep:
                # 如果替代 NEP 缺少 'Replaces' 标签，则抛出运行时错误，指明当前 NEP 被替代但替代的 NEP 没有指定被替代的 NEP 编号
                raise RuntimeError(
                    f'NEP {nr} is superseded by {replaced_by}, but that NEP has no Replaces tag.'
                )

            # 检查当前 NEP 是否在替代 NEP 的 'Replaces' 标签中
            if nr not in parse_replaces_metadata(replacement_nep):
                # 如果当前 NEP 不在替代 NEP 的 'Replaces' 标签中，则抛出运行时错误，指明当前 NEP 被替代但被替代 NEP 的 'Replaces' 标签指定了不正确的 NEP 编号
                raise RuntimeError(
                    f'NEP {nr} is superseded by {replaced_by}, but that NEP has a Replaces tag of `{replacement_nep['Replaces']}`.'
                )

        # 如果当前 NEP 存在 'Replaces' 标签
        if 'Replaces' in tags:
            # 解析当前 NEP 的 'Replaces' 标签，并遍历每个被替代的 NEP 编号
            replaced_neps = parse_replaces_metadata(tags)
            for nr_replaced in replaced_neps:
                # 获取被替代 NEP 对象的标签信息
                replaced_nep_tags = neps[nr_replaced]
                # 检查被替代 NEP 的状态是否为 'Superseded'
                if not replaced_nep_tags['Status'] == 'Superseded':
                    # 如果被替代 NEP 的状态不是 'Superseded'，则抛出运行时错误，指明当前 NEP 替代了一个未被设置为 'Superseded' 的 NEP
                    raise RuntimeError(
                        f'NEP {nr} replaces NEP {nr_replaced}, but that NEP has not been set to Superseded'
                    )

    # 返回包含更新后的 NEP 数据字典和是否存在临时 NEP 的标志的字典
    return {'neps': neps, 'has_provisional': has_provisional}
# 定义函数，处理替换的元数据中的 :Replaces: 字段，返回替换后的 NEP 编号列表
def parse_replaces_metadata(replacement_nep):
    """Handle :Replaces: as integer or list of integers"""
    # 使用正则表达式查找替换的 NEP 编号，返回匹配到的所有数字字符串列表
    replaces = re.findall(r'\d+', replacement_nep['Replaces'])
    # 将匹配到的数字字符串列表转换为整数列表
    replaced_neps = [int(s) for s in replaces]
    # 返回替换后的 NEP 编号列表
    return replaced_neps

# 调用 nep_metadata() 函数获取 NEP 元数据
meta = nep_metadata()

# 遍历给定的 NEP 类别列表
for nepcat in (
    "provisional", "accepted", "deferred", "finished", "meta",
    "open", "rejected",
):
    # 构建输入文件名和输出文件名
    infile = f"{nepcat}.rst.tmpl"
    outfile = f"{nepcat}.rst"

    # 打印编译信息，指示正在处理哪个模板文件到哪个输出文件
    print(f'Compiling {infile} -> {outfile}')
    
    # 调用 render 函数，生成指定模板文件的内容
    genf = render(infile, meta)
    
    # 打开输出文件，写入生成的内容
    with open(outfile, 'w') as f:
        f.write(genf)
```