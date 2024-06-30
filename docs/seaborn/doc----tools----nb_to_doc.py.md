# `D:\src\scipysrc\seaborn\doc\tools\nb_to_doc.py`

```
def pop_recursive(d, key, default=None):
    """dict.pop(key) where `key` is a `.`-delimited list of nested keys.
    >>> d = {'a': {'b': 1, 'c': 2}}
    >>> pop_recursive(d, 'a.c')
    2
    >>> d
    {'a': {'b': 1}}
    """
    # 将键按照点号分割为嵌套键列表
    nested = key.split('.')
    # 从字典 `d` 的根部开始遍历嵌套键
    current = d
    for k in nested[:-1]:
        # 如果当前对象具有 `get` 方法，则获取下一级嵌套键的值，否则返回默认值
        if hasattr(current, 'get'):
            current = current.get(k, {})
        else:
            return default
    # 如果当前对象没有 `pop` 方法，则返回默认值
    if not hasattr(current, 'pop'):
        return default
    # 弹出最内层的键值对，并返回其值，默认为 `default`
    return current.pop(nested[-1], default)


def strip_output(nb):
    """
    Strip the outputs, execution count/prompt number and miscellaneous
    metadata from a notebook object, unless specified to keep either the
    outputs or counts.
    """
    # 定义需要移除的元数据字段
    keys = {'metadata': [], 'cell': {'metadata': ["execution"]}}

    # 移除特定的元数据字段
    nb.metadata.pop('signature', None)
    nb.metadata.pop('widgets', None)

    # 对于需要移除的字段，递归地从 `nb.metadata` 中移除
    for field in keys['metadata']:
        pop_recursive(nb.metadata, field)

    # 如果环境变量中存在 `NB_KERNEL`，则更新 `kernelspec` 元数据
    if 'NB_KERNEL' in os.environ:
        nb.metadata['kernelspec']['name'] = os.environ['NB_KERNEL']
        nb.metadata['kernelspec']['display_name'] = os.environ['NB_KERNEL']
    `
        # 遍历笔记本中的每个单元格
        for cell in nb.cells:
            # 如果单元格中包含 'outputs' 属性，则将其设为空列表
            if 'outputs' in cell:
                cell['outputs'] = []
            # 如果单元格中包含 'prompt_number' 属性，则将其设为 None
            if 'prompt_number' in cell:
                cell['prompt_number'] = None
            # 如果单元格中包含 'execution_count' 属性，则将其设为 None
            if 'execution_count' in cell:
                cell['execution_count'] = None
    
            # 总是移除这些元数据
            # 遍历输出样式列表 ['collapsed', 'scrolled']
            for output_style in ['collapsed', 'scrolled']:
                # 如果单元格的元数据中包含当前输出样式，则将其设为 False
                if output_style in cell.metadata:
                    cell.metadata[output_style] = False
    
            # 如果单元格中包含 'metadata' 属性
            if 'metadata' in cell:
                # 遍历要移除的字段列表 ['collapsed', 'scrolled', 'ExecuteTime']
                for field in ['collapsed', 'scrolled', 'ExecuteTime']:
                    # 从单元格的元数据中移除当前字段
                    cell.metadata.pop(field, None)
    
            # 遍历预定义的键 'keys['cell']' 中的每一项
            for (extra, fields) in keys['cell'].items():
                # 如果当前单元格中包含当前额外的键
                if extra in cell:
                    # 遍历要移除的字段列表 fields
                    for field in fields:
                        # 递归地移除单元格中的指定字段
                        pop_recursive(getattr(cell, extra), field)
    
        # 返回处理后的笔记本对象
        return nb
if __name__ == "__main__":
    # 获取指定的 ipynb 文件路径并解析为各个组件
    _, fpath, outdir = sys.argv
    # 将文件路径分割为基本目录和文件名
    basedir, fname = os.path.split(fpath)
    # 从文件名中去掉末尾的 ".ipynb" 后缀，得到文件名的基本部分
    fstem = fname[:-6]

    # 读取 notebook 文件
    with open(fpath) as f:
        nb = nbformat.read(f, as_version=4)

    # 运行 notebook
    kernel = os.environ.get("NB_KERNEL", None)
    if kernel is None:
        kernel = nb["metadata"]["kernelspec"]["name"]
    # 创建执行预处理器对象，设置超时时间和内核名称，并传递额外参数
    ep = ExecutePreprocessor(
        timeout=600,
        kernel_name=kernel,
        extra_arguments=["--InlineBackend.rc=figure.dpi=88"]
    )
    # 预处理 notebook，设置路径为基本目录
    ep.preprocess(nb, {"metadata": {"path": basedir}})

    # 移除纯文本执行结果输出
    for cell in nb.get("cells", {}):
        if "show-output" in cell["metadata"].get("tags", []):
            continue
        fields = cell.get("outputs", [])
        for field in fields:
            if field["output_type"] == "execute_result":
                data_keys = field["data"].keys()
                for key in list(data_keys):
                    if key == "text/plain":
                        field["data"].pop(key)
                if not field["data"]:
                    fields.remove(field)

    # 将 notebook 转换为 .rst 格式
    exp = RSTExporter()

    # 配置 TagRemovePreprocessor 和 ExtractOutputPreprocessor
    c = Config()
    c.TagRemovePreprocessor.remove_cell_tags = {"hide"}
    c.TagRemovePreprocessor.remove_input_tags = {"hide-input"}
    c.TagRemovePreprocessor.remove_all_outputs_tags = {"hide-output"}
    c.ExtractOutputPreprocessor.output_filename_template = \
        f"{fstem}_files/{fstem}_" + "{cell_index}_{index}{extension}"

    # 注册预处理器并设置优先级为最高
    exp.register_preprocessor(TagRemovePreprocessor(config=c), True)
    exp.register_preprocessor(ExtractOutputPreprocessor(config=c), True)

    # 使用 exporter 将 notebook 节点转换为 body 和 resources
    body, resources = exp.from_notebook_node(nb)

    # 清除 notebook 上的输出并将其保存回磁盘为 .ipynb 文件
    nb = strip_output(nb)
    with open(fpath, "wt") as f:
        nbformat.write(nb, f)

    # 写入 .rst 文件
    rst_path = os.path.join(outdir, f"{fstem}.rst")
    with open(rst_path, "w") as f:
        f.write(body)

    # 写入各个图片输出
    imdir = os.path.join(outdir, f"{fstem}_files")
    if not os.path.exists(imdir):
        os.mkdir(imdir)

    for imname, imdata in resources["outputs"].items():
        if imname.startswith(fstem):
            impath = os.path.join(outdir, f"{imname}")
            with open(impath, "wb") as f:
                f.write(imdata)
```