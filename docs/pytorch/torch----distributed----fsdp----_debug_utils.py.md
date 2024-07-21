# `.\pytorch\torch\distributed\fsdp\_debug_utils.py`

```
    """
    用于可组合的 fully_shard() 代码路径，返回以下两个元素的元组：
      1. sharded module tree info: 每一行表示一个子模块名称，包含子模块的全限定名（FQN）和子模块类名。
         如果子模块由 `fully_shard` 进行分片，子模块名称将以 ' FULLY SHARDED' 后缀添加。
         每增加一个树级别，在打印名称之前会增加 4 个空格。例如，对于一个玩具模型，打印出的分片模块树信息如下：
    """
    # 获取模型的分片模块树信息，包括每个子模块的全限定名到子模块名称列表的映射
    # 返回的元组包含模块树信息的描述字符串和子模块名称到全限定名列表的字典
    model: torch.nn.Module,
) -> Tuple[str, Dict[str, List[str]]]:
    """
    def module_fn(
        module, prefix, tree_level, sharded_tree_info, sharded_module_name_to_fqns
    ):
        # 计算当前模块名前的缩进空格数
        num_spaces = tree_level * 4
        # 如果前缀以点结尾且长度大于0，则去掉最后一个点，否则保持前缀不变
        trimed_prefix = (
            prefix[:-1] if (len(prefix) > 0 and prefix[-1] == ".") else prefix
        )
        # 组装当前模块的完整名字，包括类名，并加上当前缩进空格数的空格
        prefixed_module_name = trimed_prefix + "[" + module.__class__.__name__ + "]"
        printed_prefixed_module_name = " " * num_spaces + prefixed_module_name
    
        # 获取模块的 FSDP 状态
        state = _get_module_fsdp_state(module)
        # 如果状态为 None，则将模块名加入到完整的分片树信息中，并返回
        if state is None:
            sharded_tree_info[0] += printed_prefixed_module_name + "\n"
            return
    
        # 获取模块的完全分片句柄
        handle = state._fully_sharded_module_to_handle.get(module, None)
    
        # 如果存在分片句柄，则在完整的分片树信息中记录模块名及其完全分片状态
        if handle:
            sharded_tree_info[0] += (
                printed_prefixed_module_name + " FULLY SHARDED" + "\n"
            )
        else:
            sharded_tree_info[0] += printed_prefixed_module_name + "\n"
    
        # 如果存在分片句柄，则处理模块的全局 FQN（全限定名）
        if handle:
            param = handle.flat_param
            assert isinstance(param, flat_param_file.FlatParameter)
            # 生成全局 FQN 列表，包括了顶级模型 `model` 的前缀 `prefix`
            global_fqns = [
                clean_tensor_name(prefix + name) for name in param._fqns
            ]  # 从顶级 `model` 开始前缀化
    
            # 将模块名及其对应的全局 FQN 列表加入到分片模块名到 FQN 的映射字典中
            if prefixed_module_name in sharded_module_name_to_fqns:
                sharded_module_name_to_fqns[prefixed_module_name].extend(global_fqns)
            else:
                sharded_module_name_to_fqns[prefixed_module_name] = global_fqns
    
    # 返回完整的分片树信息和分片模块名到 FQN 的映射字典
    def return_fn(sharded_tree_info, sharded_module_name_to_fqns):
        return sharded_tree_info[0], sharded_module_name_to_fqns
    # 使用 List 来在运行递归函数时原地修改其值
    sharded_tree_info: List[str] = [
        "",  # 初始化一个空字符串作为列表的第一个元素
    ]
    # 初始化一个空的字典，用于存储模块名到完全限定名列表的映射关系
    sharded_module_name_to_fqns: Dict[str, List[str]] = {}
    
    # 调用 _apply_to_modules 函数，并传入以下参数：
    # - model: 主要模型对象
    # - module_fn: 模块函数
    # - return_fn: 返回函数
    # - [key for key, _ in model.named_parameters()]: 模型中所有命名参数的列表
    # - sharded_tree_info: 用于递归函数中保存树信息的列表
    # - sharded_module_name_to_fqns: 模块名到完全限定名列表的映射字典
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [key for key, _ in model.named_parameters()],
        sharded_tree_info,
        sharded_module_name_to_fqns,
    )
```