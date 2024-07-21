# `.\pytorch\test\fx\test_fx_node_hook.py`

```py
# Owner(s): ["module: fx"]
# 导入 torch 库和相关的模块
import torch
from torch.fx import symbolic_trace
from torch.testing._internal.common_utils import TestCase

# 定义测试类 TestFXNodeHook，继承自 TestCase 类
class TestFXNodeHook(TestCase):
    # 定义测试方法 test_hooks_for_node_update
    def test_hooks_for_node_update(self):
        # 声明全局变量用于标记钩子是否被调用过
        global create_node_hook1_called
        global create_node_hook2_called
        global erase_node_hook1_called
        global erase_node_hook2_called

        # 初始化全局变量为 False
        create_node_hook1_called = False
        create_node_hook2_called = False
        erase_node_hook1_called = False
        erase_node_hook2_called = False

        # 定义一个简单的函数 fn，接受三个参数并进行操作
        def fn(a, b, c):
            # 使用 torch.nn.functional.linear 函数进行线性操作
            x = torch.nn.functional.linear(a, b)
            # 将结果与 c 相加
            x = x + c
            # 返回 x 的余弦值
            return x.cos()

        # 定义四个钩子函数，用于跟踪创建和删除节点的操作
        def create_node_hook1(node):
            global create_node_hook1_called
            create_node_hook1_called = True

        def create_node_hook2(node):
            global create_node_hook2_called
            create_node_hook2_called = True

        def erase_node_hook1(node):
            global erase_node_hook1_called
            erase_node_hook1_called = True

        def erase_node_hook2(node):
            global erase_node_hook2_called
            erase_node_hook2_called = True

        # 对函数 fn 进行符号跟踪，生成图模块 gm
        gm = symbolic_trace(fn)

        # 向 gm 注册创建节点的钩子函数
        gm._register_create_node_hook(create_node_hook1)
        gm._register_create_node_hook(create_node_hook2)

        # 向 gm 注册删除节点的钩子函数
        gm._register_erase_node_hook(erase_node_hook1)
        gm._register_erase_node_hook(erase_node_hook2)

        # 获取 gm 的图结构
        graph = gm.graph
        node_a = None
        # 在图中查找操作为 "placeholder" 的节点，找到第一个并赋给 node_a
        for node in graph.find_nodes(op="placeholder"):
            node_a = node
            break

        # 断言 node_a 不为 None
        assert node_a is not None

        # 复制 node_a 节点，生成 node_a_copy
        node_a_copy = graph.node_copy(node_a)
        # 将 node_a 的所有使用替换为 node_a_copy
        node_a.replace_all_uses_with(node_a_copy)
        # 从图中删除 node_a 节点
        graph.erase_node(node_a)

        # 断言四个钩子函数均已调用过
        assert (
            create_node_hook1_called
            and create_node_hook2_called
            and erase_node_hook1_called
            and erase_node_hook2_called
        )

        # 从 gm 中注销创建节点的钩子函数
        gm._unregister_create_node_hook(create_node_hook1)
        gm._unregister_create_node_hook(create_node_hook2)
        
        # 从 gm 中注销删除节点的钩子函数
        gm._unregister_erase_node_hook(erase_node_hook1)
        gm._unregister_erase_node_hook(erase_node_hook2)

        # 断言 gm 的创建节点钩子列表和删除节点钩子列表为空
        assert gm._create_node_hooks == []
        assert gm._erase_node_hooks == []
```