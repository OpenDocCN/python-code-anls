# `.\pytorch\test\distributed\_spmd\test_graph_utils.py`

```
# 导入操作系统模块
import os

# 导入用于图形工具的函数
from torch.distributed._spmd.graph_utils import dump_graphs_to_files
# 导入用于运行测试的工具函数（忽略类型检查错误）
from torch.testing._internal.common_utils import run_tests  # noqa: TCH001
# 导入分布式张量测试基类
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase


# 图形工具测试类，继承自分布式张量测试基类
class GraphUtilsTest(DTensorTestBase):
    
    # 定义属性world_size，返回1
    @property
    def world_size(self):
        return 1

    # 测试函数：测试dump_graphs_to_files函数的功能
    def test_dump_graphs(self):
        
        # 定义一个虚拟的图形类FakeGraph
        class FakeGraph:
            def __init__(self, postfix):
                self.graph = f"fake graph str {postfix}"

            def __str__(self) -> str:
                return self.graph
        
        # 创建一个虚拟图形字典fake_graph1
        fake_graph1 = {"fake_graph1": FakeGraph(1)}
        
        # 调用dump_graphs_to_files函数，将fake_graph1中的内容写入文件夹folder
        folder = dump_graphs_to_files(fake_graph1)
        
        # 创建另一个虚拟图形字典fake_graph2
        fake_graph2 = {"fake_graph2": FakeGraph(1)}
        
        # 再次调用dump_graphs_to_files函数，将fake_graph2中的内容写入文件夹folder中的新位置new_folder
        new_folder = dump_graphs_to_files(fake_graph2, folder)
        
        # 断言：验证folder和new_folder是否相同
        self.assertEqual(folder, new_folder)

        # 循环处理fake_graph1和fake_graph2中的图形
        for i in (1, 2):
            # 构建文件路径
            path = os.path.join(folder, f"fake_graph{i}.graph")
            
            # 断言：验证文件路径是否存在
            self.assertTrue(os.path.exists(path))
            
            # 打开文件，读取内容进行断言比较
            with open(path) as fp:
                # 根据i的值选择对应的fake_graph字典进行比较
                fake_graph = fake_graph1 if i == 1 else fake_graph2
                self.assertEqual(fp.readline(), fake_graph[f"fake_graph{i}"].graph)
            
            # 删除测试用的文件
            os.remove(path)
        
        # 删除测试用的文件夹
        os.rmdir(folder)


# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 如果条件为False，则运行测试
    if False:
        run_tests()
```