# `.\pytorch\test\test_metal.py`

```
# Owner(s): ["oncall: mobile"]

import torch  # 导入 PyTorch 库
from torch.nn import functional as F  # 导入 torch.nn 的 functional 模块并重命名为 F

from torch.testing._internal.common_utils import TestCase, run_tests  # 导入测试相关的工具类和函数
from torch.testing import FileCheck  # 导入文件检查工具
import io  # 导入 io 模块

class TestMetalRewritePass(TestCase):  # 定义测试类 TestMetalRewritePass，继承自 TestCase 类
    @staticmethod
    def validate_transformed_module(  # 定义静态方法 validate_transformed_module，用于验证转换后的模型
            # 用于 flake 工具的占位符
            self,  # self 参数，引用类实例
            pattern_count_map,  # 模式计数映射，用于指定要检查的模式及其出现次数
            data_shape,  # 数据形状，输入数据的形状
            prepack_removal=False,  # 是否移除预打包操作的标志，默认为 False
            fuse_clamping_ops=False):  # 是否融合夹紧操作的标志，默认为 False
        module_instance = self  # 将 self 赋值给 module_instance，即当前类实例
        scripted_model = torch.jit.script(module_instance)  # 对模块实例进行脚本化
        scripted_model.eval()  # 将脚本化模型设为评估模式
        input_data = torch.normal(1, 20, size=data_shape)  # 生成正态分布的随机数据作为输入数据
        ref_result = scripted_model(input_data)  # 对脚本化模型进行推理，得到参考结果
        torch._C._jit_pass_metal_insert_prepacked_ops(scripted_model._c)  # 插入预打包操作
        if fuse_clamping_ops or prepack_removal:  # 如果需要融合夹紧操作或者移除预打包操作
            scripted_model._c = torch._C._freeze_module(scripted_model._c)  # 冻结模块以应用优化
        if fuse_clamping_ops:  # 如果需要融合夹紧操作
            torch._C._jit_pass_metal_fuse_clamp_w_prepacked_conv(scripted_model._c)  # 融合夹紧操作与预打包的卷积
        if prepack_removal:  # 如果需要移除预打包操作
            torch._C._jit_pass_metal_fold_prepacking_ops(scripted_model._c)  # 折叠预打包操作
        buffer = io.BytesIO()  # 创建一个字节流缓冲区
        torch.jit.save(scripted_model, buffer)  # 将脚本化模型保存到字节流缓冲区
        buffer.seek(0)  # 将缓冲区的读写位置移动到开头
        deserialized_scripted_model = torch.jit.load(buffer)  # 从字节流缓冲区加载模型
        for pattern, v in pattern_count_map.items():  # 遍历模式计数映射中的每个模式及其对应的计数
            if v == 0:
                FileCheck().check(pattern).run(deserialized_scripted_model.graph)  # 检查模型图中是否存在指定的模式
            elif v == -1:
                FileCheck().check_not(pattern).run(deserialized_scripted_model.graph)  # 检查模型图中是否不存在指定的模式
            else:
                FileCheck().check_count(pattern, v, exactly=True).run(deserialized_scripted_model.graph)  # 检查模型图中指定模式的出现次数是否符合预期

if __name__ == "__main__":  # 如果当前脚本作为主程序执行
    run_tests()  # 执行测试
```