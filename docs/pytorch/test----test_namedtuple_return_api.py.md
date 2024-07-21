# `.\pytorch\test\test_namedtuple_return_api.py`

```
# 导入标准库 os、正则表达式模块 re、YAML 解析模块 yaml、文本包装模块 textwrap、PyTorch 深度学习库
import os
import re
import yaml
import textwrap
import torch

# 从 PyTorch 内部测试工具包中导入 TestCase 类和运行测试的辅助函数 run_tests
from torch.testing._internal.common_utils import TestCase, run_tests
# 从标准库 collections 中导入 namedtuple 类
from collections import namedtuple

# 获取当前脚本文件的目录路径
path = os.path.dirname(os.path.realpath(__file__))
# 构建 ATen 本地函数定义 YAML 文件的路径
aten_native_yaml = os.path.join(path, '../aten/src/ATen/native/native_functions.yaml')

# 定义返回值为命名元组的运算符集合
all_operators_with_namedtuple_return = {
    'max', 'min', 'aminmax', 'median', 'nanmedian', 'mode', 'kthvalue', 'svd',
    'qr', 'geqrf', 'slogdet', 'sort', 'topk', 'linalg_inv_ex',
    'triangular_solve', 'cummax', 'cummin', 'linalg_eigh', "_linalg_eigh", "_unpack_dual", 'linalg_qr',
    'linalg_svd', '_linalg_svd', 'linalg_slogdet', '_linalg_slogdet', 'fake_quantize_per_tensor_affine_cachemask',
    'fake_quantize_per_channel_affine_cachemask', 'linalg_lstsq', 'linalg_eig', 'linalg_cholesky_ex',
    'frexp', 'lu_unpack', 'histogram', 'histogramdd',
    '_fake_quantize_per_tensor_affine_cachemask_tensor_qparams',
    '_fused_moving_avg_obs_fq_helper', 'linalg_lu_factor', 'linalg_lu_factor_ex', 'linalg_lu',
    '_linalg_det', '_lu_with_info', 'linalg_ldl_factor_ex', 'linalg_ldl_factor', 'linalg_solve_ex', '_linalg_solve_ex'
}

# 定义跳过命名元组返回值的运算符集合
all_operators_with_namedtuple_return_skip_list = {
    '_scaled_dot_product_flash_attention',
    '_scaled_dot_product_fused_attention_overrideable',
    '_scaled_dot_product_flash_attention_for_cpu',
    '_scaled_dot_product_efficient_attention',
    '_scaled_dot_product_cudnn_attention',
}

# 定义测试类 TestNamedTupleAPI，继承自 TestCase
class TestNamedTupleAPI(TestCase):

    # 定义测试方法 test_import_return_types
    def test_import_return_types(self):
        # 引入 torch.return_types 模块（但未使用），忽略 F401 类型的 flake8 错误
        import torch.return_types  # noqa: F401
        # 使用 exec 函数动态引入 torch.return_types 模块中的所有内容（未明确列出）
        exec('from torch.return_types import *')
    # 定义测试函数 `test_native_functions_yaml`，用于测试某些条件
    def test_native_functions_yaml(self):
        # 用于存储找到的运算符集合
        operators_found = set()
        # 编译正则表达式，匹配函数名称和操作符
        regex = re.compile(r"^(\w*)(\(|\.)")
        # 打开并读取 `aten_native_yaml` 文件，安全加载其内容
        with open(aten_native_yaml) as file:
            # 遍历 YAML 文件中的每个条目
            for f in yaml.safe_load(file.read()):
                # 获取条目中的 `func` 字段
                f = f['func']
                # 根据箭头分隔符 `->` 提取函数返回值部分
                ret = f.split('->')[1].strip()
                # 使用正则表达式获取函数名称
                name = regex.findall(f)[0][0]
                # 如果函数名称在 `all_operators_with_namedtuple_return` 中，则加入集合
                if name in all_operators_with_namedtuple_return:
                    operators_found.add(name)
                    continue
                # 如果函数名称以 `_backward` 结尾或者 `_forward` 结尾，则跳过
                if '_backward' in name or name.endswith('_forward'):
                    continue
                # 如果返回值不以 `(` 开头，则跳过
                if not ret.startswith('('):
                    continue
                # 如果返回值为 `()`，则跳过
                if ret == '()':
                    continue
                # 如果函数名称在 `all_operators_with_namedtuple_return_skip_list` 中，则跳过
                if name in all_operators_with_namedtuple_return_skip_list:
                    continue
                # 去掉返回值最外层的括号，并按逗号分隔获取每个返回值部分
                ret = ret[1:-1].split(',')
                # 遍历每个返回值部分
                for r in ret:
                    # 去除首尾空格
                    r = r.strip()
                    # 断言每个返回值部分只包含一个单词，用于限制白名单内允许的操作符
                    self.assertEqual(len(r.split()), 1, 'only allowlisted '
                                     'operators are allowed to have named '
                                     'return type, got ' + name)
        # 断言 `all_operators_with_namedtuple_return` 和找到的运算符集合相等
        self.assertEqual(all_operators_with_namedtuple_return, operators_found, textwrap.dedent("""
        Some elements in the `all_operators_with_namedtuple_return` of test_namedtuple_return_api.py
        could not be found. Do you forget to update test_namedtuple_return_api.py after renaming some
        operator?
        """))
# 如果当前脚本被直接运行（而不是被导入到其它模块中执行），则执行以下代码块
if __name__ == '__main__':
    # 调用名为 run_tests 的函数，用于执行测试代码或程序的测试套件
    run_tests()
```