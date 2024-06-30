# `D:\src\scipysrc\sympy\sympy\polys\tests\test_injections.py`

```
# 导入所需模块和函数，用于测试全局命名空间中符号的注入
from sympy.polys.rings import vring  # 导入 vring 函数
from sympy.polys.fields import vfield  # 导入 vfield 函数
from sympy.polys.domains import QQ  # 导入 QQ

# 定义测试函数，用于测试 vring 函数的行为
def test_vring():
    ns = {'vring':vring, 'QQ':QQ}  # 创建命名空间字典，包含 vring 和 QQ
    exec('R = vring("r", QQ)', ns)  # 在命名空间中执行创建环的操作，将结果存储在 R 中
    exec('assert r == R.gens[0]', ns)  # 在命名空间中执行断言，验证 r 是否等于 R 的第一个生成元素

    exec('R = vring("rb rbb rcc rzz _rx", QQ)', ns)  # 创建多个生成元素的环，并将结果存储在 R 中
    exec('assert rb == R.gens[0]', ns)  # 验证 rb 是否等于 R 的第一个生成元素
    exec('assert rbb == R.gens[1]', ns)  # 验证 rbb 是否等于 R 的第二个生成元素
    exec('assert rcc == R.gens[2]', ns)  # 验证 rcc 是否等于 R 的第三个生成元素
    exec('assert rzz == R.gens[3]', ns)  # 验证 rzz 是否等于 R 的第四个生成元素
    exec('assert _rx == R.gens[4]', ns)  # 验证 _rx 是否等于 R 的第五个生成元素

    exec('R = vring(["rd", "re", "rfg"], QQ)', ns)  # 创建带有列表生成元素的环，并将结果存储在 R 中
    exec('assert rd == R.gens[0]', ns)  # 验证 rd 是否等于 R 的第一个生成元素
    exec('assert re == R.gens[1]', ns)  # 验证 re 是否等于 R 的第二个生成元素
    exec('assert rfg == R.gens[2]', ns)  # 验证 rfg 是否等于 R 的第三个生成元素

# 定义测试函数，用于测试 vfield 函数的行为
def test_vfield():
    ns = {'vfield':vfield, 'QQ':QQ}  # 创建命名空间字典，包含 vfield 和 QQ
    exec('F = vfield("f", QQ)', ns)  # 在命名空间中执行创建域的操作，将结果存储在 F 中
    exec('assert f == F.gens[0]', ns)  # 在命名空间中执行断言，验证 f 是否等于 F 的第一个生成元素

    exec('F = vfield("fb fbb fcc fzz _fx", QQ)', ns)  # 创建多个生成元素的域，并将结果存储在 F 中
    exec('assert fb == F.gens[0]', ns)  # 验证 fb 是否等于 F 的第一个生成元素
    exec('assert fbb == F.gens[1]', ns)  # 验证 fbb 是否等于 F 的第二个生成元素
    exec('assert fcc == F.gens[2]', ns)  # 验证 fcc 是否等于 F 的第三个生成元素
    exec('assert fzz == F.gens[3]', ns)  # 验证 fzz 是否等于 F 的第四个生成元素
    exec('assert _fx == F.gens[4]', ns)  # 验证 _fx 是否等于 F 的第五个生成元素

    exec('F = vfield(["fd", "fe", "ffg"], QQ)', ns)  # 创建带有列表生成元素的域，并将结果存储在 F 中
    exec('assert fd == F.gens[0]', ns)  # 验证 fd 是否等于 F 的第一个生成元素
    exec('assert fe == F.gens[1]', ns)  # 验证 fe 是否等于 F 的第二个生成元素
    exec('assert ffg == F.gens[2]', ns)  # 验证 ffg 是否等于 F 的第三个生成元素
```