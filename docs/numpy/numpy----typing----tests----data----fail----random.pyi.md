# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\random.pyi`

```
import numpy as np  # 导入 NumPy 库

SEED_FLOAT: float = 457.3  # 浮点型种子值
SEED_ARR_FLOAT: npt.NDArray[np.float64] = np.array([1.0, 2, 3, 4])  # 浮点型 NumPy 数组种子值
SEED_ARRLIKE_FLOAT: list[float] = [1.0, 2.0, 3.0, 4.0]  # 类似数组的浮点型列表种子值
SEED_SEED_SEQ: np.random.SeedSequence = np.random.SeedSequence(0)  # NumPy 随机数种子序列对象
SEED_STR: str = "String seeding not allowed"  # 字符串类型的种子值，不被允许

# 默认的随机数生成器
np.random.default_rng(SEED_FLOAT)  # E: 不兼容的类型
np.random.default_rng(SEED_ARR_FLOAT)  # E: 不兼容的类型
np.random.default_rng(SEED_ARRLIKE_FLOAT)  # E: 不兼容的类型
np.random.default_rng(SEED_STR)  # E: 不兼容的类型

# 种子序列对象
np.random.SeedSequence(SEED_FLOAT)  # E: 不兼容的类型
np.random.SeedSequence(SEED_ARR_FLOAT)  # E: 不兼容的类型
np.random.SeedSequence(SEED_ARRLIKE_FLOAT)  # E: 不兼容的类型
np.random.SeedSequence(SEED_SEED_SEQ)  # E: 不兼容的类型
np.random.SeedSequence(SEED_STR)  # E: 不兼容的类型

seed_seq: np.random.bit_generator.SeedSequence = np.random.SeedSequence()  # BitGenerator 类型的种子序列对象
seed_seq.spawn(11.5)  # E: 不兼容的类型
seed_seq.generate_state(3.14)  # E: 不兼容的类型
seed_seq.generate_state(3, np.uint8)  # E: 不兼容的类型
seed_seq.generate_state(3, "uint8")  # E: 不兼容的类型
seed_seq.generate_state(3, "u1")  # E: 不兼容的类型
seed_seq.generate_state(3, np.uint16)  # E: 不兼容的类型
seed_seq.generate_state(3, "uint16")  # E: 不兼容的类型
seed_seq.generate_state(3, "u2")  # E: 不兼容的类型
seed_seq.generate_state(3, np.int32)  # E: 不兼容的类型
seed_seq.generate_state(3, "int32")  # E: 不兼容的类型
seed_seq.generate_state(3, "i4")  # E: 不兼容的类型

# 比特生成器
np.random.MT19937(SEED_FLOAT)  # E: 不兼容的类型
np.random.MT19937(SEED_ARR_FLOAT)  # E: 不兼容的类型
np.random.MT19937(SEED_ARRLIKE_FLOAT)  # E: 不兼容的类型
np.random.MT19937(SEED_STR)  # E: 不兼容的类型

np.random.PCG64(SEED_FLOAT)  # E: 不兼容的类型
np.random.PCG64(SEED_ARR_FLOAT)  # E: 不兼容的类型
np.random.PCG64(SEED_ARRLIKE_FLOAT)  # E: 不兼容的类型
np.random.PCG64(SEED_STR)  # E: 不兼容的类型

np.random.Philox(SEED_FLOAT)  # E: 不兼容的类型
np.random.Philox(SEED_ARR_FLOAT)  # E: 不兼容的类型
np.random.Philox(SEED_ARRLIKE_FLOAT)  # E: 不兼容的类型
np.random.Philox(SEED_STR)  # E: 不兼容的类型

np.random.SFC64(SEED_FLOAT)  # E: 不兼容的类型
np.random.SFC64(SEED_ARR_FLOAT)  # E: 不兼容的类型
np.random.SFC64(SEED_ARRLIKE_FLOAT)  # E: 不兼容的类型
np.random.SFC64(SEED_STR)  # E: 不兼容的类型

# 生成器
np.random.Generator(None)  # E: 不兼容的类型
np.random.Generator(12333283902830213)  # E: 不兼容的类型
np.random.Generator("OxFEEDF00D")  # E: 不兼容的类型
np.random.Generator([123, 234])  # E: 不兼容的类型
np.random.Generator(np.array([123, 234], dtype="u4"))  # E: 不兼容的类型
```