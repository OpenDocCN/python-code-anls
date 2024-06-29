# `.\numpy\numpy\_core\tests\test_simd_module.py`

```
import pytest
from numpy._core._simd import targets

"""
这个测试单元用于检查常见功能的健壮性，
因此我们只需要选择一个表示任何已启用SIMD扩展的子模块来运行测试，
第二个子模块只需运行与在每个子模块之间混合数据类型的可能性相关的一个检查。
"""

# 从targets.values()中筛选出支持SIMD且非空的模块
npyvs = [npyv_mod for npyv_mod in targets.values() if npyv_mod and npyv_mod.simd]
# 取出前两个模块，如果不足两个则用None填充
npyv, npyv2 = (npyvs + [None, None])[:2]

# 无符号整数类型后缀
unsigned_sfx = ["u8", "u16", "u32", "u64"]
# 有符号整数类型后缀
signed_sfx = ["s8", "s16", "s32", "s64"]
# 浮点数类型后缀，根据npyv对象支持的simd_f32和simd_f64属性决定
fp_sfx = []
if npyv and npyv.simd_f32:
    fp_sfx.append("f32")
if npyv and npyv.simd_f64:
    fp_sfx.append("f64")

# 整数类型后缀
int_sfx = unsigned_sfx + signed_sfx
# 所有类型后缀
all_sfx = unsigned_sfx + int_sfx

# 使用pytest标记，如果没有npyv对象支持，跳过测试
@pytest.mark.skipif(not npyv, reason="could not find any SIMD extension with NPYV support")
class Test_SIMD_MODULE:

    # 参数化测试函数，测试每种类型的向量长度是否正确
    @pytest.mark.parametrize('sfx', all_sfx)
    def test_num_lanes(self, sfx):
        nlanes = getattr(npyv, "nlanes_" + sfx)
        vector = getattr(npyv, "setall_" + sfx)(1)
        assert len(vector) == nlanes

    # 参数化测试函数，测试每种类型的向量名称是否正确
    @pytest.mark.parametrize('sfx', all_sfx)
    def test_type_name(self, sfx):
        vector = getattr(npyv, "setall_" + sfx)(1)
        assert vector.__name__ == "npyv_" + sfx

    # 测试函数，验证各种操作是否会引发TypeError或ValueError异常
    def test_raises(self):
        a, b = [npyv.setall_u32(1)]*2
        for sfx in all_sfx:
            vcb = lambda intrin: getattr(npyv, f"{intrin}_{sfx}")
            pytest.raises(TypeError, vcb("add"), a)
            pytest.raises(TypeError, vcb("add"), a, b, a)
            pytest.raises(TypeError, vcb("setall"))
            pytest.raises(TypeError, vcb("setall"), [1])
            pytest.raises(TypeError, vcb("load"), 1)
            pytest.raises(ValueError, vcb("load"), [1])
            pytest.raises(ValueError, vcb("store"), [1], getattr(npyv, f"reinterpret_{sfx}_u32")(a))

    # 根据pytest标记，如果没有npyv2对象支持，跳过测试
    @pytest.mark.skipif(not npyv2, reason="could not find a second SIMD extension with NPYV support")
    def test_nomix(self):
        # 混合不同子模块的操作不允许
        a = npyv.setall_u32(1)
        a2 = npyv2.setall_u32(1)
        pytest.raises(TypeError, npyv.add_u32, a2, a2)
        pytest.raises(TypeError, npyv2.add_u32, a, a)

    # 参数化测试函数，测试无符号整数溢出情况
    @pytest.mark.parametrize('sfx', unsigned_sfx)
    def test_unsigned_overflow(self, sfx):
        nlanes = getattr(npyv, "nlanes_" + sfx)
        maxu = (1 << int(sfx[1:])) - 1
        maxu_72 = (1 << 72) - 1
        lane = getattr(npyv, "setall_" + sfx)(maxu_72)[0]
        assert lane == maxu
        lanes = getattr(npyv, "load_" + sfx)([maxu_72] * nlanes)
        assert lanes == [maxu] * nlanes
        lane = getattr(npyv, "setall_" + sfx)(-1)[0]
        assert lane == maxu
        lanes = getattr(npyv, "load_" + sfx)([-1] * nlanes)
        assert lanes == [maxu] * nlanes

    # 参数化测试函数，测试有符号整数溢出情况
    @pytest.mark.parametrize('sfx', signed_sfx)
    # 测试有符号整数溢出情况的方法
    def test_signed_overflow(self, sfx):
        # 获取指定后缀的 SIMD 向量长度
        nlanes = getattr(npyv, "nlanes_" + sfx)
        # 计算出一个比特位 71 位全为 1 的数值
        maxs_72 = (1 << 71) - 1
        # 使用 SIMD 扩展将所有 lane 设置为 maxs_72，取第一个 lane 的值
        lane = getattr(npyv, "setall_" + sfx)(maxs_72)[0]
        # 断言第一个 lane 的值应为 -1
        assert lane == -1
        # 使用 SIMD 扩展加载 nlanes 个 lane，每个 lane 的值都为 maxs_72
        lanes = getattr(npyv, "load_" + sfx)([maxs_72] * nlanes)
        # 断言加载的所有 lane 的值都为 -1
        assert lanes == [-1] * nlanes
        # 计算出一个比特位 71 位最高位为 1 的负数
        mins_72 = -1 << 71
        # 使用 SIMD 扩展将所有 lane 设置为 mins_72，取第一个 lane 的值
        lane = getattr(npyv, "setall_" + sfx)(mins_72)[0]
        # 断言第一个 lane 的值应为 0
        assert lane == 0
        # 使用 SIMD 扩展加载 nlanes 个 lane，每个 lane 的值都为 mins_72
        lanes = getattr(npyv, "load_" + sfx)([mins_72] * nlanes)
        # 断言加载的所有 lane 的值都为 0
        assert lanes == [0] * nlanes

    # 测试单精度浮点数截断情况的方法
    def test_truncate_f32(self):
        # 如果 SIMD 不支持单精度浮点数操作，则跳过测试
        if not npyv.simd_f32:
            pytest.skip("F32 isn't support by the SIMD extension")
        # 使用 SIMD 扩展将所有单精度 lane 设置为 0.1，取第一个 lane 的值
        f32 = npyv.setall_f32(0.1)[0]
        # 断言第一个 lane 的值不等于 0.1
        assert f32 != 0.1
        # 断言第一个 lane 的值四舍五入到小数点后一位等于 0.1
        assert round(f32, 1) == 0.1

    # 测试比较操作的方法
    def test_compare(self):
        # 创建一个数据范围，从 0 到 SIMD 向量长度的无符号整数
        data_range = range(0, npyv.nlanes_u32)
        # 使用 SIMD 扩展加载一个无符号整数向量 vdata，其值为 data_range
        vdata = npyv.load_u32(data_range)
        # 断言加载的 SIMD 向量 vdata 应与数据范围列表相等
        assert vdata == list(data_range)
        # 断言加载的 SIMD 向量 vdata 应与数据范围元组相等
        assert vdata == tuple(data_range)
        # 遍历数据范围，逐个断言 SIMD 向量 vdata 中的每个元素与数据范围中对应位置的元素相等
        for i in data_range:
            assert vdata[i] == data_range[i]
```