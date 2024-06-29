# `.\numpy\numpy\_core\tests\test_simd.py`

```py
# NOTE: Please avoid the use of numpy.testing since NPYV intrinsics
# may be involved in their functionality.
import pytest, math, re
import itertools
import operator
from numpy._core._simd import targets, clear_floatstatus, get_floatstatus
from numpy._core._multiarray_umath import __cpu_baseline__

class _Test_Utility:
    # submodule of the desired SIMD extension, e.g. targets["AVX512F"]
    npyv = None
    # the current data type suffix e.g. 's8'
    sfx  = None
    # target name can be 'baseline' or one or more of CPU features
    target_name = None

    def __getattr__(self, attr):
        """
        To call NPV intrinsics without the attribute 'npyv' and
        auto suffixing intrinsics according to class attribute 'sfx'
        """
        return getattr(self.npyv, attr + "_" + self.sfx)

    def _x2(self, intrin_name):
        """
        Returns the intrinsic function name suffixed with 'sfx' followed by 'x2'.
        """
        return getattr(self.npyv, f"{intrin_name}_{self.sfx}x2")

    def _data(self, start=None, count=None, reverse=False):
        """
        Create list of consecutive numbers according to number of vector's lanes.
        """
        if start is None:
            start = 1
        if count is None:
            count = self.nlanes
        rng = range(start, start + count)
        if reverse:
            rng = reversed(rng)
        if self._is_fp():
            return [x / 1.0 for x in rng]
        return list(rng)

    def _is_unsigned(self):
        """
        Checks if the current data type suffix indicates an unsigned integer type.
        """
        return self.sfx[0] == 'u'

    def _is_signed(self):
        """
        Checks if the current data type suffix indicates a signed integer type.
        """
        return self.sfx[0] == 's'

    def _is_fp(self):
        """
        Checks if the current data type suffix indicates a floating point type.
        """
        return self.sfx[0] == 'f'

    def _scalar_size(self):
        """
        Returns the size of the scalar in bytes based on the current data type suffix.
        """
        return int(self.sfx[1:])

    def _int_clip(self, seq):
        """
        Clips integer sequence 'seq' to fit within the valid range for the current data type suffix.
        """
        if self._is_fp():
            return seq
        max_int = self._int_max()
        min_int = self._int_min()
        return [min(max(v, min_int), max_int) for v in seq]

    def _int_max(self):
        """
        Returns the maximum representable integer value for the current data type suffix.
        """
        if self._is_fp():
            return None
        max_u = self._to_unsigned(self.setall(-1))[0]
        if self._is_signed():
            return max_u // 2
        return max_u

    def _int_min(self):
        """
        Returns the minimum representable integer value for the current data type suffix.
        """
        if self._is_fp():
            return None
        if self._is_unsigned():
            return 0
        return -(self._int_max() + 1)

    def _true_mask(self):
        """
        Returns the true mask for the current data type suffix.
        """
        max_unsig = getattr(self.npyv, "setall_u" + self.sfx[1:])(-1)
        return max_unsig[0]

def check_floatstatus(divbyzero=False, overflow=False,
                      underflow=False, invalid=False,
                      all=False):
    """
    Checks the floating point status flags against specified conditions.
    """
    #define NPY_FPE_DIVIDEBYZERO  1
    #define NPY_FPE_OVERFLOW      2
    #define NPY_FPE_UNDERFLOW     4
    #define NPY_FPE_INVALID       8
    err = get_floatstatus()
    ret = (all or divbyzero) and (err & 1) != 0
    ret |= (all or overflow) and (err & 2) != 0
    ret |= (all or underflow) and (err & 4) != 0
    ret |= (all or invalid) and (err & 8) != 0
    return ret
    # 将输入的向量转换为无符号数形式
    def _to_unsigned(self, vector):
        # 如果向量是列表或元组，调用对应的 numpy 加载函数来加载无符号数
        if isinstance(vector, (list, tuple)):
            return getattr(self.npyv, "load_u" + self.sfx[1:])(vector)
        else:
            # 否则，从向量对象中提取名称后缀并生成相应的转换函数名
            sfx = vector.__name__.replace("npyv_", "")
            if sfx[0] == "b":
                cvt_intrin = "cvt_u{0}_b{0}"  # 使用 b 开头的转换函数
            else:
                cvt_intrin = "reinterpret_u{0}_{1}"  # 使用 reinterpret 转换函数
            # 调用生成的转换函数来将向量转换为无符号数形式
            return getattr(self.npyv, cvt_intrin.format(sfx[1:], sfx))(vector)

    # 返回正无穷大
    def _pinfinity(self):
        return float("inf")

    # 返回负无穷大
    def _ninfinity(self):
        return -float("inf")

    # 返回 NaN（Not a Number）
    def _nan(self):
        return float("nan")

    # 返回 CPU 的特性信息
    def _cpu_features(self):
        target = self.target_name
        # 如果目标名称是 "baseline"，则替换为 __cpu_baseline__ 的值
        if target == "baseline":
            target = __cpu_baseline__
        else:
            # 否则，按 '__' 进行分割目标名称，得到多个目标信息
            target = target.split('__')  # 多目标分隔符
        # 将目标信息用空格连接成字符串并返回
        return ' '.join(target)
    # _SIMD_BOOL 类，继承自 _Test_Utility 类，用于测试布尔向量类型的统一接口
    class _SIMD_BOOL(_Test_Utility):
        """
        To test all boolean vector types at once
        """
        
        # 获取向量长度（lanes 数量），通过获取属性的方式动态调用
        def _nlanes(self):
            return getattr(self.npyv, "nlanes_u" + self.sfx[1:])
        
        # 生成测试数据，返回一个布尔向量，可以选择是否反向生成
        def _data(self, start=None, count=None, reverse=False):
            # 获取真值掩码
            true_mask = self._true_mask()
            # 生成范围为 0 到 nlanes 的序列
            rng = range(self._nlanes())
            if reverse:
                rng = reversed(rng)
            # 生成布尔向量数据，奇数索引位置使用 true_mask，偶数索引位置使用 0
            return [true_mask if x % 2 else 0 for x in rng]
        
        # 加载数据并转换为布尔向量，依据数据长度动态调用 load 和 cvt 方法
        def _load_b(self, data):
            len_str = self.sfx[1:]
            load = getattr(self.npyv, "load_u" + len_str)
            cvt = getattr(self.npyv, f"cvt_b{len_str}_u{len_str}")
            return cvt(load(data))
        
        # 测试逻辑运算操作符，包括 and、or、xor 和 not，使用相应的向量操作方法
        def test_operators_logical(self):
            """
            Logical operations for boolean types.
            Test intrinsics:
                npyv_xor_##SFX, npyv_and_##SFX, npyv_or_##SFX, npyv_not_##SFX,
                npyv_andc_b8, npvy_orc_b8, nvpy_xnor_b8
            """
            # 获取正序和反序的数据和向量数据
            data_a = self._data()
            data_b = self._data(reverse=True)
            vdata_a = self._load_b(data_a)
            vdata_b = self._load_b(data_b)
            
            # 计算按位与的结果并断言相等
            data_and = [a & b for a, b in zip(data_a, data_b)]
            vand = getattr(self, "and")(vdata_a, vdata_b)
            assert vand == data_and
            
            # 计算按位或的结果并断言相等
            data_or = [a | b for a, b in zip(data_a, data_b)]
            vor = getattr(self, "or")(vdata_a, vdata_b)
            assert vor == data_or
            
            # 计算按位异或的结果并断言相等
            data_xor = [a ^ b for a, b in zip(data_a, data_b)]
            vxor = getattr(self, "xor")(vdata_a, vdata_b)
            assert vxor == data_xor
            
            # 计算按位取反的结果并断言相等
            vnot = getattr(self, "not")(vdata_a)
            assert vnot == data_b
            
            # 如果数据长度不是 b8，则不支持 andc、orc 和 xnor 操作，直接返回
            if self.sfx not in ("b8"):
                return
            
            # 计算按位与非的结果并断言相等
            data_andc = [(a & ~b) & 0xFF for a, b in zip(data_a, data_b)]
            vandc = getattr(self, "andc")(vdata_a, vdata_b)
            assert data_andc == vandc
            
            # 计算按位或非的结果并断言相等
            data_orc = [(a | ~b) & 0xFF for a, b in zip(data_a, data_b)]
            vorc = getattr(self, "orc")(vdata_a, vdata_b)
            assert data_orc == vorc
            
            # 计算按位异或非的结果并断言相等
            data_xnor = [~(a ^ b) & 0xFF for a, b in zip(data_a, data_b)]
            vxnor = getattr(self, "xnor")(vdata_a, vdata_b)
            assert data_xnor == vxnor
        
        # 测试将布尔向量转换为位数的操作
        def test_tobits(self):
            # 定义一个 lambda 函数将数据转换为位数
            data2bits = lambda data: sum([int(x != 0) << i for i, x in enumerate(data, 0)])
            # 对正序和反序的数据进行遍历
            for data in (self._data(), self._data(reverse=True)):
                vdata = self._load_b(data)
                # 计算数据的位表示
                data_bits = data2bits(data)
                # 进行 tobits 操作并断言结果与预期相等
                tobits = self.tobits(vdata)
                bin_tobits = bin(tobits)
                assert bin_tobits == bin(data_bits)
    def test_pack(self):
        """
        Pack multiple vectors into one
        Test intrinsics:
            npyv_pack_b8_b16
            npyv_pack_b8_b32
            npyv_pack_b8_b64
        """
        # 如果不是指定的后缀，则不执行测试
        if self.sfx not in ("b16", "b32", "b64"):
            return

        # 创建向量数据
        data = self._data()
        rdata = self._data(reverse=True)
        
        # 加载向量数据为 SIMD 变量
        vdata = self._load_b(data)
        vrdata = self._load_b(rdata)
        
        # 获取对应的打包函数
        pack_simd = getattr(self.npyv, f"pack_b8_{self.sfx}")

        # 根据后缀类型执行不同的打包操作
        if self.sfx == "b16":
            # 将 rdata 和 data 的元素取低8位并打包成 spack
            spack = [(i & 0xFF) for i in (list(rdata) + list(data))]
            # 使用 SIMD 函数打包 vrdata 和 vdata
            vpack = pack_simd(vrdata, vdata)
        elif self.sfx == "b32":
            # 将 rdata 和 data 的元素取低8位并扩展两倍后打包成 spack
            spack = [(i & 0xFF) for i in (2*list(rdata) + 2*list(data))]
            # 使用 SIMD 函数打包 vrdata, vrdata, vdata, vdata
            vpack = pack_simd(vrdata, vrdata, vdata, vdata)
        elif self.sfx == "b64":
            # 将 rdata 和 data 的元素取低8位并扩展四倍后打包成 spack
            spack = [(i & 0xFF) for i in (4*list(rdata) + 4*list(data))]
            # 使用 SIMD 函数打包 vrdata, vrdata, vrdata, vrdata, vdata, vdata, vdata, vdata
            vpack = pack_simd(vrdata, vrdata, vrdata, vrdata,
                               vdata,  vdata,  vdata,  vdata)
        
        # 断言打包结果与预期结果相同
        assert vpack == spack

    @pytest.mark.parametrize("intrin", ["any", "all"])
    @pytest.mark.parametrize("data", (
        [-1, 0],
        [0, -1],
        [-1],
        [0]
    ))
    def test_operators_crosstest(self, intrin, data):
        """
        Test intrinsics:
            npyv_any_##SFX
            npyv_all_##SFX
        """
        # 将数据乘以 SIMD 向量长度后加载为 SIMD 变量
        data_a = self._load_b(data * self._nlanes())
        
        # 动态执行函数名对应的函数
        func = eval(intrin)
        
        # 获取对象实例中的函数引用
        intrin = getattr(self, intrin)
        
        # 计算预期的标量操作结果
        desired = func(data_a)
        
        # 使用 SIMD 函数进行计算
        simd = intrin(data_a)
        
        # 断言 SIMD 计算结果与预期结果相等
        assert not not simd == desired
class _SIMD_INT(_Test_Utility):
    """
    To test all integer vector types at once
    """

    def test_operators_shift(self):
        # 如果数据类型是 'u8' 或 's8'，则跳过测试
        if self.sfx in ("u8", "s8"):
            return

        # 创建包含接近整型最大值的数据集
        data_a = self._data(self._int_max() - self.nlanes)
        # 创建包含整型最小值的数据集，以相反顺序
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # 对于每个位移量进行循环测试
        for count in range(self._scalar_size()):
            # 创建左移后的数据集，用于加载并进行类型转换
            data_shl_a = self.load([a << count for a in data_a])
            # 左移操作
            shl = self.shl(vdata_a, count)
            assert shl == data_shl_a
            # 创建右移后的数据集，用于加载并进行类型转换
            data_shr_a = self.load([a >> count for a in data_a])
            # 右移操作
            shr = self.shr(vdata_a, count)
            assert shr == data_shr_a

        # 对于从 1 到标量大小的位移量进行循环测试
        for count in range(1, self._scalar_size()):
            # 创建左移后的数据集，用于加载并进行类型转换
            data_shl_a = self.load([a << count for a in data_a])
            # 左移常数位移量
            shli = self.shli(vdata_a, count)
            assert shli == data_shl_a
            # 创建右移后的数据集，用于加载并进行类型转换
            data_shr_a = self.load([a >> count for a in data_a])
            # 右移常数位移量
            shri = self.shri(vdata_a, count)
            assert shri == data_shr_a

    def test_arithmetic_subadd_saturated(self):
        # 如果数据类型是 'u32', 's32', 'u64', 或 's64'，则跳过测试
        if self.sfx in ("u32", "s32", "u64", "s64"):
            return

        # 创建包含接近整型最大值的数据集
        data_a = self._data(self._int_max() - self.nlanes)
        # 创建包含整型最小值的数据集，以相反顺序
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # 计算加法结果并进行饱和运算
        data_adds = self._int_clip([a + b for a, b in zip(data_a, data_b)])
        adds = self.adds(vdata_a, vdata_b)
        assert adds == data_adds

        # 计算减法结果并进行饱和运算
        data_subs = self._int_clip([a - b for a, b in zip(data_a, data_b)])
        subs = self.subs(vdata_a, vdata_b)
        assert subs == data_subs

    def test_math_max_min(self):
        # 创建默认数据集
        data_a = self._data()
        # 创建带有 'nlanes' 元素的数据集
        data_b = self._data(self.nlanes)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # 计算最大值，并将结果存储在列表中
        data_max = [max(a, b) for a, b in zip(data_a, data_b)]
        simd_max = self.max(vdata_a, vdata_b)
        assert simd_max == data_max

        # 计算最小值，并将结果存储在列表中
        data_min = [min(a, b) for a, b in zip(data_a, data_b)]
        simd_min = self.min(vdata_a, vdata_b)
        assert simd_min == data_min

    @pytest.mark.parametrize("start", [-100, -10000, 0, 100, 10000])
    def test_reduce_max_min(self, start):
        """
        Test intrinsics:
            npyv_reduce_max_##sfx
            npyv_reduce_min_##sfx
        """
        # 加载具有 'start' 初始值的数据集
        vdata_a = self.load(self._data(start))
        # 验证减少操作后的最大值是否与预期一致
        assert self.reduce_max(vdata_a) == max(vdata_a)
        # 验证减少操作后的最小值是否与预期一致
        assert self.reduce_min(vdata_a) == min(vdata_a)
    def test_conversions(self):
        """
        Round to nearest even integer, assume CPU control register is set to rounding.
        Test intrinsics:
            npyv_round_s32_##SFX
        """
        # 获取当前 CPU 的特性信息
        features = self._cpu_features()
        # 如果不支持双精度 SIMD 并且 CPU 特性中包含 NEON 或 ASIMD，则条件成立
        if not self.npyv.simd_f64 and re.match(r".*(NEON|ASIMD)", features):
            # 在 Armv7 上模拟最近偶数舍入成本很高
            # 我们选择将半数向上舍入。例如 0.5 -> 1, -0.5 -> -1
            _round = lambda v: int(v + (0.5 if v >= 0 else -0.5))
        else:
            # 否则使用内置的 round 函数进行舍入
            _round = round
        # 载入数据集
        vdata_a = self.load(self._data())
        # 从数据集中减去 0.5
        vdata_a = self.sub(vdata_a, self.setall(0.5))
        # 对每个元素进行舍入操作，得到舍入后的数据列表
        data_round = [_round(x) for x in vdata_a]
        # 使用特定的 SIMD 函数对数据进行整数舍入
        vround = self.round_s32(vdata_a)
        # 断言舍入后的 SIMD 结果与手动舍入的结果相等
        assert vround == data_round
class _SIMD_FP64(_Test_Utility):
    """
    To only test double precision
    """
    def test_conversions(self):
        """
        Round to nearest even integer, assume CPU control register is set to rounding.
        Test intrinsics:
            npyv_round_s32_##SFX
        """
        # 加载测试数据
        vdata_a = self.load(self._data())
        # 对加载的数据减去0.5
        vdata_a = self.sub(vdata_a, self.setall(0.5))
        # 将数据乘以-1.5
        vdata_b = self.mul(vdata_a, self.setall(-1.5))
        # 将每个向量中的元素四舍五入到最近的偶数，得到一个列表
        data_round = [round(x) for x in list(vdata_a) + list(vdata_b)]
        # 使用 SIMD 指令进行整数舍入操作
        vround = self.round_s32(vdata_a, vdata_b)
        # 断言结果是否相等
        assert vround == data_round

class _SIMD_FP(_Test_Utility):
    """
    To test all float vector types at once
    """
    def test_arithmetic_fused(self):
        # 加载测试数据并复制三份
        vdata_a, vdata_b, vdata_c = [self.load(self._data())]*3
        # 计算 vdata_c 的两倍
        vdata_cx2 = self.add(vdata_c, vdata_c)
        # 计算 a*b + c
        data_fma = self.load([a * b + c for a, b, c in zip(vdata_a, vdata_b, vdata_c)])
        # 使用 SIMD 指令进行融合乘加运算
        fma = self.muladd(vdata_a, vdata_b, vdata_c)
        # 断言结果是否相等
        assert fma == data_fma
        # 计算 a*b - c
        fms = self.mulsub(vdata_a, vdata_b, vdata_c)
        # 计算 data_fma - vdata_cx2
        data_fms = self.sub(data_fma, vdata_cx2)
        # 断言结果是否相等
        assert fms == data_fms
        # 计算 -(a*b) + c
        nfma = self.nmuladd(vdata_a, vdata_b, vdata_c)
        # 计算 vdata_cx2 - data_fma
        data_nfma = self.sub(vdata_cx2, data_fma)
        # 断言结果是否相等
        assert nfma == data_nfma
        # 计算 -(a*b) - c
        nfms = self.nmulsub(vdata_a, vdata_b, vdata_c)
        # 将 data_fma 中的每个元素乘以-1
        data_nfms = self.mul(data_fma, self.setall(-1))
        # 断言结果是否相等
        assert nfms == data_nfms
        # 计算 a*b -+ c，其中奇数索引位置相加，偶数索引位置相减
        fmas = list(self.muladdsub(vdata_a, vdata_b, vdata_c))
        # 断言结果是否相等
        assert fmas[0::2] == list(data_fms)[0::2]
        assert fmas[1::2] == list(data_fma)[1::2]

    def test_abs(self):
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        data = self._data()
        vdata = self.load(self._data())

        # 定义绝对值测试案例
        abs_cases = ((-0, 0), (ninf, pinf), (pinf, pinf), (nan, nan))
        for case, desired in abs_cases:
            # 根据 case 设置全部元素的绝对值
            data_abs = [desired]*self.nlanes
            vabs = self.abs(self.setall(case))
            # 断言结果是否近似相等，允许 NaN
            assert vabs == pytest.approx(data_abs, nan_ok=True)

        # 计算 vdata 中每个元素的相反数的绝对值
        vabs = self.abs(self.mul(vdata, self.setall(-1)))
        # 断言结果是否相等
        assert vabs == data

    def test_sqrt(self):
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        data = self._data()
        vdata = self.load(self._data())

        # 定义平方根测试案例
        sqrt_cases = ((-0.0, -0.0), (0.0, 0.0), (-1.0, nan), (ninf, nan), (pinf, pinf))
        for case, desired in sqrt_cases:
            # 根据 case 设置全部元素的平方根
            data_sqrt = [desired]*self.nlanes
            sqrt  = self.sqrt(self.setall(case))
            # 断言结果是否近似相等，允许 NaN
            assert sqrt == pytest.approx(data_sqrt, nan_ok=True)

        # 计算 vdata 中每个元素的平方根并加载到降低精度
        data_sqrt = self.load([math.sqrt(x) for x in data]) # load to truncate precision
        sqrt = self.sqrt(vdata)
        # 断言结果是否相等
        assert sqrt == data_sqrt
    def test_square(self):
        # 获取正无穷、负无穷和 NaN 值
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        # 载入测试数据
        data = self._data()
        vdata = self.load(self._data())
        # 测试平方函数
        square_cases = ((nan, nan), (pinf, pinf), (ninf, pinf))
        for case, desired in square_cases:
            # 生成期望的平方结果数组
            data_square = [desired]*self.nlanes
            # 计算给定 case 的平方
            square  = self.square(self.setall(case))
            # 断言计算结果与期望值近似相等
            assert square == pytest.approx(data_square, nan_ok=True)

        # 计算数据数组各元素的平方作为期望值
        data_square = [x*x for x in data]
        # 使用 vdata 计算平方
        square = self.square(vdata)
        # 断言计算结果与期望值相等
        assert square == data_square

    @pytest.mark.parametrize("intrin, func", [("ceil", math.ceil),
    ("trunc", math.trunc), ("floor", math.floor), ("rint", round)])
    def test_rounding(self, intrin, func):
        """
        Test intrinsics:
            npyv_rint_##SFX
            npyv_ceil_##SFX
            npyv_trunc_##SFX
            npyv_floor##SFX
        """
        # 获取指定的内在函数名称和函数对象
        intrin_name = intrin
        intrin = getattr(self, intrin)
        # 获取正无穷、负无穷和 NaN 值
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        
        # 特殊情况
        round_cases = ((nan, nan), (pinf, pinf), (ninf, ninf))
        for case, desired in round_cases:
            # 生成期望的舍入结果数组
            data_round = [desired]*self.nlanes
            # 计算给定 case 的舍入值
            _round = intrin(self.setall(case))
            # 断言计算结果与期望值近似相等
            assert _round == pytest.approx(data_round, nan_ok=True)

        # 测试多种数值和权重的舍入行为
        for x in range(0, 2**20, 256**2):
            for w in (-1.05, -1.10, -1.15, 1.05, 1.10, 1.15):
                # 生成测试数据
                data = self.load([(x+a)*w for a in range(self.nlanes)])
                # 计算数据数组各元素的舍入值
                data_round = [func(x) for x in data]
                # 使用内在函数计算舍入
                _round = intrin(data)
                # 断言计算结果与期望值相等
                assert _round == data_round

        # 测试大数值
        for i in (
            1.1529215045988576e+18, 4.6116860183954304e+18,
            5.902958103546122e+20, 2.3611832414184488e+21
        ):
            x = self.setall(i)
            y = intrin(x)
            # 计算大数值的舍入结果
            data_round = [func(n) for n in x]
            # 断言计算结果与期望值相等
            assert y == data_round

        # 测试带符号零
        if intrin_name == "floor":
            data_szero = (-0.0,)
        else:
            data_szero = (-0.0, -0.25, -0.30, -0.45, -0.5)

        for w in data_szero:
            # 计算带符号零的舍入值并转为无符号数
            _round = self._to_unsigned(intrin(self.setall(w)))
            data_round = self._to_unsigned(self.setall(-0.0))
            # 断言计算结果与期望值相等
            assert _round == data_round

    @pytest.mark.parametrize("intrin", [
        "max", "maxp", "maxn", "min", "minp", "minn"
    ])
    def test_max_min(self, intrin):
        """
        Test intrinsics:
            npyv_max_##sfx
            npyv_maxp_##sfx
            npyv_maxn_##sfx
            npyv_min_##sfx
            npyv_minp_##sfx
            npyv_minn_##sfx
            npyv_reduce_max_##sfx
            npyv_reduce_maxp_##sfx
            npyv_reduce_maxn_##sfx
            npyv_reduce_min_##sfx
            npyv_reduce_minp_##sfx
            npyv_reduce_minn_##sfx
        """
        # 定义正无穷、负无穷和NaN
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        
        # 根据当前指令类型确定是否有NaN的情况
        chk_nan = {"xp": 1, "np": 1, "nn": 2, "xn": 2}.get(intrin[-2:], 0)
        
        # 根据指令动态生成函数对象
        func = eval(intrin[:3])
        
        # 获取对应的 reduce 函数
        reduce_intrin = getattr(self, "reduce_" + intrin)
        
        # 获取指令对应的 intrin 函数
        intrin = getattr(self, intrin)
        
        # 计算半数通道数
        hf_nlanes = self.nlanes//2

        # 定义测试用例
        cases = (
            ([0.0, -0.0], [-0.0, 0.0]),
            ([10, -10],  [10, -10]),
            ([pinf, 10], [10, ninf]),
            ([10, pinf], [ninf, 10]),
            ([10, -10], [10, -10]),
            ([-10, 10], [-10, 10])
        )
        
        # 遍历测试用例
        for op1, op2 in cases:
            # 加载数据到向量 vdata_a 和 vdata_b
            vdata_a = self.load(op1*hf_nlanes)
            vdata_b = self.load(op2*hf_nlanes)
            
            # 计算标量数据并进行 SIMD 计算
            data = func(vdata_a, vdata_b)
            simd = intrin(vdata_a, vdata_b)
            
            # 断言 SIMD 结果与标量计算结果一致
            assert simd == data
            
            # 计算单个向量的结果并进行 reduce 操作
            data = func(vdata_a)
            simd = reduce_intrin(vdata_a)
            
            # 断言 SIMD reduce 结果与标量计算结果一致
            assert simd == data

        # 处理 NaN 的情况
        if not chk_nan:
            return
        
        # 定义处理 NaN 的匿名函数
        if chk_nan == 1:
            test_nan = lambda a, b: (
                b if math.isnan(a) else a if math.isnan(b) else b
            )
        else:
            test_nan = lambda a, b: (
                nan if math.isnan(a) or math.isnan(b) else b
            )
        
        # 针对 NaN 的测试用例
        cases = (
            (nan, 10),
            (10, nan),
            (nan, pinf),
            (pinf, nan),
            (nan, nan)
        )
        
        # 遍历 NaN 的测试用例
        for op1, op2 in cases:
            # 加载数据到向量 vdata_ab
            vdata_ab = self.load([op1, op2]*hf_nlanes)
            
            # 计算期望的数据
            data = test_nan(op1, op2)
            
            # 执行 reduce 操作并断言结果与预期一致
            simd = reduce_intrin(vdata_ab)
            assert simd == pytest.approx(data, nan_ok=True)
            
            # 设置所有向量数据为 op1 和 op2，计算期望结果并进行 SIMD 计算
            vdata_a = self.setall(op1)
            vdata_b = self.setall(op2)
            data = [data] * self.nlanes
            simd = intrin(vdata_a, vdata_b)
            
            # 断言 SIMD 计算结果与期望一致
            assert simd == pytest.approx(data, nan_ok=True)

    def test_reciprocal(self):
        # 获取正无穷、负无穷和NaN
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        
        # 获取数据
        data = self._data()
        
        # 加载数据到向量 vdata
        vdata = self.load(self._data())

        # 计算 reciprocal 的测试用例
        recip_cases = ((nan, nan), (pinf, 0.0), (ninf, -0.0), (0.0, pinf), (-0.0, ninf))
        
        # 遍历 reciprocal 的测试用例
        for case, desired in recip_cases:
            # 计算期望的数据
            data_recip = [desired]*self.nlanes
            
            # 计算 reciprocal 并断言结果与期望一致
            recip = self.recip(self.setall(case))
            assert recip == pytest.approx(data_recip, nan_ok=True)

        # 计算所有数据的 reciprocal
        data_recip = self.load([1/x for x in data]) # load to truncate precision
        
        # 计算 reciprocal 并断言结果与期望一致
        recip = self.recip(vdata)
        assert recip == data_recip
    def test_special_cases(self):
        """
        Compare Not NaN. Test intrinsics:
            npyv_notnan_##SFX
        """
        # 调用 self.notnan 方法，将所有元素设置为 NaN，然后进行比较，期望结果为全零数组
        nnan = self.notnan(self.setall(self._nan()))
        # 断言 nnan 应该等于长度为 self.nlanes 的全零列表
        assert nnan == [0]*self.nlanes

    @pytest.mark.parametrize("intrin_name", [
        "rint", "trunc", "ceil", "floor"
    ])
    def test_unary_invalid_fpexception(self, intrin_name):
        # 获取当前测试的内置函数 intrin
        intrin = getattr(self, intrin_name)
        # 对于特定的无效浮点数值进行测试，包括 NaN、正无穷、负无穷
        for d in [float("nan"), float("inf"), -float("inf")]:
            v = self.setall(d)
            # 清除浮点数状态
            clear_floatstatus()
            # 调用 intrin 处理向量 v
            intrin(v)
            # 断言检查浮点数状态的 invalid 位是否为 False
            assert check_floatstatus(invalid=True) == False

    @pytest.mark.parametrize('py_comp,np_comp', [
        (operator.lt, "cmplt"),
        (operator.le, "cmple"),
        (operator.gt, "cmpgt"),
        (operator.ge, "cmpge"),
        (operator.eq, "cmpeq"),
        (operator.ne, "cmpneq")
    ])
    def test_comparison_with_nan(self, py_comp, np_comp):
        # 获取正、负无穷大和 NaN 的特定值
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        # 获取所有真值的掩码
        mask_true = self._true_mask()

        def to_bool(vector):
            # 将向量转换为布尔值列表
            return [lane == mask_true for lane in vector]

        # 获取对应的 SIMD 操作函数
        intrin = getattr(self, np_comp)
        # 比较测试案例，包括 (0, nan), (nan, 0), (nan, nan), (pinf, nan), (ninf, nan), (-0.0, +0.0)
        cmp_cases = ((0, nan), (nan, 0), (nan, nan), (pinf, nan),
                     (ninf, nan), (-0.0, +0.0))
        for case_operand1, case_operand2 in cmp_cases:
            # 创建数据列表，长度为 self.nlanes，并设置为相应的操作数值
            data_a = [case_operand1]*self.nlanes
            data_b = [case_operand2]*self.nlanes
            # 将数据转换为 SIMD 向量
            vdata_a = self.setall(case_operand1)
            vdata_b = self.setall(case_operand2)
            # 获取 SIMD 操作后的布尔结果向量
            vcmp = to_bool(intrin(vdata_a, vdata_b))
            # 获取 Python 操作后的布尔结果列表
            data_cmp = [py_comp(a, b) for a, b in zip(data_a, data_b)]
            # 断言 SIMD 操作后的结果与 Python 操作后的结果相同
            assert vcmp == data_cmp

    @pytest.mark.parametrize("intrin", ["any", "all"])
    @pytest.mark.parametrize("data", (
        [float("nan"), 0],
        [0, float("nan")],
        [float("nan"), 1],
        [1, float("nan")],
        [float("nan"), float("nan")],
        [0.0, -0.0],
        [-0.0, 0.0],
        [1.0, -0.0]
    ))
    def test_operators_crosstest(self, intrin, data):
        """
        Test intrinsics:
            npyv_any_##SFX
            npyv_all_##SFX
        """
        # 加载测试数据并创建函数对象
        data_a = self.load(data * self.nlanes)
        func = eval(intrin)
        # 获取对应的 SIMD 操作函数
        intrin = getattr(self, intrin)
        # 计算 Python 函数的预期结果
        desired = func(data_a)
        # 执行 SIMD 操作
        simd = intrin(data_a)
        # 断言 SIMD 操作后的结果不为空（即非 False），且与 Python 函数的结果相同
        assert not not simd == desired
class _SIMD_ALL(_Test_Utility):
    """
    To test all vector types at once
    """
    # 测试内存加载功能
    def test_memory_load(self):
        # 获取测试数据
        data = self._data()
        # 不对齐加载
        load_data = self.load(data)
        assert load_data == data
        # 对齐加载
        loada_data = self.loada(data)
        assert loada_data == data
        # 流加载
        loads_data = self.loads(data)
        assert loads_data == data
        # 加载低位部分
        loadl = self.loadl(data)
        loadl_half = list(loadl)[:self.nlanes//2]
        data_half = data[:self.nlanes//2]
        assert loadl_half == data_half
        assert loadl != data # 检测溢出

    # 测试内存存储功能
    def test_memory_store(self):
        # 获取测试数据
        data = self._data()
        vdata = self.load(data)
        # 不对齐存储
        store = [0] * self.nlanes
        self.store(store, vdata)
        assert store == data
        # 对齐存储
        store_a = [0] * self.nlanes
        self.storea(store_a, vdata)
        assert store_a == data
        # 流存储
        store_s = [0] * self.nlanes
        self.stores(store_s, vdata)
        assert store_s == data
        # 存储低位部分
        store_l = [0] * self.nlanes
        self.storel(store_l, vdata)
        assert store_l[:self.nlanes//2] == data[:self.nlanes//2]
        assert store_l != vdata # 检测溢出
        # 存储高位部分
        store_h = [0] * self.nlanes
        self.storeh(store_h, vdata)
        assert store_h[:self.nlanes//2] == data[self.nlanes//2:]
        assert store_h != vdata  # 检测溢出

    @pytest.mark.parametrize("intrin, elsizes, scale, fill", [
        ("self.load_tillz, self.load_till", (32, 64), 1, [0xffff]),
        ("self.load2_tillz, self.load2_till", (32, 64), 2, [0xffff, 0x7fff]),
    ])
    # 测试部分加载功能
    def test_memory_partial_load(self, intrin, elsizes, scale, fill):
        if self._scalar_size() not in elsizes:
            return
        npyv_load_tillz, npyv_load_till = eval(intrin)
        data = self._data()
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes**2, self.nlanes**4] # 测试超出范围
        for n in lanes:
            load_till = npyv_load_till(data, n, *fill)
            load_tillz = npyv_load_tillz(data, n)
            n *= scale
            data_till = data[:n] + fill * ((self.nlanes-n) // scale)
            assert load_till == data_till
            data_tillz = data[:n] + [0] * (self.nlanes-n)
            assert load_tillz == data_tillz

    @pytest.mark.parametrize("intrin, elsizes, scale", [
        ("self.store_till", (32, 64), 1),
        ("self.store2_till", (32, 64), 2),
    ])
    # 测试部分存储功能，检查标量大小是否在指定的元素大小列表中
    def test_memory_partial_store(self, intrin, elsizes, scale):
        # 如果标量大小不在elsizes列表中，则退出测试
        if self._scalar_size() not in elsizes:
            return
        # 将intrin字符串转换为函数对象
        npyv_store_till = eval(intrin)
        # 获取数据
        data = self._data()
        # 获取反转后的数据
        data_rev = self._data(reverse=True)
        # 载入数据
        vdata = self.load(data)
        # 生成一个包含self.nlanes个元素的列表
        lanes = list(range(1, self.nlanes + 1))
        # 添加额外的两个元素到lanes列表中
        lanes += [self.nlanes**2, self.nlanes**4]
        # 遍历lanes列表
        for n in lanes:
            # 复制反转后的数据到data_till
            data_till = data_rev.copy()
            # 将data中的前n*scale个元素复制到data_till中的对应位置
            data_till[:n*scale] = data[:n*scale]
            # 获取反转后的数据
            store_till = self._data(reverse=True)
            # 调用npyv_store_till函数
            npyv_store_till(store_till, n, vdata)
            # 断言store_till与data_till相等
            assert store_till == data_till

    # 标记pytest参数化测试函数，用于非连续加载内存的测试
    @pytest.mark.parametrize("intrin, elsizes, scale", [
        ("self.loadn", (32, 64), 1),
        ("self.loadn2", (32, 64), 2),
    ])
    # 测试非连续加载内存功能
    def test_memory_noncont_load(self, intrin, elsizes, scale):
        # 如果标量大小不在elsizes列表中，则退出测试
        if self._scalar_size() not in elsizes:
            return
        # 将intrin字符串转换为函数对象
        npyv_loadn = eval(intrin)
        # 遍历stride从-64到63的范围
        for stride in range(-64, 64):
            # 如果stride小于0
            if stride < 0:
                # 获取反转后的数据，使用负的stride参数
                data = self._data(stride, -stride*self.nlanes)
                # 创建data_stride列表，将data中的部分元素交错连接起来
                data_stride = list(itertools.chain(
                    *zip(*[data[-i::stride] for i in range(scale, 0, -1)])
                ))
            # 如果stride等于0
            elif stride == 0:
                # 获取数据
                data = self._data()
                # 将数据的前scale个元素复制到data_stride中，重复self.nlanes//scale次
                data_stride = data[0:scale] * (self.nlanes//scale)
            # 如果stride大于0
            else:
                # 获取stride*self.nlanes个元素的数据
                data = self._data(count=stride*self.nlanes)
                # 创建data_stride列表，将data中的部分元素按照stride分组并交错连接起来
                data_stride = list(itertools.chain(
                    *zip(*[data[i::stride] for i in range(scale)]))
                )
            # 将data_stride转换为无符号整数类型
            data_stride = self.load(data_stride)  # cast unsigned
            # 使用npyv_loadn函数加载数据
            loadn = npyv_loadn(data, stride)
            # 断言loadn与data_stride相等
            assert loadn == data_stride

    # 标记pytest参数化测试函数，用于加载至特定边界值的内存测试
    @pytest.mark.parametrize("intrin, elsizes, scale, fill", [
        ("self.loadn_tillz, self.loadn_till", (32, 64), 1, [0xffff]),
        ("self.loadn2_tillz, self.loadn2_till", (32, 64), 2, [0xffff, 0x7fff]),
    ])
    # 定义一个测试函数，用于测试内存非连续部分加载的情况
    def test_memory_noncont_partial_load(self, intrin, elsizes, scale):
        # 如果标量大小不在给定的elsizes列表中，则直接返回
        if self._scalar_size() not in elsizes:
            return
        
        # 根据字符串intrin获取对应的函数或方法，并赋值给npyv_loadn_tillz和npyv_loadn_till
        npyv_loadn_tillz, npyv_loadn_till = eval(intrin)
        
        # 创建一个包含1到self.nlanes + 1的列表，并添加额外的两个元素
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes**2, self.nlanes**4]
        
        # 遍历从-64到63的步长值
        for stride in range(-64, 64):
            if stride < 0:
                # 如果步长为负数，则调用self._data方法获取数据
                data = self._data(stride, -stride*self.nlanes)
                # 通过列表推导式和zip函数，创建部分加载后的数据列表data_stride
                data_stride = list(itertools.chain(
                    *zip(*[data[-i::stride] for i in range(scale, 0, -1)])
                ))
            elif stride == 0:
                # 如果步长为0，则直接调用self._data方法获取数据，并复制scale个元素
                data = self._data()
                data_stride = data[0:scale] * (self.nlanes//scale)
            else:
                # 否则，根据步长调用self._data方法获取数据
                data = self._data(count=stride*self.nlanes)
                # 使用列表推导式和zip函数，创建部分加载后的数据列表data_stride
                data_stride = list(itertools.chain(
                    *zip(*[data[i::stride] for i in range(scale)])
                ))
            
            # 将data_stride转换为无符号整数形式的列表
            data_stride = list(self.load(data_stride))
            
            # 遍历lanes列表中的元素
            for n in lanes:
                # 计算nscale值
                nscale = n * scale
                # 计算llanes值
                llanes = self.nlanes - nscale
                
                # 构造部分加载后的数据列表data_stride_till
                data_stride_till = (
                    data_stride[:nscale] + fill * (llanes//scale)
                )
                
                # 调用npyv_loadn_till函数处理data数据，验证结果与data_stride_till是否相等
                loadn_till = npyv_loadn_till(data, stride, n, *fill)
                assert loadn_till == data_stride_till
                
                # 构造部分加载且填充为0的数据列表data_stride_tillz
                data_stride_tillz = data_stride[:nscale] + [0] * llanes
                
                # 调用npyv_loadn_tillz函数处理data数据，验证结果与data_stride_tillz是否相等
                loadn_tillz = npyv_loadn_tillz(data, stride, n)
                assert loadn_tillz == data_stride_tillz
    # 定义测试函数 test_memory_noncont_store，接受参数 intrin, elsizes, scale
    def test_memory_noncont_store(self, intrin, elsizes, scale):
        # 如果当前对象的标量大小不在 elsizes 中，则直接返回
        if self._scalar_size() not in elsizes:
            return
        # 根据字符串 intrin 动态评估得到 npyv_storen 函数
        npyv_storen = eval(intrin)
        # 获得当前对象的数据
        data = self._data()
        # 载入数据到向量数据 vdata
        vdata = self.load(data)
        # 计算每个向量的半长度
        hlanes = self.nlanes // scale
        
        # 循环不同的步长
        for stride in range(1, 64):
            # 初始化数据存储结构为一组特定值
            data_storen = [0xff] * stride * self.nlanes
            # 遍历数据索引范围以填充 data_storen
            for s in range(0, hlanes*stride, stride):
                i = (s//stride)*scale
                data_storen[s:s+scale] = data[i:i+scale]
            # 初始化 storen 为特定值组成的列表，后续追加额外值
            storen = [0xff] * stride * self.nlanes
            storen += [0x7f]*64
            # 使用 npyv_storen 函数存储数据到 storen 中
            npyv_storen(storen, stride, vdata)
            # 断言前部分数据与 data_storen 相等
            assert storen[:-64] == data_storen
            # 断言后 64 个元素为 [0x7f]
            assert storen[-64:] == [0x7f]*64  # detect overflow

        # 对于负的步长值
        for stride in range(-64, 0):
            # 初始化数据存储结构为一组特定值
            data_storen = [0xff] * -stride * self.nlanes
            # 遍历数据索引范围以填充 data_storen
            for s in range(0, hlanes*stride, stride):
                i = (s//stride)*scale
                data_storen[s-scale:s or None] = data[i:i+scale]
            # 初始化 storen 为 [0x7f] 组成的列表，后续追加特定值
            storen = [0x7f]*64
            storen += [0xff] * -stride * self.nlanes
            # 使用 npyv_storen 函数存储数据到 storen 中
            npyv_storen(storen, stride, vdata)
            # 断言后部分数据与 data_storen 相等
            assert storen[64:] == data_storen
            # 断言前 64 个元素为 [0x7f]
            assert storen[:64] == [0x7f]*64  # detect overflow
        
        # 对于步长为 0 的情况
        # 初始化数据存储结构为一组 [0x7f]
        data_storen = [0x7f] * self.nlanes
        storen = data_storen.copy()
        # 将数据的末尾部分复制到 data_storen 的前 scale 个位置
        data_storen[0:scale] = data[-scale:]
        # 使用 npyv_storen 函数存储数据到 storen 中
        npyv_storen(storen, 0, vdata)
        # 断言 storen 与 data_storen 相等
        assert storen == data_storen

    # 使用 pytest.mark.parametrize 运行参数化测试，参数为 intrin, elsizes, scale
    @pytest.mark.parametrize("intrin, elsizes, scale", [
        ("self.storen_till", (32, 64), 1),
        ("self.storen2_till", (32, 64), 2),
    ])
    def test_memory_noncont_partial_store(self, intrin, elsizes, scale):
        # 检查当前标量大小是否在给定的元素大小列表中，如果不在则直接返回
        if self._scalar_size() not in elsizes:
            return

        # 根据传入的指令名称，获取对应的函数对象
        npyv_storen_till = eval(intrin)

        # 获取当前对象的数据
        data = self._data()

        # 加载数据到向量数据对象中
        vdata = self.load(data)

        # 构建一个包含指定范围的 lane 数字的列表
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes**2, self.nlanes**4]

        # 计算缩放后的 lane 数量
        hlanes = self.nlanes // scale

        # 遍历不同的步长
        for stride in range(1, 64):
            for n in lanes:
                # 初始化数据直到指定位置
                data_till = [0xff] * stride * self.nlanes

                # 根据给定的 n 和缩放因子构建 tdata
                tdata = data[:n*scale] + [0xff] * (self.nlanes-n*scale)

                # 填充数据直到指定位置
                for s in range(0, hlanes*stride, stride)[:n]:
                    i = (s//stride)*scale
                    data_till[s:s+scale] = tdata[i:i+scale]

                # 初始化存储数据直到指定位置
                storen_till = [0xff] * stride * self.nlanes
                storen_till += [0x7f]*64

                # 调用指定的存储函数
                npyv_storen_till(storen_till, stride, n, vdata)

                # 断言存储数据直到指定位置与预期的数据相同
                assert storen_till[:-64] == data_till

                # 检测溢出
                assert storen_till[-64:] == [0x7f]*64  # detect overflow

        # 负步长情况下的遍历
        for stride in range(-64, 0):
            for n in lanes:
                # 初始化数据直到指定位置
                data_till = [0xff] * -stride * self.nlanes

                # 根据给定的 n 和缩放因子构建 tdata
                tdata = data[:n*scale] + [0xff] * (self.nlanes-n*scale)

                # 填充数据直到指定位置
                for s in range(0, hlanes*stride, stride)[:n]:
                    i = (s//stride)*scale
                    data_till[s-scale:s or None] = tdata[i:i+scale]

                # 初始化存储数据直到指定位置
                storen_till = [0x7f]*64
                storen_till += [0xff] * -stride * self.nlanes

                # 调用指定的存储函数
                npyv_storen_till(storen_till, stride, n, vdata)

                # 断言存储数据直到指定位置与预期的数据相同
                assert storen_till[64:] == data_till

                # 检测溢出
                assert storen_till[:64] == [0x7f]*64  # detect overflow

        # 步长为 0 的情况
        for n in lanes:
            # 初始化数据直到指定位置
            data_till = [0x7f] * self.nlanes

            # 复制数据直到指定位置
            storen_till = data_till.copy()

            # 从数据的末尾复制指定长度的数据到存储数据中
            data_till[0:scale] = data[:n*scale][-scale:]

            # 调用指定的存储函数
            npyv_storen_till(storen_till, 0, n, vdata)

            # 断言存储数据与预期的数据相同
            assert storen_till == data_till

    @pytest.mark.parametrize("intrin, table_size, elsize", [
        # 参数化测试不同的内联函数和表格大小
        ("self.lut32", 32, 32),
        ("self.lut16", 16, 64)
    ])
    def test_lut(self, intrin, table_size, elsize):
        """
        Test lookup table intrinsics:
            npyv_lut32_##sfx
            npyv_lut16_##sfx
        """
        # 检查元素大小是否与当前对象的标量大小相同，如果不同则直接返回
        if elsize != self._scalar_size():
            return

        # 根据传入的指令名称，获取对应的函数对象
        intrin = eval(intrin)

        # 获取设置所有元素的索引函数对象
        idx_itrin = getattr(self.npyv, f"setall_u{elsize}")

        # 创建一个指定范围的表格
        table = range(0, table_size)

        # 遍历表格中的每个索引
        for i in table:
            # 设置所有元素为当前索引值的向量
            broadi = self.setall(i)

            # 使用索引设置函数获取索引值
            idx = idx_itrin(i)

            # 调用内联函数处理表格和索引，返回结果
            lut = intrin(table, idx)

            # 断言内联函数返回的结果与预期的广播向量相同
            assert lut == broadi
    # 定义一个测试方法，用于测试各种杂项功能
    def test_misc(self):
        # 调用 self.zero() 方法，返回全零的列表，长度为 self.nlanes
        broadcast_zero = self.zero()
        # 断言 broadcast_zero 应该等于 [0] * self.nlanes
        assert broadcast_zero == [0] * self.nlanes
        
        # 遍历范围从 1 到 9
        for i in range(1, 10):
            # 调用 self.setall(i) 方法，返回所有元素为 i 的列表
            broadcasti = self.setall(i)
            # 断言 broadcasti 应该等于 [i] * self.nlanes
            assert broadcasti == [i] * self.nlanes

        # 分别获取两种数据：正常顺序和反向顺序的 self._data()
        data_a, data_b = self._data(), self._data(reverse=True)
        # 使用 self.load() 方法加载两种数据，得到 vdata_a 和 vdata_b
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # 测试 self.set() 方法，传入 data_a，期望返回相同的数据
        vset = self.set(*data_a)
        assert vset == data_a
        # 测试 self.setf() 方法，传入 10 和 data_a，期望返回 data_a
        vsetf = self.setf(10, *data_a)
        assert vsetf == data_a

        # 测试 _simd 的类型向量的稳定性，
        # reinterpret* 内在本身通过编译器测试
        sfxes = ["u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64"]
        # 如果支持 simd_f64，添加 "f64" 到 sfxes
        if self.npyv.simd_f64:
            sfxes.append("f64")
        # 如果支持 simd_f32，添加 "f32" 到 sfxes
        if self.npyv.simd_f32:
            sfxes.append("f32")
        # 遍历所有的后缀 sfxes
        for sfx in sfxes:
            # 构造 reinterpret_sfx 方法的名字，调用它并获取返回值的名称
            vec_name = getattr(self, "reinterpret_" + sfx)(vdata_a).__name__
            # 断言 vec_name 应该等于 "npyv_" + sfx
            assert vec_name == "npyv_" + sfx

        # select & mask 操作
        # 使用 self.cmpeq() 比较 self.zero() 和 self.zero()，得到掩码
        select_a = self.select(self.cmpeq(self.zero(), self.zero()), vdata_a, vdata_b)
        # 断言 select_a 应该等于 data_a
        assert select_a == data_a
        # 使用 self.cmpneq() 比较 self.zero() 和 self.zero()，得到掩码
        select_b = self.select(self.cmpneq(self.zero(), self.zero()), vdata_a, vdata_b)
        # 断言 select_b 应该等于 data_b
        assert select_b == data_b

        # 测试提取元素
        # 断言 self.extract0(vdata_b) 应该等于 vdata_b 的第一个元素
        assert self.extract0(vdata_b) == vdata_b[0]

        # cleanup 操作仅在 AVX 中用于清零寄存器，避免 AVX-SSE 过渡损失
        # 在这里没有需要测试的内容
        self.npyv.cleanup()
    def test_reorder(self):
        # 准备测试数据，一个正序，一个逆序
        data_a, data_b  = self._data(), self._data(reverse=True)
        # 载入数据到向量数据
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)
        
        # 取前半部分数据
        data_a_lo = data_a[:self.nlanes//2]
        data_b_lo = data_b[:self.nlanes//2]
        
        # 取后半部分数据
        data_a_hi = data_a[self.nlanes//2:]
        data_b_hi = data_b[self.nlanes//2:]
        
        # 合并两个低部分数据
        combinel = self.combinel(vdata_a, vdata_b)
        assert combinel == data_a_lo + data_b_lo
        
        # 合并两个高部分数据
        combineh = self.combineh(vdata_a, vdata_b)
        assert combineh == data_a_hi + data_b_hi
        
        # 合并两倍
        combine = self.combine(vdata_a, vdata_b)
        assert combine == (data_a_lo + data_b_lo, data_a_hi + data_b_hi)

        # zip(interleave)
        # 按交错顺序载入数据
        data_zipl = self.load([
            v for p in zip(data_a_lo, data_b_lo) for v in p
        ])
        data_ziph = self.load([
            v for p in zip(data_a_hi, data_b_hi) for v in p
        ])
        vzip = self.zip(vdata_a, vdata_b)
        assert vzip == (data_zipl, data_ziph)
        vzip = [0]*self.nlanes*2
        self._x2("store")(vzip, (vdata_a, vdata_b))
        assert vzip == list(data_zipl) + list(data_ziph)

        # unzip(deinterleave)
        # 按反交错顺序解压数据
        unzip = self.unzip(data_zipl, data_ziph)
        assert unzip == (data_a, data_b)
        unzip = self._x2("load")(list(data_zipl) + list(data_ziph))
        assert unzip == (data_a, data_b)

    def test_reorder_rev64(self):
        # 反转每个64位元素的顺序
        ssize = self._scalar_size()
        if ssize == 64:
            return
        data_rev64 = [
            y for x in range(0, self.nlanes, 64//ssize)
              for y in reversed(range(x, x + 64//ssize))
        ]
        rev64 = self.rev64(self.load(range(self.nlanes)))
        assert rev64 == data_rev64

    def test_reorder_permi128(self):
        """
        Test permuting elements for each 128-bit lane.
        npyv_permi128_##sfx
        """
        ssize = self._scalar_size()
        if ssize < 32:
            return
        data = self.load(self._data())
        permn = 128//ssize
        permd = permn-1
        nlane128 = self.nlanes//permn
        shfl = [0, 1] if ssize == 64 else [0, 2, 4, 6]
        for i in range(permn):
            # 计算排列索引
            indices = [(i >> shf) & permd for shf in shfl]
            vperm = self.permi128(data, *indices)
            # 验证排列后的数据是否正确
            data_vperm = [
                data[j + (e & -permn)]
                for e, j in enumerate(indices*nlane128)
            ]
            assert vperm == data_vperm

    @pytest.mark.parametrize('func, intrin', [
        (operator.lt, "cmplt"),
        (operator.le, "cmple"),
        (operator.gt, "cmpgt"),
        (operator.ge, "cmpge"),
        (operator.eq, "cmpeq")
    ])
    # 对比运算测试函数，接受函数 `func` 和字符串 `intrin` 作为参数
    def test_operators_comparison(self, func, intrin):
        # 如果是浮点数类型，则使用默认数据
        if self._is_fp():
            data_a = self._data()
        else:
            # 否则，使用 `_int_max() - self.nlanes` 为起点的数据
            data_a = self._data(self._int_max() - self.nlanes)
        # 使用 `_int_min()` 为起点，反向的数据
        data_b = self._data(self._int_min(), reverse=True)
        # 载入 `data_a` 和 `data_b` 数据
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)
        # 从当前对象中获取 `intrin` 字符串对应的方法或属性
        intrin = getattr(self, intrin)

        # 获取真值掩码
        mask_true = self._true_mask()
        
        # 定义将向量转换为布尔值的函数
        def to_bool(vector):
            return [lane == mask_true for lane in vector]

        # 对 `data_a` 和 `data_b` 进行函数 `func` 的比较
        data_cmp = [func(a, b) for a, b in zip(data_a, data_b)]
        # 使用 `intrin` 方法对 `vdata_a` 和 `vdata_b` 进行操作，然后将结果转换为布尔值
        cmp = to_bool(intrin(vdata_a, vdata_b))
        # 断言 `cmp` 的结果与 `data_cmp` 一致
        assert cmp == data_cmp

    # 逻辑运算测试函数
    def test_operators_logical(self):
        # 如果是浮点数类型，则使用默认数据
        if self._is_fp():
            data_a = self._data()
        else:
            # 否则，使用 `_int_max() - self.nlanes` 为起点的数据
            data_a = self._data(self._int_max() - self.nlanes)
        # 使用 `_int_min()` 为起点，反向的数据
        data_b = self._data(self._int_min(), reverse=True)
        # 载入 `data_a` 和 `data_b` 数据
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # 如果是浮点数类型，则将 `vdata_a` 和 `vdata_b` 转换为无符号整数
        if self._is_fp():
            data_cast_a = self._to_unsigned(vdata_a)
            data_cast_b = self._to_unsigned(vdata_b)
            cast, cast_data = self._to_unsigned, self._to_unsigned
        else:
            # 否则，直接使用 `data_a` 和 `data_b`
            data_cast_a, data_cast_b = data_a, data_b
            cast, cast_data = lambda a: a, self.load

        # 对 `data_cast_a` 和 `data_cast_b` 进行按位异或操作，并转换为相应类型
        data_xor = cast_data([a ^ b for a, b in zip(data_cast_a, data_cast_b)])
        vxor = cast(self.xor(vdata_a, vdata_b))
        # 断言 `vxor` 的结果与 `data_xor` 一致
        assert vxor == data_xor

        # 对 `data_cast_a` 和 `data_cast_b` 进行按位或操作，并转换为相应类型
        data_or  = cast_data([a | b for a, b in zip(data_cast_a, data_cast_b)])
        vor  = cast(getattr(self, "or")(vdata_a, vdata_b))
        # 断言 `vor` 的结果与 `data_or` 一致
        assert vor == data_or

        # 对 `data_cast_a` 和 `data_cast_b` 进行按位与操作，并转换为相应类型
        data_and = cast_data([a & b for a, b in zip(data_cast_a, data_cast_b)])
        vand = cast(getattr(self, "and")(vdata_a, vdata_b))
        # 断言 `vand` 的结果与 `data_and` 一致
        assert vand == data_and

        # 对 `data_cast_a` 进行按位取反操作，并转换为相应类型
        data_not = cast_data([~a for a in data_cast_a])
        vnot = cast(getattr(self, "not")(vdata_a))
        # 断言 `vnot` 的结果与 `data_not` 一致
        assert vnot == data_not

        # 如果 `self.sfx` 不在 ("u8") 中，则返回
        if self.sfx not in ("u8"):
            return
        # 对 `data_cast_a` 和 `data_cast_b` 进行按位与非操作
        data_andc = [a & ~b for a, b in zip(data_cast_a, data_cast_b)]
        vandc = cast(getattr(self, "andc")(vdata_a, vdata_b))
        # 断言 `vandc` 的结果与 `data_andc` 一致
        assert vandc == data_andc

    # 交叉测试函数，使用参数化测试
    @pytest.mark.parametrize("intrin", ["any", "all"])
    @pytest.mark.parametrize("data", (
        [1, 2, 3, 4],
        [-1, -2, -3, -4],
        [0, 1, 2, 3, 4],
        [0x7f, 0x7fff, 0x7fffffff, 0x7fffffffffffffff],
        [0, -1, -2, -3, 4],
        [0],
        [1],
        [-1]
    ))
    def test_operators_crosstest(self, intrin, data):
        """
        Test intrinsics:
            npyv_any_##SFX
            npyv_all_##SFX
        """
        # 将 `data` 加载为长度为 `self.nlanes` 的数据
        data_a = self.load(data * self.nlanes)
        # 使用 `intrin` 字符串对应的方法或属性
        func = eval(intrin)
        intrin = getattr(self, intrin)
        # 获取期望结果
        desired = func(data_a)
        # 使用 `intrin` 方法对 `data_a` 进行操作，获取 SIMD 结果
        simd = intrin(data_a)
        # 断言 SIMD 结果不为空，并与期望结果一致
        assert not not simd == desired
    def test_conversion_boolean(self):
        # 根据当前后缀生成布尔转换函数
        bsfx = "b" + self.sfx[1:]
        to_boolean = getattr(self.npyv, "cvt_%s_%s" % (bsfx, self.sfx))
        from_boolean = getattr(self.npyv, "cvt_%s_%s" % (self.sfx, bsfx))

        # 生成假值和真值的布尔向量
        false_vb = to_boolean(self.setall(0))
        true_vb  = self.cmpeq(self.setall(0), self.setall(0))
        # 断言假值和真值不相等
        assert false_vb != true_vb

        # 从布尔向量还原为原始类型的值
        false_vsfx = from_boolean(false_vb)
        true_vsfx = from_boolean(true_vb)
        # 断言还原后的值不相等
        assert false_vsfx != true_vsfx

    def test_conversion_expand(self):
        """
        Test expand intrinsics:
            npyv_expand_u16_u8
            npyv_expand_u32_u16
        """
        # 如果后缀不是'u8'或'u16'，则直接返回
        if self.sfx not in ("u8", "u16"):
            return
        # 根据当前后缀确定扩展后的类型
        totype = self.sfx[0]+str(int(self.sfx[1:])*2)
        expand = getattr(self.npyv, f"expand_{totype}_{self.sfx}")
        # 准备数据，以便接近边缘以检测任何偏差
        data  = self._data(self._int_max() - self.nlanes)
        vdata = self.load(data)
        # 执行扩展操作
        edata = expand(vdata)
        # 分别取数据的下半部分和上半部分
        data_lo = data[:self.nlanes//2]
        data_hi = data[self.nlanes//2:]
        # 断言扩展后的结果与分离的数据部分相等
        assert edata == (data_lo, data_hi)

    def test_arithmetic_subadd(self):
        # 如果是浮点数类型，则使用随机数据
        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        # 准备数据B，反向加载最小整数
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # 非饱和加法操作
        data_add = self.load([a + b for a, b in zip(data_a, data_b)]) # 加载用于类型转换
        add  = self.add(vdata_a, vdata_b)
        # 断言加法操作的结果相等
        assert add == data_add
        data_sub  = self.load([a - b for a, b in zip(data_a, data_b)])
        sub  = self.sub(vdata_a, vdata_b)
        # 断言减法操作的结果相等
        assert sub == data_sub

    def test_arithmetic_mul(self):
        # 如果后缀是'u64'或's64'，则直接返回
        if self.sfx in ("u64", "s64"):
            return

        # 如果是浮点数类型，则使用随机数据
        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        # 准备数据B，反向加载最小整数
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # 执行乘法操作
        data_mul = self.load([a * b for a, b in zip(data_a, data_b)])
        mul = self.mul(vdata_a, vdata_b)
        # 断言乘法操作的结果相等
        assert mul == data_mul

    def test_arithmetic_div(self):
        # 如果不是浮点数类型，则直接返回
        if not self._is_fp():
            return

        # 准备数据A和B，分别加载为向量
        data_a, data_b = self._data(), self._data(reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # 执行除法操作
        data_div = self.load([a / b for a, b in zip(data_a, data_b)])
        div = self.div(vdata_a, vdata_b)
        # 断言除法操作的结果相等
        assert div == data_div
    def test_arithmetic_intdiv(self):
        """
        Test integer division intrinsics:
            npyv_divisor_##sfx
            npyv_divc_##sfx
        """
        # 如果是浮点数测试，则直接返回，不进行整数除法测试
        if self._is_fp():
            return

        # 获取整数的最小值
        int_min = self._int_min()

        def trunc_div(a, d):
            """
            Divide towards zero works with large integers > 2^53,
            and wrap around overflow similar to what C does.
            """
            # 特殊情况处理：当被除数为整数最小值且除数为-1时，返回被除数本身
            if d == -1 and a == int_min:
                return a
            # 判断被除数和除数的符号
            sign_a, sign_d = a < 0, d < 0
            # 根据符号确定向零舍入的整数除法
            if a == 0 or sign_a == sign_d:
                return a // d
            return (a + sign_d - sign_a) // d + 1

        # 构造测试数据，包括正数和负数，用于测试整数溢出
        data = [1, -int_min]  # to test overflow
        data += range(0, 2**8, 2**5)
        data += range(0, 2**8, 2**5-1)
        bsize = self._scalar_size()
        if bsize > 8:
            data += range(2**8, 2**16, 2**13)
            data += range(2**8, 2**16, 2**13-1)
        if bsize > 16:
            data += range(2**16, 2**32, 2**29)
            data += range(2**16, 2**32, 2**29-1)
        if bsize > 32:
            data += range(2**32, 2**64, 2**61)
            data += range(2**32, 2**64, 2**61-1)
        
        # 对测试数据进行负数化
        data += [-x for x in data]
        
        # 使用itertools的product函数，生成所有可能的被除数和除数组合，进行测试
        for dividend, divisor in itertools.product(data, data):
            # 将除数转换为所需格式
            divisor = self.setall(divisor)[0]  # cast
            if divisor == 0:
                continue
            # 将被除数加载到所需格式
            dividend = self.load(self._data(dividend))
            # 计算使用向零舍入的整数除法的期望结果
            data_divc = [trunc_div(a, divisor) for a in dividend]
            # 获取除数的参数
            divisor_parms = self.divisor(divisor)
            # 调用被测函数进行除法运算
            divc = self.divc(dividend, divisor_parms)
            # 断言期望结果与被测结果相等
            assert divc == data_divc

    def test_arithmetic_reduce_sum(self):
        """
        Test reduce sum intrinsics:
            npyv_sum_##sfx
        """
        # 如果数据类型不在指定的范围内，则直接返回，不进行求和操作测试
        if self.sfx not in ("u32", "u64", "f32", "f64"):
            return
        
        # 执行 reduce sum 测试
        data = self._data()
        vdata = self.load(data)

        # 计算 Python 原生列表的求和结果
        data_sum = sum(data)
        # 调用被测函数进行求和操作
        vsum = self.sum(vdata)
        # 断言被测求和结果与 Python 原生求和结果相等
        assert vsum == data_sum

    def test_arithmetic_reduce_sumup(self):
        """
        Test extend reduce sum intrinsics:
            npyv_sumup_##sfx
        """
        # 如果数据类型不在指定的范围内，则直接返回，不进行求和操作测试
        if self.sfx not in ("u8", "u16"):
            return
        
        # 执行 extend reduce sum 测试
        rdata = (0, self.nlanes, self._int_min(), self._int_max()-self.nlanes)
        for r in rdata:
            data = self._data(r)
            vdata = self.load(data)
            # 计算 Python 原生列表的求和结果
            data_sum = sum(data)
            # 调用被测函数进行扩展求和操作
            vsum = self.sumup(vdata)
            # 断言被测扩展求和结果与 Python 原生求和结果相等
            assert vsum == data_sum
    # 定义测试函数，用于测试条件性的加减运算，支持所有的数据类型
    def test_mask_conditional(self):
        """
        Conditional addition and subtraction for all supported data types.
        Test intrinsics:
            npyv_ifadd_##SFX, npyv_ifsub_##SFX
        """
        # 载入数据，使用默认顺序
        vdata_a = self.load(self._data())
        # 载入数据，使用反向顺序
        vdata_b = self.load(self._data(reverse=True))
        # 创建真值掩码，比较零向量是否等于自身
        true_mask  = self.cmpeq(self.zero(), self.zero())
        # 创建假值掩码，比较零向量是否不等于自身
        false_mask = self.cmpneq(self.zero(), self.zero())

        # 计算数据的差值
        data_sub = self.sub(vdata_b, vdata_a)
        # 使用真值掩码进行条件减法运算
        ifsub = self.ifsub(true_mask, vdata_b, vdata_a, vdata_b)
        # 断言条件减法的结果与直接计算的差值相等
        assert ifsub == data_sub
        # 使用假值掩码进行条件减法运算
        ifsub = self.ifsub(false_mask, vdata_a, vdata_b, vdata_b)
        # 断言条件减法的结果等于 vdata_b
        assert ifsub == vdata_b

        # 计算数据的和
        data_add = self.add(vdata_b, vdata_a)
        # 使用真值掩码进行条件加法运算
        ifadd = self.ifadd(true_mask, vdata_b, vdata_a, vdata_b)
        # 断言条件加法的结果与直接计算的和相等
        assert ifadd == data_add
        # 使用假值掩码进行条件加法运算
        ifadd = self.ifadd(false_mask, vdata_a, vdata_b, vdata_b)
        # 断言条件加法的结果等于 vdata_b
        assert ifadd == vdata_b

        # 如果不是浮点数类型，则直接返回，不继续执行以下代码
        if not self._is_fp():
            return
        
        # 计算数据的除法
        data_div = self.div(vdata_b, vdata_a)
        # 使用真值掩码进行条件除法运算
        ifdiv = self.ifdiv(true_mask, vdata_b, vdata_a, vdata_b)
        # 断言条件除法的结果与直接计算的除法结果相等
        assert ifdiv == data_div
        # 使用真值掩码进行条件零除法运算
        ifdivz = self.ifdivz(true_mask, vdata_b, vdata_a)
        # 断言条件零除法的结果与直接计算的除法结果相等
        assert ifdivz == data_div
        # 使用假值掩码进行条件除法运算
        ifdiv = self.ifdiv(false_mask, vdata_a, vdata_b, vdata_b)
        # 断言条件除法的结果等于 vdata_b
        assert ifdiv == vdata_b
        # 使用假值掩码进行条件零除法运算
        ifdivz = self.ifdivz(false_mask, vdata_a, vdata_b)
        # 断言条件零除法的结果等于零向量
        assert ifdivz == self.zero()
bool_sfx = ("b8", "b16", "b32", "b64")
int_sfx = ("u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64")
fp_sfx  = ("f32", "f64")
all_sfx = int_sfx + fp_sfx

# 定义测试用例的注册表，映射后缀到对应的测试类
tests_registry = {
    bool_sfx: _SIMD_BOOL,
    int_sfx : _SIMD_INT,
    fp_sfx  : _SIMD_FP,
    ("f32",): _SIMD_FP32,
    ("f64",): _SIMD_FP64,
    all_sfx : _SIMD_ALL
}

# 遍历目标字典中的每个目标和其对应的 npyv 对象
for target_name, npyv in targets.items():
    # 获取当前目标的 SIMD 宽度信息
    simd_width = npyv.simd if npyv else ''
    # 将目标名称按 '__' 分割，用于多目标的情况
    pretty_name = target_name.split('__') # multi-target separator
    
    if len(pretty_name) > 1:
        # 如果有多个目标，将其格式化为形如 '(target1 target2)' 的字符串
        pretty_name = f"({' '.join(pretty_name)})"
    else:
        pretty_name = pretty_name[0]

    skip = ""
    skip_sfx = dict()

    if not npyv:
        # 如果 npyv 不存在，表示当前机器不支持该目标
        skip = f"target '{pretty_name}' isn't supported by current machine"
    elif not npyv.simd:
        # 如果目标支持的 SIMD 特性不存在，则表示不支持该目标
        skip = f"target '{pretty_name}' isn't supported by NPYV"
    else:
        # 检查是否支持单精度和双精度的 SIMD 计算
        if not npyv.simd_f32:
            skip_sfx["f32"] = f"target '{pretty_name}' doesn't support single-precision"
        if not npyv.simd_f64:
            skip_sfx["f64"] = f"target '{pretty_name}' doesn't support double-precision"

    # 遍历测试注册表中的每个后缀列表及其对应的测试类
    for sfxes, cls in tests_registry.items():
        for sfx in sfxes:
            # 获取当前后缀对应的跳过信息，如果没有则使用整体的跳过信息
            skip_m = skip_sfx.get(sfx, skip)
            inhr = (cls,)
            attr = dict(npyv=targets[target_name], sfx=sfx, target_name=target_name)
            # 根据命名规则创建测试类的名称
            tcls = type(f"Test{cls.__name__}_{simd_width}_{target_name}_{sfx}", inhr, attr)
            if skip_m:
                # 如果存在跳过信息，则为该测试类添加 pytest 的跳过标记
                pytest.mark.skip(reason=skip_m)(tcls)
            # 将创建的测试类添加到全局命名空间中
            globals()[tcls.__name__] = tcls
```