# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_graycode.py`

```
# 导入必要的库和模块
from sympy.combinatorics.graycode import (GrayCode, bin_to_gray,
    random_bitstring, get_subset_from_bitstring, graycode_subsets,
    gray_to_bin)
from sympy.testing.pytest import raises

# 定义测试函数 test_graycode
def test_graycode():
    # 创建 GrayCode 对象 g，生成长度为 2 的格雷码序列
    g = GrayCode(2)
    # 初始化空列表用于存储生成的格雷码
    got = []
    # 生成格雷码序列并遍历
    for i in g.generate_gray():
        # 如果格雷码以 '0' 开头，则跳过当前格雷码
        if i.startswith('0'):
            g.skip()
        # 将当前格雷码添加到列表中
        got.append(i)
    # 断言生成的格雷码列表与预期结果相等
    assert got == '00 11 10'.split()
    
    # 创建 GrayCode 对象 a，生成长度为 6 的格雷码序列
    a = GrayCode(6)
    # 断言当前格雷码序列是否全为 '0'
    assert a.current == '0'*6
    # 断言当前格雷码序列的排名是否为 0
    assert a.rank == 0
    # 断言生成的格雷码序列长度是否为 64
    assert len(list(a.generate_gray())) == 64
    
    # 预定义的格雷码序列列表
    codes = ['011001', '011011', '011010', '011110', '011111', '011101',
             '011100', '010100', '010101', '010111', '010110', '010010',
             '010011', '010001', '010000', '110000', '110001', '110011',
             '110010', '110110', '110111', '110101', '110100', '111100',
             '111101', '111111', '111110', '111010', '111011', '111001',
             '111000', '101000', '101001', '101011', '101010', '101110',
             '101111', '101101', '101100', '100100', '100101', '100111',
             '100110', '100010', '100011', '100001', '100000']
    # 断言从指定起始格雷码生成的格雷码序列是否与预期结果相等
    assert list(a.generate_gray(start='011001')) == codes
    # 断言从指定排名生成的格雷码序列是否与预期结果相等
    assert list(a.generate_gray(rank=GrayCode(6, start='011001').rank)) == codes
    # 断言下一个格雷码对象的当前序列是否为 '000001'
    assert a.next().current == '000001'
    # 断言下两个格雷码对象的当前序列是否为 '000011'
    assert a.next(2).current == '000011'
    # 断言上一个格雷码对象的当前序列是否为 '100000'
    assert a.next(-1).current == '100000'
    
    # 创建指定起始序列的长度为 5 的格雷码对象 a
    a = GrayCode(5, start='10010')
    # 断言当前格雷码对象的排名是否为 28
    assert a.rank == 28
    # 创建指定起始序列的长度为 6 的格雷码对象 a
    a = GrayCode(6, start='101000')
    # 断言当前格雷码对象的排名是否为 48
    assert a.rank == 48
    
    # 断言创建指定排名的长度为 6 的格雷码对象的当前序列是否为 '000110'
    assert GrayCode(6, rank=4).current == '000110'
    # 断言创建指定排名的长度为 6 的格雷码对象的排名是否为 4
    assert GrayCode(6, rank=4).rank == 4
    # 断言根据起始序列生成的格雷码对象的排名列表是否与预期结果相等
    assert [GrayCode(4, start=s).rank for s in GrayCode(4).generate_gray()] == [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                                                   9, 10, 11, 12, 13, 14, 15]
    # 创建指定排名的长度为 15 的格雷码对象 a
    a = GrayCode(15, rank=15)
    # 断言当前格雷码对象的当前序列是否为 '000000000001000'
    assert a.current == '000000000001000'
    
    # 断言将二进制字符串 '111' 转换为格雷码的结果是否为 '100'
    assert bin_to_gray('111') == '100'
    
    # 创建长度为 5 的随机二进制字符串
    a = random_bitstring(5)
    # 断言生成的随机二进制字符串的类型是否为 str
    assert type(a) is str
    # 断言生成的随机二进制字符串的长度是否为 5
    assert len(a) == 5
    # 断言生成的随机二进制字符串是否只包含 '0' 和 '1'
    assert all(i in ['0', '1'] for i in a)
    
    # 断言根据指定的位字符串获取子集的结果是否与预期结果相等
    assert get_subset_from_bitstring(['a', 'b', 'c', 'd'], '0011') == ['c', 'd']
    # 断言根据指定的位字符串获取子集的结果是否与预期结果相等
    assert get_subset_from_bitstring('abcd', '1001') == ['a', 'd']
    # 断言生成给定集合的所有格雷码子集的结果是否与预期结果相等
    assert list(graycode_subsets(['a', 'b', 'c'])) == \
        [[], ['c'], ['b', 'c'], ['b'], ['a', 'b'], ['a', 'b', 'c'],
         ['a', 'c'], ['a']]
    
    # 断言在创建长度为 0 的格雷码对象时抛出 ValueError 异常
    raises(ValueError, lambda: GrayCode(0))
    # 断言在创建长度为 2.2 的格雷码对象时抛出 ValueError 异常
    raises(ValueError, lambda: GrayCode(2.2))
    # 断言在创建指定起始序列为非二进制字符串时抛出 ValueError 异常
    raises(ValueError, lambda: GrayCode(2, start=[1, 1, 0]))
    # 断言在创建指定排名为非整数时抛出 ValueError 异常
    raises(ValueError, lambda: GrayCode(2, rank=2.5))
    # 断言在根据非法位字符串获取子集时抛出 ValueError 异常
    raises(ValueError, lambda: get_subset_from_bitstring(['c', 'a', 'c'], '1100'))
    # 断言在生成指定起始序列的格雷码对象时抛出 ValueError 异常
    raises(ValueError, lambda: list(GrayCode(3).generate_gray(start="1111")))


# 定义测试函数 test_live_issue_117，用于测试指定问题的格雷码转换
def test_live_issue_117():
    # 断言将二进制字符串 '0100' 转换为格雷码的结果是否为 '0110'
    assert bin_to_gray('0100') == '0110'
    # 断言将二进制字符串 '0101' 转换为格雷码的结果是否为 '0111'
    assert bin_to_gray('0101') == '0111'
    # 遍历指定二进制字符串列表，断言经格雷码转换后再转回二进制字符串结果不变
    for bits in ('0100', '0101'):
        assert gray_to_bin(bin_to_gray(bits)) == bits
```