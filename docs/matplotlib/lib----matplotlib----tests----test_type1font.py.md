# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_type1font.py`

```py
# 导入 matplotlib 库中的 _type1font 模块，用于处理 Type1 字体文件
import matplotlib._type1font as t1f
# 导入操作系统路径相关模块
import os.path
# 导入用于生成文本差异的 difflib 模块
import difflib
# 导入用于单元测试的 pytest 模块


# 定义测试函数 test_Type1Font，用于测试 Type1 字体文件的功能
def test_Type1Font():
    # 构建 Type1 字体文件的完整路径
    filename = os.path.join(os.path.dirname(__file__), 'cmr10.pfb')
    # 使用 Type1Font 类创建字体对象 font
    font = t1f.Type1Font(filename)
    
    # 对字体进行倾斜处理，返回倾斜后的字体对象 slanted
    slanted = font.transform({'slant': 1})
    # 对字体进行压缩处理，返回压缩后的字体对象 condensed
    condensed = font.transform({'extend': 0.5})
    
    # 使用二进制模式打开字体文件，读取全部内容到 rawdata
    with open(filename, 'rb') as fd:
        rawdata = fd.read()
    
    # 断言：字体的第一个部分等于字体文件中特定位置的内容
    assert font.parts[0] == rawdata[0x0006:0x10c5]
    # 断言：字体的第二个部分等于字体文件中特定位置的内容
    assert font.parts[1] == rawdata[0x10cb:0x897f]
    # 断言：字体的第三个部分等于字体文件中特定位置的内容
    assert font.parts[2] == rawdata[0x8985:0x8ba6]
    
    # 断言：解密后的字体内容以指定的前缀开始
    assert font.decrypted.startswith(b'dup\n/Private 18 dict dup begin')
    # 断言：解密后的字体内容以指定的后缀结尾
    assert font.decrypted.endswith(b'mark currentfile closefile\n')
    
    # 断言：倾斜后的字体解密内容以指定的前缀开始
    assert slanted.decrypted.startswith(b'dup\n/Private 18 dict dup begin')
    # 断言：倾斜后的字体解密内容以指定的后缀结尾
    assert slanted.decrypted.endswith(b'mark currentfile closefile\n')
    
    # 断言：字体的第一个部分包含特定的唯一标识符
    assert b'UniqueID 5000793' in font.parts[0]
    # 断言：解密后的字体内容包含特定的唯一标识符
    assert b'UniqueID 5000793' in font.decrypted
    
    # 断言：字体对象的位置信息字典中，唯一标识符 'UniqueID' 的位置范围
    assert font._pos['UniqueID'] == [(797, 818), (4483, 4504)]
    
    # 计算字体第一个部分的长度
    len0 = len(font.parts[0])
    # 遍历字体对象的位置信息字典
    for key in font._pos.keys():
        for pos0, pos1 in font._pos[key]:
            if pos0 < len0:
                # 根据位置信息提取数据，考虑解密后的情况
                data = font.parts[0][pos0:pos1]
            else:
                data = font.decrypted[pos0-len0:pos1-len0]
            # 断言：提取的数据以指定的键作为 ASCII 编码字符串开始
            assert data.startswith(f'/{key}'.encode('ascii'))
    
    # 断言：字体对象的位置信息字典中包含特定的键集合
    assert {'FontType', 'FontMatrix', 'PaintType', 'ItalicAngle', 'RD'} < set(font._pos.keys())
    
    # 断言：倾斜后的字体第一个部分不包含特定的唯一标识符
    assert b'UniqueID 5000793' not in slanted.parts[0]
    # 断言：倾斜后的字体解密内容不包含特定的唯一标识符
    assert b'UniqueID 5000793' not in slanted.decrypted
    # 断言：倾斜后的字体对象的位置信息字典中不包含 'UniqueID' 键
    assert 'UniqueID' not in slanted._pos
    
    # 断言：字体的属性中，'Weight' 键对应的值为 'Medium'
    assert font.prop['Weight'] == 'Medium'
    # 断言：字体的属性中，'isFixedPitch' 键对应的值为 False
    assert not font.prop['isFixedPitch']
    # 断言：字体的属性中，'ItalicAngle' 键对应的值为 0
    assert font.prop['ItalicAngle'] == 0
    # 断言：倾斜后的字体的属性中，'ItalicAngle' 键对应的值为 -45
    assert slanted.prop['ItalicAngle'] == -45
    # 断言：字体的属性中，'Encoding' 键对应的列表中第五个元素为 'Pi'
    assert font.prop['Encoding'][5] == 'Pi'
    # 断言：字体的属性中，'CharStrings' 键对应的 'Pi' 键的值类型为 bytes
    assert isinstance(font.prop['CharStrings']['Pi'], bytes)
    # 断言：字体的缩略语字典中，'ND' 键对应的值为 'ND'
    assert font._abbr['ND'] == 'ND'
    
    # 创建 difflib.Differ 对象用于比较两个字体部分内容的差异
    differ = difflib.Differ()
    
    # 比较字体的第一个部分与倾斜后字体的第一个部分的内容差异
    diff = list(differ.compare(
        font.parts[0].decode('latin-1').splitlines(),
        slanted.parts[0].decode('latin-1').splitlines()))
    
    # 遍历比较结果列表，逐行断言包含特定的差异信息
    for line in (
         # 移除 UniqueID
         '- /UniqueID 5000793 def',
         # 修改字体名称
         '- /FontName /CMR10 def',
         '+ /FontName/CMR10_Slant_1000 def',
         # 修改 FontMatrix
         '- /FontMatrix [0.001 0 0 0.001 0 0 ]readonly def',
         '+ /FontMatrix [0.001 0 0.001 0.001 0 0] readonly def',
         # 修改 ItalicAngle
         '-  /ItalicAngle 0 def',
         '+  /ItalicAngle -45.0 def'):
        assert line in diff, 'diff to slanted font must contain %s' % line
    
    # 比较字体的第一个部分与压缩后字体的第一个部分的内容差异
    diff = list(differ.compare(
        font.parts[0].decode('latin-1').splitlines(),
        condensed.parts[0].decode('latin-1').splitlines()))
    for line in (
         # 移除 UniqueID 的定义行
         '- /UniqueID 5000793 def',
         # 修改字体名称的定义行
         '- /FontName /CMR10 def',
         # 向差异记录中添加修改后的字体名称定义行
         '+ /FontName/CMR10_Extend_500 def',
         # 修改 FontMatrix 的定义行
         '- /FontMatrix [0.001 0 0 0.001 0 0 ]readonly def',
         # 向差异记录中添加修改后的 FontMatrix 定义行
         '+ /FontMatrix [0.0005 0 0 0.001 0 0] readonly def'):
        # 断言当前行存在于差异记录中，以确保修改的正确性
        assert line in diff, 'diff to condensed font must contain %s' % line
# 测试函数，用于测试 Type1Font 类的功能
def test_Type1Font_2():
    # 设置文件名为当前文件夹下的 Courier10PitchBT-Bold.pfb 文件路径
    filename = os.path.join(os.path.dirname(__file__), 'Courier10PitchBT-Bold.pfb')
    # 使用 Type1Font 类加载指定字体文件
    font = t1f.Type1Font(filename)
    # 断言字体属性中的 'Weight' 属性为 'Bold'
    assert font.prop['Weight'] == 'Bold'
    # 断言字体属性中的 'isFixedPitch' 属性为 True
    assert font.prop['isFixedPitch']
    # 断言字体属性中的 'Encoding' 属性中索引 65 的值为 'A'，表明字体使用 StandardEncoding
    assert font.prop['Encoding'][65] == 'A'
    # 获取字体部分中与 'Encoding' 相关的位置信息
    (pos0, pos1), = font._pos['Encoding']
    # 断言字体部分中指定位置的数据符合预期的字节序列
    assert font.parts[0][pos0:pos1] == b'/Encoding StandardEncoding'
    # 断言字体缩略语 '_abbr' 中 'ND' 键对应的值为 '|-'
    assert font._abbr['ND'] == '|-'


# 测试函数，用于测试 tokenization 过程
def test_tokenize():
    # 定义包含特定数据的字节序列
    data = (b'1234/abc false -9.81  Foo <<[0 1 2]<0 1ef a\t>>>\n'
            b'(string with(nested\t\\) par)ens\\\\)')
    # 定义 token 类型的名称缩写
    n, w, num, kw, d = 'name', 'whitespace', 'number', 'keyword', 'delimiter'
    b, s = 'boolean', 'string'
    # 定义预期的正确 token 序列，包含不同类型的 token 及其对应值
    correct = [
        (num, 1234), (n, 'abc'), (w, ' '), (b, False), (w, ' '), (num, -9.81),
        (w, '  '), (kw, 'Foo'), (w, ' '), (d, '<<'), (d, '['), (num, 0),
        (w, ' '), (num, 1), (w, ' '), (num, 2), (d, ']'), (s, b'\x01\xef\xa0'),
        (d, '>>'), (w, '\n'), (s, 'string with(nested\t) par)ens\\')
    ]
    # 去除正确序列中的所有空白字符 token，得到预期的无空白字符的正确序列
    correct_no_ws = [x for x in correct if x[0] != w]

    # 定义将 token 流转换为元组列表的辅助函数
    def convert(tokens):
        return [(t.kind, t.value()) for t in tokens]

    # 断言对给定数据进行 tokenization 的结果与预期的正确 token 序列相匹配
    assert convert(t1f._tokenize(data, False)) == correct
    # 断言对给定数据进行 tokenization 的结果去除空白字符后与预期的无空白字符的正确序列相匹配
    assert convert(t1f._tokenize(data, True)) == correct_no_ws

    # 定义从 token 流中获取指定数量 token 的辅助函数
    def bin_after(n):
        tokens = t1f._tokenize(data, True)
        result = []
        for _ in range(n):
            result.append(next(tokens))
        result.append(tokens.send(10))
        return convert(result)

    # 循环测试获取不同数量 token 后的结果是否符合预期
    for n in range(1, len(correct_no_ws)):
        result = bin_after(n)
        # 断言结果的前 n 个 token 与预期的无空白字符的正确序列的前 n 个元素相匹配
        assert result[:-1] == correct_no_ws[:n]
        # 断言结果的最后一个 token 的类型为 'binary'
        assert result[-1][0] == 'binary'
        # 断言结果的最后一个 token 的值为 bytes 类型
        assert isinstance(result[-1][1], bytes)


# 测试函数，用于测试 tokenization 过程中的错误处理
def test_tokenize_errors():
    # 断言对包含未终止的字符串的数据进行 tokenization 会引发 ValueError 异常
    with pytest.raises(ValueError):
        list(t1f._tokenize(b'1234 (this (string) is unterminated\\)', True))
    # 断言对包含非法格式的数据进行 tokenization 会引发 ValueError 异常
    with pytest.raises(ValueError):
        list(t1f._tokenize(b'/Foo<01234', True))
    # 断言对包含非法格式的数据进行 tokenization 会引发 ValueError 异常
    with pytest.raises(ValueError):
        list(t1f._tokenize(b'/Foo<01234abcg>/Bar', True))


# 测试函数，用于测试处理过度精度的问题
def test_overprecision():
    # 使用 cmr10.pfb 字体文件创建 Type1Font 对象
    filename = os.path.join(os.path.dirname(__file__), 'cmr10.pfb')
    font = t1f.Type1Font(filename)
    # 对字体进行斜体转换
    slanted = font.transform({'slant': .167})
    # 将斜体转换后的数据按行解码并分割
    lines = slanted.parts[0].decode('ascii').splitlines()
    # 获取包含 '/FontMatrix' 的行中的矩阵数据
    matrix, = [line[line.index('[')+1:line.index(']')]
               for line in lines if '/FontMatrix' in line]
    # 获取包含 '/ItalicAngle' 的行中的角度数据
    angle, = [word
              for line in lines if '/ItalicAngle' in line
              for word in line.split() if word[0] in '-0123456789']
    # 断言矩阵数据符合预期的精度
    assert matrix == '0.001 0 0.000167 0.001 0 0'
    # 断言角度数据符合预期的精度
    assert angle == '-9.4809'
# 定义一个测试函数，用于测试加密和解密的往返过程
def test_encrypt_decrypt_roundtrip():
    # 定义一个二进制数据作为测试的明文数据
    data = b'this is my plaintext \0\1\2\3'
    # 调用 Type1Font 类的 _encrypt 方法，对明文数据进行加密
    encrypted = t1f.Type1Font._encrypt(data, 'eexec')
    # 调用 Type1Font 类的 _decrypt 方法，对加密后的数据进行解密
    decrypted = t1f.Type1Font._decrypt(encrypted, 'eexec')
    # 断言加密后的数据与解密后的数据不相同
    assert encrypted != decrypted
    # 断言解密后的数据与原始明文数据相同，验证解密的正确性
    assert data == decrypted
```