# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_afm.py`

```
# 导入所需模块
from io import BytesIO  # 导入 BytesIO 类，用于操作二进制数据流
import pytest  # 导入 pytest 模块，用于编写和运行测试
import logging  # 导入 logging 模块，用于记录日志信息

from matplotlib import _afm  # 导入 matplotlib 中的 _afm 模块
from matplotlib import font_manager as fm  # 导入 matplotlib 中的 font_manager 模块，使用别名 fm

# AFM 测试数据，包含字体文件的元数据和字符度量信息
AFM_TEST_DATA = b"""StartFontMetrics 2.0
Comment Comments are ignored.
Comment Creation Date:Mon Nov 13 12:34:11 GMT 2017
FontName MyFont-Bold
EncodingScheme FontSpecific
FullName My Font Bold
FamilyName Test Fonts
Weight Bold
ItalicAngle 0.0
IsFixedPitch false
UnderlinePosition -100
UnderlineThickness 56,789  # 注意：下划线厚度使用逗号作为小数分隔符
Version 001.000
Notice Copyright \xa9 2017 No one.
FontBBox 0 -321 1234 369
StartCharMetrics 3
C 0 ; WX 250 ; N space ; B 0 0 0 0 ;
C 42 ; WX 1141 ; N foo ; B 40 60 800 360 ;
C 99 ; WX 583 ; N bar ; B 40 -10 543 210 ;
EndCharMetrics
EndFontMetrics
"""


def test_nonascii_str():
    # 测试 _afm._to_str() 函数，确保能正确解码非 ASCII 字符
    inp_str = "привет"  # 包含非 ASCII 字符的字符串
    byte_str = inp_str.encode("utf8")  # 将字符串编码为 UTF-8 字节序列

    ret = _afm._to_str(byte_str)  # 调用 _afm._to_str() 进行解码
    assert ret == inp_str  # 断言解码结果与原始输入字符串相同


def test_parse_header():
    fh = BytesIO(AFM_TEST_DATA)  # 使用测试数据创建 BytesIO 对象
    header = _afm._parse_header(fh)  # 解析字体文件头部信息
    assert header == {
        b'StartFontMetrics': 2.0,
        b'FontName': 'MyFont-Bold',
        b'EncodingScheme': 'FontSpecific',
        b'FullName': 'My Font Bold',
        b'FamilyName': 'Test Fonts',
        b'Weight': 'Bold',
        b'ItalicAngle': 0.0,
        b'IsFixedPitch': False,
        b'UnderlinePosition': -100,
        b'UnderlineThickness': 56.789,  # 下划线厚度被解析为浮点数
        b'Version': '001.000',
        b'Notice': b'Copyright \xa9 2017 No one.',  # 注意：版权信息包含非 ASCII 字符
        b'FontBBox': [0, -321, 1234, 369],  # 字体边界框的坐标信息
        b'StartCharMetrics': 3,  # 字符度量信息的起始位置
    }


def test_parse_char_metrics():
    fh = BytesIO(AFM_TEST_DATA)  # 使用测试数据创建 BytesIO 对象
    _afm._parse_header(fh)  # 定位到字符度量信息的位置
    metrics = _afm._parse_char_metrics(fh)  # 解析字符度量信息
    assert metrics == (
        {0: (250.0, 'space', [0, 0, 0, 0]),  # 字符编码 0 的度量信息
         42: (1141.0, 'foo', [40, 60, 800, 360]),  # 字符编码 42 的度量信息
         99: (583.0, 'bar', [40, -10, 543, 210]),  # 字符编码 99 的度量信息
         },
        {'space': (250.0, 'space', [0, 0, 0, 0]),  # 字符名为 'space' 的度量信息
         'foo': (1141.0, 'foo', [40, 60, 800, 360]),  # 字符名为 'foo' 的度量信息
         'bar': (583.0, 'bar', [40, -10, 543, 210]),  # 字符名为 'bar' 的度量信息
         })


def test_get_familyname_guessed():
    fh = BytesIO(AFM_TEST_DATA)  # 使用测试数据创建 BytesIO 对象
    font = _afm.AFM(fh)  # 使用字节流创建 AFM 对象
    del font._header[b'FamilyName']  # 移除 FamilyName 条目，强制进行猜测
    assert font.get_familyname() == 'My Font'  # 断言猜测的字体家族名称为 'My Font'


def test_font_manager_weight_normalization():
    # 测试字体管理器中字体重量的归一化
    font = _afm.AFM(BytesIO(
        AFM_TEST_DATA.replace(b"Weight Bold\n", b"Weight Custom\n")))  # 替换字体数据中的重量信息
    assert fm.afmFontProperty("", font).weight == "normal"  # 断言修改后的字体重量为 'normal'


@pytest.mark.parametrize(
    "afm_data",
    [
        b"""nope
really nope""",
        b"""StartFontMetrics 2.0
Comment Comments are ignored.
Comment Creation Date:Mon Nov 13 12:34:11 GMT 2017
FontName MyFont-Bold
EncodingScheme FontSpecific""",
    ],
)
def test_bad_afm(afm_data):
    fh = BytesIO(afm_data)  # 使用参数化测试数据创建 BytesIO 对象
    # 使用 pytest 框架来测试 _afm._parse_header 函数是否会引发 RuntimeError 异常
    with pytest.raises(RuntimeError):
        # 调用 _afm._parse_header 函数，预期会抛出 RuntimeError 异常
        _afm._parse_header(fh)
# 使用 pytest 模块的 @pytest.mark.parametrize 装饰器，定义参数化测试函数 test_malformed_header
@pytest.mark.parametrize(
    "afm_data",  # 参数化测试的参数名称为 afm_data
    [
        # 第一个测试参数，包含 AFM 数据的字节字符串
        b"""StartFontMetrics 2.0
Comment Comments are ignored.
Comment Creation Date:Mon Nov 13 12:34:11 GMT 2017
Aardvark bob
FontName MyFont-Bold
EncodingScheme FontSpecific
StartCharMetrics 3""",
        # 第二个测试参数，包含 AFM 数据的字节字符串
        b"""StartFontMetrics 2.0
Comment Comments are ignored.
Comment Creation Date:Mon Nov 13 12:34:11 GMT 2017
ItalicAngle zero degrees
FontName MyFont-Bold
EncodingScheme FontSpecific
StartCharMetrics 3""",
    ],
)
# 定义测试函数 test_malformed_header，接收参数 afm_data 和 caplog（用于捕获日志）
def test_malformed_header(afm_data, caplog):
    # 使用 afm_data 创建一个 BytesIO 对象 fh
    fh = BytesIO(afm_data)
    # 设置 caplog 的日志级别为 ERROR，并在这个级别下运行 _afm._parse_header(fh) 函数
    with caplog.at_level(logging.ERROR):
        _afm._parse_header(fh)

    # 断言捕获的日志记录数量为 1
    assert len(caplog.records) == 1
```