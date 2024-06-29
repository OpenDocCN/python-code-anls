# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_dviread.py`

```py
# 导入必要的模块：json用于处理JSON数据，Path用于处理文件路径，shutil用于文件操作，matplotlib.dviread是一个特定的模块，pytest用于编写和运行测试
import json
from pathlib import Path
import shutil

import matplotlib.dviread as dr
import pytest

# 定义测试函数test_PsfontsMap，接受monkeypatch作为参数
def test_PsfontsMap(monkeypatch):
    # 用lambda函数替换dr模块中的find_tex_file函数，使其返回解码后的输入值
    monkeypatch.setattr(dr, 'find_tex_file', lambda x: x.decode())

    # 构造文件路径，指向'baseline_images/dviread/test.map'文件
    filename = str(Path(__file__).parent / 'baseline_images/dviread/test.map')
    # 使用filename初始化PsfontsMap对象
    fontmap = dr.PsfontsMap(filename)
    
    # 检查几个字体的所有属性
    for n in [1, 2, 3, 4, 5]:
        key = b'TeXfont%d' % n
        entry = fontmap[key]
        # 断言字体的TeX名称与key相同
        assert entry.texname == key
        # 断言字体的PS名称与b'PSfont%d' % n相同
        assert entry.psname == b'PSfont%d' % n
        # 根据n的不同情况，断言字体的编码方式
        if n not in [3, 5]:
            assert entry.encoding == 'font%d.enc' % n
        elif n == 3:
            assert entry.encoding == 'enc3.foo'
        # 对于TeXfont5不关心其编码，因为它指定了多个编码
        if n not in [1, 5]:
            assert entry.filename == 'font%d.pfa' % n
        else:
            assert entry.filename == 'font%d.pfb' % n
        # 对于TeXfont4特别断言其效果
        if n == 4:
            assert entry.effects == {'slant': -0.1, 'extend': 1.2}
        else:
            assert entry.effects == {}
    
    # 处理特殊情况的断言
    entry = fontmap[b'TeXfont6']
    assert entry.filename is None
    assert entry.encoding is None
    
    entry = fontmap[b'TeXfont7']
    assert entry.filename is None
    assert entry.encoding == 'font7.enc'
    
    entry = fontmap[b'TeXfont8']
    assert entry.filename == 'font8.pfb'
    assert entry.encoding is None
    
    entry = fontmap[b'TeXfont9']
    assert entry.psname == b'TeXfont9'
    assert entry.filename == '/absolute/font9.pfb'
    
    # 对于重复的第一个条目的断言
    entry = fontmap[b'TeXfontA']
    assert entry.psname == b'PSfontA1'
    
    # 斜体和扩展仅适用于T1字体的断言
    entry = fontmap[b'TeXfontB']
    assert entry.psname == b'PSfontB6'
    
    # 子集化的TrueType字体必须有编码的断言
    entry = fontmap[b'TeXfontC']
    assert entry.psname == b'PSfontC3'
    
    # 未找到字体的情况下会抛出LookupError异常
    with pytest.raises(LookupError, match='no-such-font'):
        fontmap[b'no-such-font']
    
    with pytest.raises(LookupError, match='%'):
        fontmap[b'%']

# 根据shutil.which("kpsewhich")的可用性决定是否跳过测试
@pytest.mark.skipif(shutil.which("kpsewhich") is None,
                    reason="kpsewhich is not available")
def test_dviread():
    # 构造目录路径，指向'baseline_images/dviread'目录
    dirpath = Path(__file__).parent / 'baseline_images/dviread'
    # 使用'test.json'文件初始化correct字典
    with (dirpath / 'test.json').open() as f:
        correct = json.load(f)
    # 使用'test.dvi'文件初始化Dvi对象，并使用None作为参数
    with dr.Dvi(str(dirpath / 'test.dvi'), None) as dvi:
        # 初始化data列表，遍历dvi对象的每一页
        data = [{'text': [[t.x, t.y,
                           chr(t.glyph),
                           t.font.texname.decode('ascii'),
                           round(t.font.size, 2)]
                          for t in page.text],
                 'boxes': [[b.x, b.y, b.height, b.width] for b in page.boxes]}
                for page in dvi]
    
    # 断言data与correct相同
    assert data == correct
```