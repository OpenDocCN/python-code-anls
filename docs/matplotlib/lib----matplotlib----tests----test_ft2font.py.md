# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_ft2font.py`

```
# 从 pathlib 模块中导入 Path 类，用于处理文件路径
from pathlib import Path
# 导入 io 模块，用于处理文件流
import io
# 导入 pytest 模块，用于编写和运行测试
import pytest
# 导入 matplotlib 的 ft2font 模块，用于处理字体文件
from matplotlib import ft2font
# 导入 matplotlib.testing.decorators 模块中的 check_figures_equal 装饰器，用于比较图像
from matplotlib.testing.decorators import check_figures_equal
# 导入 matplotlib.font_manager 模块，用于管理字体
import matplotlib.font_manager as fm
# 导入 matplotlib.pyplot 模块，用于绘图
import matplotlib.pyplot as plt


# 定义测试函数 test_fallback_errors，测试 FT2Font 对象的错误处理能力
def test_fallback_errors():
    # 查找指定字体的文件路径
    file_name = fm.findfont('DejaVu Sans')

    # 测试当 _fallback_list 不是列表时是否抛出 TypeError 异常
    with pytest.raises(TypeError, match="Fallback list must be a list"):
        ft2font.FT2Font(file_name, _fallback_list=(0,))  # type: ignore[arg-type]

    # 测试当 _fallback_list 中的元素不是 FT2Font 对象时是否抛出 TypeError 异常
    with pytest.raises(
            TypeError, match="Fallback fonts must be FT2Font objects."
    ):
        ft2font.FT2Font(file_name, _fallback_list=[0])  # type: ignore[list-item]


# 定义测试函数 test_ft2font_positive_hinting_factor，测试 hinting_factor 参数的合法性
def test_ft2font_positive_hinting_factor():
    # 查找指定字体的文件路径
    file_name = fm.findfont('DejaVu Sans')
    # 测试当 hinting_factor 小于等于 0 时是否抛出 ValueError 异常
    with pytest.raises(
            ValueError, match="hinting_factor must be greater than 0"
    ):
        ft2font.FT2Font(file_name, 0)


# 使用 pytest.mark.parametrize 装饰器定义参数化测试函数 test_fallback_smoke，测试字体的回退机制
@pytest.mark.parametrize('family_name, file_name',
                          [("WenQuanYi Zen Hei",  "wqy-zenhei.ttc"),
                           ("Noto Sans CJK JP", "NotoSansCJK.ttc"),
                           ("Noto Sans TC", "NotoSansTC-Regular.otf")]
                         )
def test_fallback_smoke(family_name, file_name):
    # 创建 FontProperties 对象，设置字体族
    fp = fm.FontProperties(family=[family_name])
    # 如果当前系统中找不到指定的字体文件，则跳过测试
    if Path(fm.findfont(fp)).name != file_name:
        pytest.skip(f"Font {family_name} ({file_name}) is missing")
    # 设置全局的字体大小
    plt.rcParams['font.size'] = 20
    # 创建指定大小的图形对象
    fig = plt.figure(figsize=(4.75, 1.85))
    # 在图形中添加文本，使用指定的字体族
    fig.text(0.05, 0.45, "There are 几个汉字 in between!",
             family=['DejaVu Sans', family_name])
    fig.text(0.05, 0.85, "There are 几个汉字 in between!",
             family=[family_name])

    # 遍历保存图像的不同格式
    for fmt in ['png', 'raw']:  # ["svg", "pdf", "ps"]:
        # 将图形保存为指定格式的字节流
        fig.savefig(io.BytesIO(), format=fmt)


# 使用 pytest.mark.parametrize 装饰器定义参数化测试函数 test_font_fallback_chinese，测试中文字体回退
@pytest.mark.parametrize('family_name, file_name',
                         [("WenQuanYi Zen Hei",  "wqy-zenhei"),
                          ("Noto Sans CJK JP", "NotoSansCJK"),
                          ("Noto Sans TC", "NotoSansTC-Regular.otf")]
                         )
# 使用 check_figures_equal 装饰器比较测试结果与参考结果的图像
@check_figures_equal(extensions=["png", "pdf", "eps", "svg"])
def test_font_fallback_chinese(fig_test, fig_ref, family_name, file_name):
    # 创建 FontProperties 对象，设置字体族
    fp = fm.FontProperties(family=[family_name])
    # 如果当前系统中找不到指定的字体文件，则跳过测试
    if file_name not in Path(fm.findfont(fp)).name:
        pytest.skip(f"Font {family_name} ({file_name}) is missing")

    # 定义要绘制的文本内容
    text = ["There are", "几个汉字", "in between!"]

    # 设置全局的字体大小
    plt.rcParams["font.size"] = 20
    # 定义测试字体族和参考字体族的列表
    test_fonts = [["DejaVu Sans", family_name]] * 3
    ref_fonts = [["DejaVu Sans"], [family_name], ["DejaVu Sans"]]

    # 遍历文本内容和字体族列表，向参考图和测试图添加文本
    for j, (txt, test_font, ref_font) in enumerate(
            zip(text, test_fonts, ref_fonts)
    ):
        fig_ref.text(0.05, .85 - 0.15*j, txt, family=ref_font)
        fig_test.text(0.05, .85 - 0.15*j, txt, family=test_font)
# 使用 pytest.mark.parametrize 装饰器标记测试函数，定义多组参数化测试数据
@pytest.mark.parametrize("font_list",
                         [['DejaVu Serif', 'DejaVu Sans'],  # 第一组测试数据
                          ['DejaVu Sans Mono']],          # 第二组测试数据
                         ids=["two fonts", "one font"])    # 指定每组数据的标识符

# 定义测试函数 test_fallback_missing，接受 recwarn 和 font_list 参数
def test_fallback_missing(recwarn, font_list):
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形中心位置添加文本 "Hello 🙃 World!"，指定字体族为 font_list 中的值
    fig.text(.5, .5, "Hello 🙃 World!", family=font_list)
    # 绘制图形
    fig.canvas.draw()
    # 断言：检查 recwarn 中的每个警告消息是否为 UserWarning 类型
    assert all(isinstance(warn.message, UserWarning) for warn in recwarn)
    # 断言：验证警告消息的内容以指定字符串开头
    assert recwarn[0].message.args[0].startswith(
           "Glyph 128579 (\\N{UPSIDE-DOWN FACE}) missing from font(s)")
    # 断言：验证警告消息中包含 font_list 中的每个字体名称
    assert all([font in recwarn[0].message.args[0] for font in font_list])


# 使用 pytest.mark.parametrize 装饰器标记测试函数，定义多组参数化测试数据
@pytest.mark.parametrize(
    "family_name, file_name",
    [
        ("WenQuanYi Zen Hei", "wqy-zenhei"),          # 第一组测试数据
        ("Noto Sans CJK JP", "NotoSansCJK"),          # 第二组测试数据
        ("Noto Sans TC", "NotoSansTC-Regular.otf")   # 第三组测试数据
    ],
)
# 定义测试函数 test__get_fontmap，接受 family_name 和 file_name 参数
def test__get_fontmap(family_name, file_name):
    # 创建 FontProperties 对象，指定字体族为 family_name
    fp = fm.FontProperties(family=[family_name])
    # 查找符合 FontProperties 条件的字体文件路径，并获取其文件名
    found_file_name = Path(fm.findfont(fp)).name
    # 如果 file_name 不在找到的文件名中，则跳过当前测试并输出相应信息
    if file_name not in found_file_name:
        pytest.skip(f"Font {family_name} ({file_name}) is missing")

    # 定义测试文本
    text = "There are 几个汉字 in between!"
    # 获取包含指定字体族的字体对象
    ft = fm.get_font(
        fm.fontManager._find_fonts_by_props(
            fm.FontProperties(family=["DejaVu Sans", family_name])
        )
    )
    # 获取文本的字体映射信息
    fontmap = ft._get_fontmap(text)
    # 遍历字体映射中的每个字符和对应的字体信息
    for char, font in fontmap.items():
        # 如果字符的 Unicode 编码大于 127，则验证其字体文件名是否与 found_file_name 相同
        if ord(char) > 127:
            assert Path(font.fname).name == found_file_name
        else:
            # 否则验证其字体文件名是否为 "DejaVuSans.ttf"
            assert Path(font.fname).name == "DejaVuSans.ttf"
```