# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_sankey.py`

```py
import pytest  # 导入 pytest 库
from numpy.testing import assert_allclose, assert_array_equal  # 从 numpy.testing 导入两个断言函数

from matplotlib.sankey import Sankey  # 导入 Sankey 类
from matplotlib.testing.decorators import check_figures_equal  # 导入 check_figures_equal 装饰器


def test_sankey():
    # 创建一个 Sankey 实例并运行 add 方法，用于测试代码是否正常运行
    sankey = Sankey()
    sankey.add()


def test_label():
    # 测试设置标签功能
    s = Sankey(flows=[0.25], labels=['First'], orientations=[-1])
    assert s.diagrams[0].texts[0].get_text() == 'First\n0.25'


def test_format_using_callable():
    # 使用可调用函数测试格式化标签的功能
    def show_three_decimal_places(value):
        return f'{value:.3f}'

    s = Sankey(flows=[0.25], labels=['First'], orientations=[-1],
               format=show_three_decimal_places)

    assert s.diagrams[0].texts[0].get_text() == 'First\n0.250'


@pytest.mark.parametrize('kwargs, msg', (
    ({'gap': -1}, "'gap' is negative"),
    ({'gap': 1, 'radius': 2}, "'radius' is greater than 'gap'"),
    ({'head_angle': -1}, "'head_angle' is negative"),
    ({'tolerance': -1}, "'tolerance' is negative"),
    ({'flows': [1, -1], 'orientations': [-1, 0, 1]},
     r"The shapes of 'flows' \(2,\) and 'orientations'"),
    ({'flows': [1, -1], 'labels': ['a', 'b', 'c']},
     r"The shapes of 'flows' \(2,\) and 'labels'"),
    ))
def test_sankey_errors(kwargs, msg):
    # 使用 pytest.mark.parametrize 进行参数化测试，验证 Sankey 类的错误输入处理
    with pytest.raises(ValueError, match=msg):
        Sankey(**kwargs)


@pytest.mark.parametrize('kwargs, msg', (
    ({'trunklength': -1}, "'trunklength' is negative"),
    ({'flows': [0.2, 0.3], 'prior': 0}, "The scaled sum of the connected"),
    ({'prior': -1}, "The index of the prior diagram is negative"),
    ({'prior': 1}, "The index of the prior diagram is 1"),
    ({'connect': (-1, 1), 'prior': 0}, "At least one of the connection"),
    ({'connect': (2, 1), 'prior': 0}, "The connection index to the source"),
    ({'connect': (1, 3), 'prior': 0}, "The connection index to this dia"),
    ({'connect': (1, 1), 'prior': 0, 'flows': [-0.2, 0.2],
      'orientations': [2]}, "The value of orientations"),
    ({'connect': (1, 1), 'prior': 0, 'flows': [-0.2, 0.2],
      'pathlengths': [2]}, "The lengths of 'flows'"),
    ))
def test_sankey_add_errors(kwargs, msg):
    # 使用 pytest.mark.parametrize 进行参数化测试，验证 Sankey 类的 add 方法的错误输入处理
    sankey = Sankey()
    with pytest.raises(ValueError, match=msg):
        sankey.add(flows=[0.2, -0.2])
        sankey.add(**kwargs)


def test_sankey2():
    # 测试 Sankey 类的更复杂用例，包括 flows、labels、orientations 和 unit 参数
    s = Sankey(flows=[0.25, -0.25, 0.5, -0.5], labels=['Foo'],
               orientations=[-1], unit='Bar')
    sf = s.finish()
    assert_array_equal(sf[0].flows, [0.25, -0.25, 0.5, -0.5])
    assert sf[0].angles == [1, 3, 1, 3]
    assert all([text.get_text()[0:3] == 'Foo' for text in sf[0].texts])
    assert all([text.get_text()[-3:] == 'Bar' for text in sf[0].texts])
    assert sf[0].text.get_text() == ''
    assert_allclose(sf[0].tips,
                    [(-1.375, -0.52011255),
                     (1.375, -0.75506044),
                     (-0.75, -0.41522509),
                     (0.75, -0.8599479)])
    # 创建 Sankey 对象，并设置流量、标签、方向和单位
    s = Sankey(flows=[0.25, -0.25, 0, 0.5, -0.5], labels=['Foo'],
               orientations=[-1], unit='Bar')
    # 完成 Sankey 图的构建，得到最终的图形对象
    sf = s.finish()
    # 断言 Sankey 图中第一个图块的流量属性与预期一致
    assert_array_equal(sf[0].flows, [0.25, -0.25, 0, 0.5, -0.5])
    # 断言 Sankey 图中第一个图块的角度属性与预期一致
    assert sf[0].angles == [1, 3, None, 1, 3]
    # 断言 Sankey 图中第一个图块的尖端位置与预期非常接近
    assert_allclose(sf[0].tips,
                    [(-1.375, -0.52011255),
                     (1.375, -0.75506044),
                     (0, 0),
                     (-0.75, -0.41522509),
                     (0.75, -0.8599479)])
# 使用装饰器检查两个图形是否相等，限定扩展名为 'png'
@check_figures_equal(extensions=['png'])
# 定义测试函数 test_sankey3，接收两个图形对象 fig_test 和 fig_ref
def test_sankey3(fig_test, fig_ref):
    # 获取 fig_test 的当前 Axes 对象
    ax_test = fig_test.gca()
    # 创建 Sankey 图表对象 s_test，指定轮流的流量和方向
    s_test = Sankey(ax=ax_test, flows=[0.25, -0.25, -0.25, 0.25, 0.5, -0.5],
                    orientations=[1, -1, 1, -1, 0, 0])
    # 完成 s_test 图表的绘制
    s_test.finish()

    # 获取 fig_ref 的当前 Axes 对象
    ax_ref = fig_ref.gca()
    # 创建 Sankey 图表对象 s_ref，没有指定流量和方向，稍后添加
    s_ref = Sankey(ax=ax_ref)
    # 添加流量和方向到 s_ref 图表对象
    s_ref.add(flows=[0.25, -0.25, -0.25, 0.25, 0.5, -0.5],
              orientations=[1, -1, 1, -1, 0, 0])
    # 完成 s_ref 图表的绘制
    s_ref.finish()
```