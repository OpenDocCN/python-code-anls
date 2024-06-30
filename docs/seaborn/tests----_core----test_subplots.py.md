# `D:\src\scipysrc\seaborn\tests\_core\test_subplots.py`

```
    # 导入所需模块和类
    import itertools

    import numpy as np
    import pytest

    # 从 seaborn 库中导入 Subplots 类
    from seaborn._core.subplots import Subplots


    class TestSpecificationChecks:

        # 测试在同时指定 `col` 和 `row` 时，不能包装 facets 的情况
        def test_both_facets_and_wrap(self):

            # 定义错误消息
            err = "Cannot wrap facets when specifying both `col` and `row`."
            # 指定包装值为 3，并且指定 `col` 和 `row`
            facet_spec = {"wrap": 3, "variables": {"col": "a", "row": "b"}}
            # 使用 pytest 来检查是否会抛出 RuntimeError，并且匹配特定错误消息
            with pytest.raises(RuntimeError, match=err):
                Subplots({}, facet_spec, {})

        # 测试在同时对 `x` 和 `y` 进行配对时，不能包装子图的情况
        def test_cross_xy_pairing_and_wrap(self):

            # 定义错误消息
            err = "Cannot wrap subplots when pairing on both `x` and `y`."
            # 指定包装值为 3，并且指定 `x` 和 `y` 的结构
            pair_spec = {"wrap": 3, "structure": {"x": ["a", "b"], "y": ["y", "z"]}}
            # 使用 pytest 来检查是否会抛出 RuntimeError，并且匹配特定错误消息
            with pytest.raises(RuntimeError, match=err):
                Subplots({}, {}, pair_spec)

        # 测试在对 `x` 进行配对时，不能对列进行 facets 的情况
        def test_col_facets_and_x_pairing(self):

            # 定义错误消息
            err = "Cannot facet the columns while pairing on `x`."
            # 指定列的 facets 结构
            facet_spec = {"variables": {"col": "a"}}
            # 指定 `x` 的结构
            pair_spec = {"structure": {"x": ["x", "y"]}}
            # 使用 pytest 来检查是否会抛出 RuntimeError，并且匹配特定错误消息
            with pytest.raises(RuntimeError, match=err):
                Subplots({}, facet_spec, pair_spec)

        # 测试在对 `y` 进行配对时，不能包装列的情况
        def test_wrapped_columns_and_y_pairing(self):

            # 定义错误消息
            err = "Cannot wrap the columns while pairing on `y`."
            # 指定列的 facets 结构，并且包装值为 2
            facet_spec = {"variables": {"col": "a"}, "wrap": 2}
            # 指定 `y` 的结构
            pair_spec = {"structure": {"y": ["x", "y"]}}
            # 使用 pytest 来检查是否会抛出 RuntimeError，并且匹配特定错误消息
            with pytest.raises(RuntimeError, match=err):
                Subplots({}, facet_spec, pair_spec)

        # 测试在对 `x` 进行配对时，不能在 facets 中包装行的情况
        def test_wrapped_x_pairing_and_facetd_rows(self):

            # 定义错误消息
            err = "Cannot wrap the columns while faceting the rows."
            # 指定行的 facets 结构，并且包装值为 2
            facet_spec = {"variables": {"row": "a"}}
            # 指定 `x` 的结构，并且包装值为 2
            pair_spec = {"structure": {"x": ["x", "y"]}, "wrap": 2}
            # 使用 pytest 来检查是否会抛出 RuntimeError，并且匹配特定错误消息
            with pytest.raises(RuntimeError, match=err):
                Subplots({}, facet_spec, pair_spec)


    class TestSubplotSpec:

        # 测试创建单个子图的情况
        def test_single_subplot(self):

            # 创建 Subplots 对象
            s = Subplots({}, {}, {})

            # 断言单个子图的数量为 1
            assert s.n_subplots == 1
            # 断言子图的列数为 1
            assert s.subplot_spec["ncols"] == 1
            # 断言子图的行数为 1
            assert s.subplot_spec["nrows"] == 1
            # 断言子图的 x 轴共享为 True
            assert s.subplot_spec["sharex"] is True
            # 断言子图的 y 轴共享为 True
            assert s.subplot_spec["sharey"] is True

        # 测试创建单个 facets 的情况
        def test_single_facet(self):

            # 指定 facets 的键
            key = "a"
            # 指定顺序
            order = list("abc")
            # 定义 facets 的详细结构，包括列变量和结构
            spec = {"variables": {"col": key}, "structure": {"col": order}}
            # 创建 Subplots 对象
            s = Subplots({}, spec, {})

            # 断言子图的数量与顺序长度相同
            assert s.n_subplots == len(order)
            # 断言子图的列数与顺序长度相同
            assert s.subplot_spec["ncols"] == len(order)
            # 断言子图的行数为 1
            assert s.subplot_spec["nrows"] == 1
            # 断言子图的 x 轴共享为 True
            assert s.subplot_spec["sharex"] is True
            # 断言子图的 y 轴共享为 True
            assert s.subplot_spec["sharey"] is True
    def test_two_facets(self):
        # 定义列键和行键
        col_key = "a"
        row_key = "b"
        # 定义列顺序和行顺序
        col_order = list("xy")
        row_order = list("xyz")
        # 定义图表规格
        spec = {
            "variables": {"col": col_key, "row": row_key},
            "structure": {"col": col_order, "row": row_order},
        }
        # 创建子图对象实例
        s = Subplots({}, spec, {})

        # 断言子图数量是否正确
        assert s.n_subplots == len(col_order) * len(row_order)
        # 断言子图列数是否正确
        assert s.subplot_spec["ncols"] == len(col_order)
        # 断言子图行数是否正确
        assert s.subplot_spec["nrows"] == len(row_order)
        # 断言是否共享x轴
        assert s.subplot_spec["sharex"] is True
        # 断言是否共享y轴
        assert s.subplot_spec["sharey"] is True

    def test_col_facet_wrapped(self):
        # 定义列键和包装数
        key = "b"
        wrap = 3
        # 定义列顺序
        order = list("abcde")
        # 定义图表规格
        spec = {"variables": {"col": key}, "structure": {"col": order}, "wrap": wrap}
        # 创建子图对象实例
        s = Subplots({}, spec, {})

        # 断言子图数量是否正确
        assert s.n_subplots == len(order)
        # 断言子图列数是否正确
        assert s.subplot_spec["ncols"] == wrap
        # 断言子图行数是否正确
        assert s.subplot_spec["nrows"] == len(order) // wrap + 1
        # 断言是否共享x轴
        assert s.subplot_spec["sharex"] is True
        # 断言是否共享y轴
        assert s.subplot_spec["sharey"] is True

    def test_row_facet_wrapped(self):
        # 定义行键和包装数
        key = "b"
        wrap = 3
        # 定义行顺序
        order = list("abcde")
        # 定义图表规格
        spec = {"variables": {"row": key}, "structure": {"row": order}, "wrap": wrap}
        # 创建子图对象实例
        s = Subplots({}, spec, {})

        # 断言子图数量是否正确
        assert s.n_subplots == len(order)
        # 断言子图列数是否正确
        assert s.subplot_spec["ncols"] == len(order) // wrap + 1
        # 断言子图行数是否正确
        assert s.subplot_spec["nrows"] == wrap
        # 断言是否共享x轴
        assert s.subplot_spec["sharex"] is True
        # 断言是否共享y轴
        assert s.subplot_spec["sharey"] is True

    def test_col_facet_wrapped_single_row(self):
        # 定义列键和顺序
        key = "b"
        order = list("abc")
        # 定义包装数
        wrap = len(order) + 2
        # 定义图表规格
        spec = {"variables": {"col": key}, "structure": {"col": order}, "wrap": wrap}
        # 创建子图对象实例
        s = Subplots({}, spec, {})

        # 断言子图数量是否正确
        assert s.n_subplots == len(order)
        # 断言子图列数是否正确
        assert s.subplot_spec["ncols"] == len(order)
        # 断言子图行数是否正确
        assert s.subplot_spec["nrows"] == 1
        # 断言是否共享x轴
        assert s.subplot_spec["sharex"] is True
        # 断言是否共享y轴
        assert s.subplot_spec["sharey"] is True

    def test_x_and_y_paired(self):
        # 定义x和y的顺序
        x = ["x", "y", "z"]
        y = ["a", "b"]
        # 定义图表规格
        s = Subplots({}, {}, {"structure": {"x": x, "y": y}})

        # 断言子图数量是否正确
        assert s.n_subplots == len(x) * len(y)
        # 断言子图列数是否正确
        assert s.subplot_spec["ncols"] == len(x)
        # 断言子图行数是否正确
        assert s.subplot_spec["nrows"] == len(y)
        # 断言x轴共享方式是否正确
        assert s.subplot_spec["sharex"] == "col"
        # 断言y轴共享方式是否正确
        assert s.subplot_spec["sharey"] == "row"

    def test_x_paired(self):
        # 定义x的顺序
        x = ["x", "y", "z"]
        # 定义图表规格
        s = Subplots({}, {}, {"structure": {"x": x}})

        # 断言子图数量是否正确
        assert s.n_subplots == len(x)
        # 断言子图列数是否正确
        assert s.subplot_spec["ncols"] == len(x)
        # 断言子图行数是否正确
        assert s.subplot_spec["nrows"] == 1
        # 断言x轴共享方式是否正确
        assert s.subplot_spec["sharex"] == "col"
        # 断言y轴共享方式是否正确
        assert s.subplot_spec["sharey"] is True
    # 定义测试方法，验证在只有 y 变量的情况下的子图布局
    def test_y_paired(self):

        # 定义 y 变量的列表
        y = ["x", "y", "z"]
        # 创建 Subplots 对象 s，传入空字典作为第一个参数，空字典作为第二个参数，包含 y 结构的字典作为第三个参数
        s = Subplots({}, {}, {"structure": {"y": y}})

        # 断言子图数量等于 y 变量列表的长度
        assert s.n_subplots == len(y)
        # 断言子图布局中的列数为 1
        assert s.subplot_spec["ncols"] == 1
        # 断言子图布局中的行数等于 y 变量列表的长度
        assert s.subplot_spec["nrows"] == len(y)
        # 断言子图布局中的 x 轴共享为 True
        assert s.subplot_spec["sharex"] is True
        # 断言子图布局中的 y 轴共享为 "row"
        assert s.subplot_spec["sharey"] == "row"

    # 定义测试方法，验证在 x 变量被配对和包裹的情况下的子图布局
    def test_x_paired_and_wrapped(self):

        # 定义 x 变量的列表
        x = ["a", "b", "x", "y", "z"]
        # 定义 wrap 参数为 3
        wrap = 3
        # 创建 Subplots 对象 s，传入空字典作为第一个参数，空字典作为第二个参数，包含 x 结构和 wrap 参数的字典作为第三个参数
        s = Subplots({}, {}, {"structure": {"x": x}, "wrap": wrap})

        # 断言子图数量等于 x 变量列表的长度
        assert s.n_subplots == len(x)
        # 断言子图布局中的列数为 wrap
        assert s.subplot_spec["ncols"] == wrap
        # 断言子图布局中的行数等于 x 变量列表长度整除 wrap 后加 1
        assert s.subplot_spec["nrows"] == len(x) // wrap + 1
        # 断言子图布局中的 x 轴共享为 False
        assert s.subplot_spec["sharex"] is False
        # 断言子图布局中的 y 轴共享为 True
        assert s.subplot_spec["sharey"] is True

    # 定义测试方法，验证在 y 变量被配对和包裹的情况下的子图布局
    def test_y_paired_and_wrapped(self):

        # 定义 y 变量的列表
        y = ["a", "b", "x", "y", "z"]
        # 定义 wrap 参数为 2
        wrap = 2
        # 创建 Subplots 对象 s，传入空字典作为第一个参数，空字典作为第二个参数，包含 y 结构和 wrap 参数的字典作为第三个参数
        s = Subplots({}, {}, {"structure": {"y": y}, "wrap": wrap})

        # 断言子图数量等于 y 变量列表的长度
        assert s.n_subplots == len(y)
        # 断言子图布局中的列数为 y 变量列表长度整除 wrap 后加 1
        assert s.subplot_spec["ncols"] == len(y) // wrap + 1
        # 断言子图布局中的行数为 wrap
        assert s.subplot_spec["nrows"] == wrap
        # 断言子图布局中的 x 轴共享为 True
        assert s.subplot_spec["sharex"] is True
        # 断言子图布局中的 y 轴共享为 False
        assert s.subplot_spec["sharey"] is False

    # 定义测试方法，验证在 y 变量被配对和包裹且只有单行的情况下的子图布局
    def test_y_paired_and_wrapped_single_row(self):

        # 定义 y 变量的列表
        y = ["x", "y", "z"]
        # 定义 wrap 参数为 1
        wrap = 1
        # 创建 Subplots 对象 s，传入空字典作为第一个参数，空字典作为第二个参数，包含 y 结构和 wrap 参数的字典作为第三个参数
        s = Subplots({}, {}, {"structure": {"y": y}, "wrap": wrap})

        # 断言子图数量等于 y 变量列表的长度
        assert s.n_subplots == len(y)
        # 断言子图布局中的列数等于 y 变量列表的长度
        assert s.subplot_spec["ncols"] == len(y)
        # 断言子图布局中的行数为 1
        assert s.subplot_spec["nrows"] == 1
        # 断言子图布局中的 x 轴共享为 True
        assert s.subplot_spec["sharex"] is True
        # 断言子图布局中的 y 轴共享为 False
        assert s.subplot_spec["sharey"] is False

    # 定义测试方法，验证在 y 变量被配对并且在列上划分的情况下的子图布局
    def test_col_faceted_y_paired(self):

        # 定义 y 变量的列表
        y = ["x", "y", "z"]
        # 定义 key 变量为 "a"
        key = "a"
        # 定义 order 变量为 "abc" 的列表
        order = list("abc")
        # 定义 facet_spec 和 pair_spec 字典
        facet_spec = {"variables": {"col": key}, "structure": {"col": order}}
        pair_spec = {"structure": {"y": y}}
        # 创建 Subplots 对象 s，传入空字典作为第一个参数，facet_spec 和 pair_spec 字典作为第二、第三个参数
        s = Subplots({}, facet_spec, pair_spec)

        # 断言子图数量等于 order 列表长度乘以 y 变量列表长度
        assert s.n_subplots == len(order) * len(y)
        # 断言子图布局中的列数为 order 列表的长度
        assert s.subplot_spec["ncols"] == len(order)
        # 断言子图布局中的行数为 y 变量列表的长度
        assert s.subplot_spec["nrows"] == len(y)
        # 断言子图布局中的 x 轴共享为 True
        assert s.subplot_spec["sharex"] is True
        # 断言子图布局中的 y 轴共享为 "row"
        assert s.subplot_spec["sharey"] == "row"

    # 定义测试方法，验证在 x 变量被配对并且在行上划分的情况下的子图布局
    def test_row_faceted_x_paired(self):

        # 定义 x 变量的列表
        x = ["f", "s"]
        # 定义 key 变量为 "a"
        key = "a"
        # 定义 order 变量为 "abc" 的列表
        order = list("abc")
        # 定义 facet_spec 和 pair_spec 字典
        facet_spec = {"variables": {"row": key}, "structure": {"row": order}}
        pair_spec = {"structure": {"x": x}}
        # 创建 Subplots 对象 s，传入空字典作为第一个参数，facet_spec 和 pair_spec 字典作为第二、第三个参数
        s = Subplots({}, facet_spec, pair_spec)

        # 断言子图数量等于 order 列表长度乘以 x 变量列表长度
        assert s.n_subplots == len(order) * len(x)
        # 断言子图布局中的列数为 x 变量列表的长度
        assert s.subplot_spec["ncols"] == len(x)
        # 断言子图布局中的行数为 order 列表的长度
        assert s.subplot_spec["nrows"] == len(order)
        # 断言子图布局中的 x 轴共享为 "col"
        assert s.subplot_spec["sharex"] == "col"
        # 断言子图布局中的 y 轴共享为 True
        assert s.subplot_spec["sharey"] is True
    # 定义测试函数，用于测试在非交叉情况下的 x 和 y 轴配对
    def test_x_any_y_paired_non_cross(self):

        # 定义 x 和 y 轴的标签列表
        x = ["a", "b", "c"]
        y = ["x", "y", "z"]
        # 定义图表的规格，包括结构和是否交叉
        spec = {"structure": {"x": x, "y": y}, "cross": False}
        # 创建 Subplots 对象 s，使用指定的结构规格
        s = Subplots({}, {}, spec)

        # 断言：子图数目应与 x 轴标签数量相等
        assert s.n_subplots == len(x)
        # 断言：子图规格中的列数应与 y 轴标签数量相等
        assert s.subplot_spec["ncols"] == len(y)
        # 断言：子图规格中的行数应为 1
        assert s.subplot_spec["nrows"] == 1
        # 断言：子图规格中的 sharex 属性应为 False
        assert s.subplot_spec["sharex"] is False
        # 断言：子图规格中的 sharey 属性应为 False
        assert s.subplot_spec["sharey"] is False

    # 定义测试函数，用于测试在非交叉情况下的 x 和 y 轴配对，并进行换行处理
    def test_x_any_y_paired_non_cross_wrapped(self):

        # 定义 x 和 y 轴的标签列表
        x = ["a", "b", "c"]
        y = ["x", "y", "z"]
        # 定义每行的子图数目
        wrap = 2
        # 定义图表的规格，包括结构、是否交叉以及换行数
        spec = {"structure": {"x": x, "y": y}, "cross": False, "wrap": wrap}
        # 创建 Subplots 对象 s，使用指定的结构规格
        s = Subplots({}, {}, spec)

        # 断言：子图数目应与 x 轴标签数量相等
        assert s.n_subplots == len(x)
        # 断言：子图规格中的列数应与指定的 wrap 数相等
        assert s.subplot_spec["ncols"] == wrap
        # 断言：子图规格中的行数应为 x 轴标签数量除以 wrap 向上取整后加 1
        assert s.subplot_spec["nrows"] == len(x) // wrap + 1
        # 断言：子图规格中的 sharex 属性应为 False
        assert s.subplot_spec["sharex"] is False
        # 断言：子图规格中的 sharey 属性应为 False
        assert s.subplot_spec["sharey"] is False

    # 定义测试函数，用于测试强制不共享子图的情况
    def test_forced_unshared_facets(self):

        # 创建 Subplots 对象 s，指定 sharex 属性为 False，sharey 属性为 "row"
        s = Subplots({"sharex": False, "sharey": "row"}, {}, {})
        
        # 断言：子图规格中的 sharex 属性应为 False
        assert s.subplot_spec["sharex"] is False
        # 断言：子图规格中的 sharey 属性应为 "row"
        assert s.subplot_spec["sharey"] == "row"
# 定义一个名为 TestSubplotElements 的测试类，用于测试子图元素的功能
class TestSubplotElements:

    # 定义测试单个子图的方法
    def test_single_subplot(self):

        # 创建一个 Subplots 对象 s，传入空字典作为参数
        s = Subplots({}, {}, {})
        # 初始化图形对象 f，调用 s 的 init_figure 方法，传入空字典作为参数
        f = s.init_figure({}, {})

        # 断言子图对象 s 的长度为 1
        assert len(s) == 1
        # 遍历子图对象 s 中的元素
        for i, e in enumerate(s):
            # 对每个子图元素 e 进行断言，检查其四个边是否存在
            for side in ["left", "right", "bottom", "top"]:
                assert e[side]
            # 断言子图元素 e 中的列和行维度是否为 None
            for dim in ["col", "row"]:
                assert e[dim] is None
            # 断言子图元素 e 中的 x 和 y 轴是否正确设置
            for axis in "xy":
                assert e[axis] == axis
            # 断言子图元素 e 中的 ax 属性与初始化的图形对象 f 的相应轴是否一致
            assert e["ax"] == f.axes[i]

    # 使用 pytest 参数化装饰器进行参数化测试，参数为 dim，取值为 ["col", "row"]
    @pytest.mark.parametrize("dim", ["col", "row"])
    # 定义测试单个分面维度的方法，参数为 dim
    def test_single_facet_dim(self, dim):

        # 定义 key 为 "a"，order 为 ['a', 'b', 'c']
        key = "a"
        order = list("abc")
        # 定义分面规格 spec，包含变量和结构信息
        spec = {"variables": {dim: key}, "structure": {dim: order}}
        # 创建一个 Subplots 对象 s，传入空字典和 spec 作为参数
        s = Subplots({}, spec, {})
        # 初始化图形对象，调用 s 的 init_figure 方法，传入 spec 和空字典作为参数
        s.init_figure(spec, {})

        # 断言子图对象 s 的长度与 order 列表的长度相等
        assert len(s) == len(order)

        # 遍历子图对象 s 中的元素
        for i, e in enumerate(s):
            # 断言子图元素 e 中的 dim 维度与 order[i] 相等
            assert e[dim] == order[i]
            # 断言子图元素 e 中的 x 和 y 轴是否正确设置
            for axis in "xy":
                assert e[axis] == axis
            # 根据 dim 的值设置顶部、底部、左侧和右侧边界的断言条件
            assert e["top"] == (dim == "col" or i == 0)
            assert e["bottom"] == (dim == "col" or i == len(order) - 1)
            assert e["left"] == (dim == "row" or i == 0)
            assert e["right"] == (dim == "row" or i == len(order) - 1)

    # 使用 pytest 参数化装饰器进行参数化测试，参数为 dim，取值为 ["col", "row"]
    @pytest.mark.parametrize("dim", ["col", "row"])
    # 定义测试单个分面维度包裹的方法，参数为 dim
    def test_single_facet_dim_wrapped(self, dim):

        # 定义 key 为 "b"，order 为 ['a', 'b', 'c']，wrap 为 2
        key = "b"
        order = list("abc")
        wrap = len(order) - 1
        # 定义分面规格 spec，包含变量、结构和包裹信息
        spec = {"variables": {dim: key}, "structure": {dim: order}, "wrap": wrap}
        # 创建一个 Subplots 对象 s，传入空字典和 spec 作为参数
        s = Subplots({}, spec, {})
        # 初始化图形对象，调用 s 的 init_figure 方法，传入 spec 和空字典作为参数
        s.init_figure(spec, {})

        # 断言子图对象 s 的长度与 order 列表的长度相等
        assert len(s) == len(order)

        # 遍历子图对象 s 中的元素
        for i, e in enumerate(s):
            # 断言子图元素 e 中的 dim 维度与 order[i] 相等
            assert e[dim] == order[i]
            # 断言子图元素 e 中的 x 和 y 轴是否正确设置
            for axis in "xy":
                assert e[axis] == axis

            # 定义 sides 字典，包含不同维度对应的边界顺序
            sides = {
                "col": ["top", "bottom", "left", "right"],
                "row": ["left", "right", "top", "bottom"],
            }

            # 定义 tests 元组，根据当前索引 i 和 wrap 的值设置边界断言条件
            tests = (
                i < wrap,
                i >= wrap or i >= len(s) % wrap,
                i % wrap == 0,
                i % wrap == wrap - 1 or i + 1 == len(s),
            )

            # 遍历 sides[dim] 中的边界，进行断言
            for side, expected in zip(sides[dim], tests):
                assert e[side] == expected
    # 定义测试函数，用于测试具有两个方面维度的子图
    def test_both_facet_dims(self):

        # 设定列和行的标识符
        col = "a"
        row = "b"
        # 设定列和行的顺序列表
        col_order = list("ab")
        row_order = list("xyz")
        # 定义面板规格，包括变量和结构信息
        facet_spec = {
            "variables": {"col": col, "row": row},
            "structure": {"col": col_order, "row": row_order},
        }
        # 创建子图对象实例并初始化图形
        s = Subplots({}, facet_spec, {})
        s.init_figure(facet_spec, {})

        # 计算列数和行数
        n_cols = len(col_order)
        n_rows = len(row_order)
        # 断言子图数量等于列数乘以行数
        assert len(s) == n_cols * n_rows
        # 转换子图列表为列表
        es = list(s)

        # 断言每个子图在第一行时具有'top'属性
        for e in es[:n_cols]:
            assert e["top"]
        # 断言每个子图在第一列时具有'left'属性
        for e in es[::n_cols]:
            assert e["left"]
        # 断言每个子图在最后一列时具有'right'属性
        for e in es[n_cols - 1::n_cols]:
            assert e["right"]
        # 断言每个子图在最后一行时具有'bottom'属性
        for e in es[-n_cols:]:
            assert e["bottom"]

        # 使用itertools.product迭代所有行列组合，断言每个子图的'col'和'row'属性正确
        for e, (row_, col_) in zip(es, itertools.product(row_order, col_order)):
            assert e["col"] == col_
            assert e["row"] == row_

        # 断言每个子图的'x'和'y'属性等于"x"和"y"
        for e in es:
            assert e["x"] == "x"
            assert e["y"] == "y"

    # 使用pytest的参数化装饰器，测试单个配对变量的情况
    @pytest.mark.parametrize("var", ["x", "y"])
    def test_single_paired_var(self, var):

        # 确定另一个变量
        other_var = {"x": "y", "y": "x"}[var]
        # 定义变量配对列表
        pairings = ["x", "y", "z"]
        # 定义配对规格，包括变量和结构信息
        pair_spec = {
            "variables": {f"{var}{i}": v for i, v in enumerate(pairings)},
            "structure": {var: [f"{var}{i}" for i, _ in enumerate(pairings)]},
        }

        # 创建子图对象实例并初始化图形
        s = Subplots({}, {}, pair_spec)
        s.init_figure(pair_spec)

        # 断言子图数量等于变量配对列表的长度
        assert len(s) == len(pair_spec["structure"][var])

        # 遍历子图列表，断言每个子图的变量属性与预期相符
        for i, e in enumerate(s):
            assert e[var] == f"{var}{i}"
            assert e[other_var] == other_var
            assert e["col"] is e["row"] is None

        # 定义测试条件和期望结果
        tests = i == 0, True, True, i == len(s) - 1
        # 定义每个变量对应的边缘属性列表
        sides = {
            "x": ["left", "right", "top", "bottom"],
            "y": ["top", "bottom", "left", "right"],
        }

        # 遍历变量对应的边缘属性，断言每个子图的边缘属性与期望结果相符
        for side, expected in zip(sides[var], tests):
            assert e[side] == expected

    # 使用pytest的参数化装饰器，测试变量为"x"和"y"的情况
    @pytest.mark.parametrize("var", ["x", "y"])
    # 测试单个配对变量包装
    def test_single_paired_var_wrapped(self, var):
        # 根据变量 var 获取另一个变量名
        other_var = {"x": "y", "y": "x"}[var]
        # 定义配对列表
        pairings = ["x", "y", "z", "a", "b"]
        # 计算包装点
        wrap = len(pairings) - 2
        # 构建配对规格字典
        pair_spec = {
            "variables": {f"{var}{i}": val for i, val in enumerate(pairings)},
            "structure": {var: [f"{var}{i}" for i, _ in enumerate(pairings)]},
            "wrap": wrap
        }
        # 创建 Subplots 实例
        s = Subplots({}, {}, pair_spec)
        # 初始化图形
        s.init_figure(pair_spec)

        # 断言 subplot 的数量等于配对列表的长度
        assert len(s) == len(pairings)

        # 遍历 subplot 列表
        for i, e in enumerate(s):
            # 断言当前 subplot 的 var 属性等于 var 后跟序号
            assert e[var] == f"{var}{i}"
            # 断言当前 subplot 的 other_var 属性等于 other_var
            assert e[other_var] == other_var
            # 断言当前 subplot 的 "col" 和 "row" 属性均为 None
            assert e["col"] is e["row"] is None

            # 定义测试条件列表
            tests = (
                i < wrap,
                i >= wrap or i >= len(s) % wrap,
                i % wrap == 0,
                i % wrap == wrap - 1 or i + 1 == len(s),
            )
            # 定义边界和对应的预期结果
            sides = {
                "x": ["top", "bottom", "left", "right"],
                "y": ["left", "right", "top", "bottom"],
            }
            # 遍历边界和预期结果进行断言
            for side, expected in zip(sides[var], tests):
                assert e[side] == expected

    # 测试两个配对变量
    def test_both_paired_variables(self):
        # 定义 x 和 y 的配对列表
        x = ["x0", "x1"]
        y = ["y0", "y1", "y2"]
        # 构建配对规格字典
        pair_spec = {"structure": {"x": x, "y": y}}
        # 创建 Subplots 实例
        s = Subplots({}, {}, pair_spec)
        # 初始化图形
        s.init_figure(pair_spec)

        # 计算列数和行数
        n_cols = len(x)
        n_rows = len(y)
        # 断言 subplot 的数量等于列数乘以行数
        assert len(s) == n_cols * n_rows
        # 将 subplot 列表转换为列表
        es = list(s)

        # 断言每一列的 subplot 的 "top" 属性为 True
        for e in es[:n_cols]:
            assert e["top"]
        # 断言每一行的 subplot 的 "left" 属性为 True
        for e in es[::n_cols]:
            assert e["left"]
        # 断言每一行的最后一个 subplot 的 "right" 属性为 True
        for e in es[n_cols - 1::n_cols]:
            assert e["right"]
        # 断言每一列的最后一个 subplot 的 "bottom" 属性为 True
        for e in es[-n_cols:]:
            assert e["bottom"]

        # 断言每一个 subplot 的 "col" 和 "row" 属性均为 None
        for e in es:
            assert e["col"] is e["row"] is None

        # 遍历 y 列表的长度，然后遍历 x 列表的长度，进行进一步的断言
        for i in range(len(y)):
            for j in range(len(x)):
                e = es[i * len(x) + j]
                assert e["x"] == f"x{j}"
                assert e["y"] == f"y{i}"

    # 测试两个配对变量且不交叉
    def test_both_paired_non_cross(self):
        # 构建不交叉的配对规格字典
        pair_spec = {
            "structure": {"x": ["x0", "x1", "x2"], "y": ["y0", "y1", "y2"]},
            "cross": False
        }
        # 创建 Subplots 实例
        s = Subplots({}, {}, pair_spec)
        # 初始化图形
        s.init_figure(pair_spec)

        # 遍历 subplot 列表
        for i, e in enumerate(s):
            # 断言当前 subplot 的 "x" 属性等于 x 后跟序号
            assert e["x"] == f"x{i}"
            # 断言当前 subplot 的 "y" 属性等于 y 后跟序号
            assert e["y"] == f"y{i}"
            # 断言当前 subplot 的 "col" 和 "row" 属性均为 None
            assert e["col"] is e["row"] is None
            # 断言当前 subplot 的 "left" 属性为 True 如果是第一个 subplot
            assert e["left"] == (i == 0)
            # 断言当前 subplot 的 "right" 属性为 True 如果是最后一个 subplot
            assert e["right"] == (i == (len(s) - 1))
            # 断言当前 subplot 的 "top" 和 "bottom" 属性为 True
            assert e["top"]
            assert e["bottom"]

    # 使用 pytest 的参数化功能，测试 "col" 和 "row" 变量
    @pytest.mark.parametrize("dim,var", [("col", "y"), ("row", "x")])
    # 定义一个测试方法，用于测试具有单个维度和配对变量的情况
    def test_one_facet_one_paired(self, dim, var):

        # 根据给定的变量 var，获取其对应的另一个变量
        other_var = {"x": "y", "y": "x"}[var]
        # 根据给定的维度 dim，获取其对应的另一个维度
        other_dim = {"col": "row", "row": "col"}[dim]
        # 定义一个顺序列表
        order = list("abc")
        # 定义一个包含维度和结构信息的字典，用于构建子图
        facet_spec = {"variables": {dim: "s"}, "structure": {dim: order}}

        # 定义一个配对列表
        pairings = ["x", "y", "t"]
        # 定义一个包含变量和结构信息的字典，用于构建子图的配对
        pair_spec = {
            "variables": {f"{var}{i}": val for i, val in enumerate(pairings)},
            "structure": {var: [f"{var}{i}" for i in range(len(pairings))]},
        }

        # 创建子图对象，初始化子图及其配对
        s = Subplots({}, facet_spec, pair_spec)
        s.init_figure(pair_spec)

        # 根据维度的不同，确定子图的列数和行数
        n_cols = len(order) if dim == "col" else len(pairings)
        n_rows = len(order) if dim == "row" else len(pairings)

        # 断言子图中的元素数量符合预期
        assert len(s) == len(order) * len(pairings)

        # 将子图转换为列表形式
        es = list(s)

        # 对于每个子图元素，检查其在子图中的位置特性
        for e in es[:n_cols]:
            assert e["top"]
        for e in es[::n_cols]:
            assert e["left"]
        for e in es[n_cols - 1::n_cols]:
            assert e["right"]
        for e in es[-n_cols:]:
            assert e["bottom"]

        # 如果维度是行，则重新整形子图列表
        if dim == "row":
            es = np.reshape(es, (n_rows, n_cols)).T.ravel()

        # 对每个子图元素进行进一步的断言检查
        for i, e in enumerate(es):
            assert e[dim] == order[i % len(pairings)]
            assert e[other_dim] is None
            assert e[var] == f"{var}{i // len(order)}"
            assert e[other_var] == other_var
```