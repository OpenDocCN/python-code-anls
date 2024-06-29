# `D:\src\scipysrc\pandas\pandas\tests\io\formats\test_ipython_compat.py`

```
# 导入 NumPy 库，简写为 np
import numpy as np

# 导入 pandas 库的配置模块中的 config 对象，简写为 cf
import pandas._config.config as cf

# 从 pandas 库中导入 DataFrame 和 MultiIndex 类
from pandas import (
    DataFrame,
    MultiIndex,
)


# 定义一个测试类 TestTableSchemaRepr
class TestTableSchemaRepr:
    # 定义测试方法 test_publishes，接受 ip 参数
    def test_publishes(self, ip):
        # 创建 ipython 实例，使用 ip.config 作为配置
        ipython = ip.instance(config=ip.config)
        
        # 创建一个 DataFrame 对象 df，包含一列名为 "A" 的数据 [1, 2]
        df = DataFrame({"A": [1, 2]})
        
        # 创建一个包含两个元素的列表 objects，分别为 df["A"] 和 df
        objects = [df["A"], df]  # dataframe / series
        
        # 预期的输出格式类型，用于断言比较
        expected_keys = [
            {"text/plain", "application/vnd.dataresource+json"},
            {"text/plain", "text/html", "application/vnd.dataresource+json"},
        ]

        # 设置上下文 opt，配置 display.html.table_schema 为 True
        opt = cf.option_context("display.html.table_schema", True)
        
        # 初始化 last_obj 为 None
        last_obj = None
        
        # 遍历 objects 列表中的元素 obj 和对应的 expected_keys
        for obj, expected in zip(objects, expected_keys):
            # 更新 last_obj 为当前 obj
            last_obj = obj
            
            # 使用 opt 上下文格式化 obj 的显示输出
            with opt:
                formatted = ipython.display_formatter.format(obj)
                
            # 断言格式化后的输出的键集合与 expected 相等
            assert set(formatted[0].keys()) == expected

        # 设置带有 latex 渲染的上下文 with_latex
        with_latex = cf.option_context("styler.render.repr", "latex")

        # 在 opt 和 with_latex 上下文中格式化 last_obj 的显示输出
        with opt, with_latex:
            formatted = ipython.display_formatter.format(last_obj)

        # 预期的输出格式类型，用于最后一次断言比较
        expected = {
            "text/plain",
            "text/html",
            "text/latex",
            "application/vnd.dataresource+json",
        }
        
        # 断言格式化后的输出的键集合与 expected 相等
        assert set(formatted[0].keys()) == expected

    # 定义测试方法 test_publishes_not_implemented，接受 ip 参数
    def test_publishes_not_implemented(self, ip):
        # 创建一个包含列 MultiIndex 的 DataFrame
        # 使用 MultiIndex.from_product 创建一个 MultiIndex 对象 midx
        midx = MultiIndex.from_product([["A", "B"], ["a", "b", "c"]])
        
        # 使用 np.random.default_rng 生成标准正态分布的随机数据填充 DataFrame df
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, len(midx))), columns=midx
        )

        # 设置上下文 opt，配置 display.html.table_schema 为 True
        opt = cf.option_context("display.html.table_schema", True)

        # 在 opt 上下文中格式化 df 的显示输出
        with opt:
            formatted = ip.instance(config=ip.config).display_formatter.format(df)

        # 预期的输出格式类型，用于断言比较
        expected = {"text/plain", "text/html"}
        
        # 断言格式化后的输出的键集合与 expected 相等
        assert set(formatted[0].keys()) == expected

    # 定义测试方法 test_config_on
    def test_config_on(self):
        # 创建一个 DataFrame 对象 df，包含一列名为 "A" 的数据 [1, 2]
        df = DataFrame({"A": [1, 2]})
        
        # 在 display.html.table_schema 为 True 的上下文中调用 df 的 _repr_data_resource_() 方法
        with cf.option_context("display.html.table_schema", True):
            result = df._repr_data_resource_()

        # 断言 result 不为 None
        assert result is not None

    # 定义测试方法 test_config_default_off
    def test_config_default_off(self):
        # 创建一个 DataFrame 对象 df，包含一列名为 "A" 的数据 [1, 2]
        df = DataFrame({"A": [1, 2]})
        
        # 在 display.html.table_schema 为 False 的上下文中调用 df 的 _repr_data_resource_() 方法
        with cf.option_context("display.html.table_schema", False):
            result = df._repr_data_resource_()

        # 断言 result 为 None
        assert result is None
    # 定义测试函数，用于测试数据资源格式化功能
    def test_enable_data_resource_formatter(self, ip):
        # GH#10491：GitHub issue编号
        # 获取IPython实例中的显示格式化器列表
        formatters = ip.instance(config=ip.config).display_formatter.formatters
        # 设置MIME类型为数据资源JSON格式
        mimetype = "application/vnd.dataresource+json"

        # 使用上下文管理器修改HTML表格模式显示设置为True
        with cf.option_context("display.html.table_schema", True):
            # 断言数据资源JSON格式已经在格式化器中
            assert "application/vnd.dataresource+json" in formatters
            # 断言数据资源JSON格式的显示已启用
            assert formatters[mimetype].enabled

        # 依然存在，但是已被禁用
        assert "application/vnd.dataresource+json" in formatters
        assert not formatters[mimetype].enabled

        # 能够重新设置
        with cf.option_context("display.html.table_schema", True):
            # 断言数据资源JSON格式依然在格式化器中
            assert "application/vnd.dataresource+json" in formatters
            # 断言数据资源JSON格式的显示已启用
            assert formatters[mimetype].enabled
            # 简单测试其工作是否正常
            ip.instance(config=ip.config).display_formatter.format(cf)
```