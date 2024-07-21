# `.\pytorch\benchmarks\instruction_counts\definitions\standard.py`

```py
"""
Default set of benchmarks.

Parser notes:
    `parse_stmts`:
        - Width for the left (Python) column MUST be 40 characters.
        - The column separator is " | ", not "|". Whitespace matters.

    `GroupedVariants`:
        - `Setup` and `Global_Setup` (case insensitive) are reserved keywords
          to populate `setup` and `global_setup` for every generated benchmark.
        - To set a label for the succeeding block, add `# @YOUR_LABEL` (Python)
          or `// @YOUR_LABEL` (C++).
"""

# 导入必要的模块和类
from core.api import GroupedModules, GroupedStmts, GroupedVariants
from core.types import FlatIntermediateDefinition
from core.utils import flatten, parse_stmts

# 导入设置相关的类
from definitions.setup import Setup

# 定义一个全局变量 BENCHMARKS，并指定其类型为 FlatIntermediateDefinition
BENCHMARKS: FlatIntermediateDefinition = flatten(
    """
    定义了一个字符串，包含了示例代码的一部分，展示了Python和C++的对比示例以及相应的注释格式
    ),
    "Indexing": GroupedVariants(
        *parse_stmts(
            r"""
    Python                                   | C++
    ---------------------------------------- | ----------------------------------------
    # @setup                                 | // @setup
                                             | 使用命名空间 torch::indexing;
    torch.manual_seed(6626_10_34)            | torch::manual_seed(66261034);
                                             |
    x = torch.randn(1, 1, 1)                 | auto x = torch::randn({1, 1, 1});
    y = torch.randn(1, 1, 1)                 | auto y = torch::randn({1, 1, 1});
                                             |
    # @Tensor-Scalar                         | // @Tensor-Scalar
    x[0] = 1                                 | x.index_put_({0}, 1);
    x[0, 0] = 1                              | x.index_put_({0, 0}, 1);
    x[0, 0, 0] = 1                           | x.index_put_({0, 0, 0}, 1);
                                             |
    # @Tensor-Scalar (Advanced)              | // @Tensor-Scalar (Advanced)
    x[...] = 1                               | x.index_put_({"..."}, 1);
    x[:] = 1                                 | x.index_put_({Slice(None, None, None)}, 1);
    x[None] = 1                              | x.index_put_({None}, 1);
    x[False] = 1                             | x.index_put_({false}, 1);
    x[True] = 1                              | x.index_put_({true}, 1);
                                             |
    # @Tensor-Tensor                         | // @Tensor-Tensor
    x[0] = y[0]                              | x.index_put_({0}, y.index({0}));
    x[0, 0] = y[0, 0]                        | x.index_put_({0, 0}, y.index({0, 0}));
    x[0, 0, 0] = y[0, 0, 0]                  | x.index_put_({0, 0, 0}, y.index({0, 0, 0}));
                                             |
    # @Tensor-Tensor (Advanced)              | // @Tensor-Tensor (Advanced)
    x[...] = y[...]                          | x.index_put_({"..."}, y.index({"..."}));
    x[:] = y[:]                              | x.index_put_({Slice(None, None, None)}, y.index({Slice(None, None, None)}));
    x[None] = y[None]                        | x.index_put_({None}, y.index({None}));
    x[False] = y[False]                      | x.index_put_({false}, y.index({false}));
    x[True] = y[True]                        | x.index_put_({true}, y.index({true}));
        """
            )
        ),
        "Metadata and views": GroupedVariants(
            *parse_stmts(
                r"""
        Python                                   | C++
        ---------------------------------------- | ----------------------------------------
        # @setup                                 | // @setup
        x = torch.ones((4, 4))                   | auto x = torch::ones({4, 4});
                                                 | 
        # @size                                  | // @size
        x.size()[0]                              | x.sizes()[0];
                                                 | 
        # @stride                                | // @stride
        x.stride(0)                              | x.stride(0);
                                                 | 
        # @as_strided                            | // @as_strided
        torch.as_strided(x, (2, 3), (4, 1), 2)   | torch::as_strided(x, {2, 3}, {4, 1}, 2);
                                                 | 
        # @select                                | // @select
        x.select(1, 1)                           | x.select(1, 1);
                                                 | 
        # @unsqueeze                             | // @unsqueeze
        x.unsqueeze(0)                           | x.unsqueeze(0);
                                                 | 
        # @view                                  | // @view
        x.view(-1, 1)                            | x.view({-1, 1});
                                                 | 
        # @transpose                             | // @transpose
        x.t()                                    | x.t();
                                                 | 
        # @reshape                               | // @reshape
        x.reshape((16, 1))                       | x.reshape({16, 1});
    }


注释：

        """
            )  # 结束多行字符串定义
        ),  # 结束 GroupedVariants 对象的参数列表
        "Metadata and views": GroupedVariants(  # 创建 "Metadata and views" 组的 GroupedVariants 对象
            *parse_stmts(  # 对通过 parse_stmts 解析得到的语句列表进行展开作为参数传入 GroupedVariants
                r"""  # 使用原始字符串定义多行文本
        Python                                   | C++  # 显示 Python 和 C++ 两列的标题
        ---------------------------------------- | ----------------------------------------
        # @setup                                 | // @setup  # 注释和对应的 C++ 代码注释
        x = torch.ones((4, 4))                   | auto x = torch::ones({4, 4});  # 创建一个4x4的张量 x
                                                 | 
        # @size                                  | // @size  # 注释和对应的 C++ 代码注释
        x.size()[0]                              | x.sizes()[0];  # 获取张量 x 的第一维大小
                                                 | 
        # @stride                                | // @stride  # 注释和对应的 C++ 代码注释
        x.stride(0)                              | x.stride(0);  # 获取张量 x 在第一维上的步幅
                                                 | 
        # @as_strided                            | // @as_strided  # 注释和对应的 C++ 代码注释
        torch.as_strided(x, (2, 3), (4, 1), 2)   | torch::as_strided(x, {2, 3}, {4, 1}, 2);  # 使用给定参数创建张量 x 的新视图
                                                 | 
        # @select                                | // @select  # 注释和对应的 C++ 代码注释
        x.select(1, 1)                           | x.select(1, 1);  # 选择张量 x 的第一维上索引为 1 的元素
                                                 | 
        # @unsqueeze                             | // @unsqueeze  # 注释和对应的 C++ 代码注释
        x.unsqueeze(0)                           | x.unsqueeze(0);  # 在张量 x 的第一维上增加一个维度
                                                 | 
        # @view                                  | // @view  # 注释和对应的 C++ 代码注释
        x.view(-1, 1)                            | x.view({-1, 1});  # 对张量 x 进行形状重塑
                                                 | 
        # @transpose                             | // @transpose  # 注释和对应的 C++ 代码注释
        x.t()                                    | x.t();  # 对张量 x 进行转置操作
                                                 | 
        # @reshape                               | // @reshape  # 注释和对应的 C++ 代码注释
        x.reshape((16, 1))                       | x.reshape({16, 1});  # 对张量 x 进行形状重塑为 (16, 1)
    }
)
```