# `.\pytorch\torch\distributed\argparse_util.py`

```py
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from argparse import Action


class env(Action):
    """
    Get argument values from ``PET_{dest}`` before defaulting to the given ``default`` value.

    For flags (e.g. ``--standalone``)
    use ``check_env`` instead.

    .. note:: when multiple option strings are specified, ``dest`` is
              the longest option string (e.g. for ``"-f", "--foo"``
              the env var to set is ``PET_FOO`` not ``PET_F``)

    Example:
    ::

     parser.add_argument("-f", "--foo", action=env, default="bar")

     ./program                                      -> args.foo="bar"
     ./program -f baz                               -> args.foo="baz"
     ./program --foo baz                            -> args.foo="baz"
     PET_FOO="env_bar" ./program -f baz    -> args.foo="baz"
     PET_FOO="env_bar" ./program --foo baz -> args.foo="baz"
     PET_FOO="env_bar" ./program           -> args.foo="env_bar"

     parser.add_argument("-f", "--foo", action=env, required=True)

     ./program                                      -> fails
     ./program -f baz                               -> args.foo="baz"
     PET_FOO="env_bar" ./program           -> args.foo="env_bar"
     PET_FOO="env_bar" ./program -f baz    -> args.foo="baz"
    """

    def __init__(self, dest, default=None, required=False, **kwargs) -> None:
        # 构造函数初始化
        env_name = f"PET_{dest.upper()}"
        # 从环境变量中获取默认值，如果没有则使用给定的默认值
        default = os.environ.get(env_name, default)

        # 如果找到默认值，则不需要在命令行参数中设置该选项
        # 因此将 required 设置为 False
        if default:
            required = False

        super().__init__(dest=dest, default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # 在调用时设置命名空间中的属性值
        setattr(namespace, self.dest, values)


class check_env(Action):
    """
    Check whether the env var ``PET_{dest}`` exists before defaulting to the given ``default`` value.

    Equivalent to
    ``store_true`` argparse built-in action except that the argument can
    be omitted from the commandline if the env var is present and has a
    non-zero value.

    .. note:: it is redundant to pass ``default=True`` for arguments
              that use this action because a flag should be ``True``
              when present and ``False`` otherwise.

    Example:
    ::

     parser.add_argument("--standalone", action=check_env)

     ./program                                  -> args.standalone=False
     PET_STANDALONE=1 ./program  -> args.standalone=True
    """
    # 定义一个自定义的参数类，继承自 argparse.Action
    def __init__(self, dest, default=False, **kwargs) -> None:
        # 根据参数的目的地生成对应的环境变量名
        env_name = f"PET_{dest.upper()}"
        # 从环境变量中获取值，默认为 1 或 0，转换成布尔类型作为默认值
        default = bool(int(os.environ.get(env_name, "1" if default else "0")))
        # 调用父类 argparse.Action 的构造函数进行初始化
        super().__init__(dest=dest, const=True, default=default, nargs=0, **kwargs)

    # 当解析器解析到该参数时调用的方法
    def __call__(self, parser, namespace, values, option_string=None):
        # 将参数值设置为 const 的值，即 True
        setattr(namespace, self.dest, self.const)
```