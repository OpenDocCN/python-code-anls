# `.\pytorch\test\distributed\elastic\utils\data\cycling_iterator_test.py`

```
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入 Python 标准库中的单元测试模块
import unittest

# 从 torch.distributed.elastic.utils.data 中导入 CyclingIterator 类
from torch.distributed.elastic.utils.data import CyclingIterator


class CyclingIteratorTest(unittest.TestCase):
    def generator(self, epoch, stride, max_epochs):
        # 生成一个连续增长的列表，每个 epoch 生成一个
        # 例如：[0,1,2] [3,4,5] [6,7,8] ...
        return iter([stride * epoch + i for i in range(0, stride)])

    def test_cycling_iterator(self):
        stride = 3
        max_epochs = 90

        def generator_fn(epoch):
            return self.generator(epoch, stride, max_epochs)

        # 创建 CyclingIterator 实例，用于测试迭代器的行为
        it = CyclingIterator(n=max_epochs, generator_fn=generator_fn)
        # 遍历迭代器，验证生成的值是否符合预期
        for i in range(0, stride * max_epochs):
            self.assertEqual(i, next(it))

        # 检查迭代器耗尽后是否会触发 StopIteration 异常
        with self.assertRaises(StopIteration):
            next(it)

    def test_cycling_iterator_start_epoch(self):
        stride = 3
        max_epochs = 2
        start_epoch = 1

        def generator_fn(epoch):
            return self.generator(epoch, stride, max_epochs)

        # 创建带有指定起始 epoch 的 CyclingIterator 实例
        it = CyclingIterator(max_epochs, generator_fn, start_epoch)
        # 遍历迭代器，验证从指定起始 epoch 开始生成的值是否符合预期
        for i in range(stride * start_epoch, stride * max_epochs):
            self.assertEqual(i, next(it))

        # 检查迭代器耗尽后是否会触发 StopIteration 异常
        with self.assertRaises(StopIteration):
            next(it)
```