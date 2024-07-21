# `.\pytorch\test\distributed\elastic\rendezvous\rendezvous_backend_test.py`

```
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Optional, Tuple

from torch.distributed.elastic.rendezvous import RendezvousStateError
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import (
    RendezvousBackend,
    Token,
)

class RendezvousBackendTestMixin(ABC):
    _backend: RendezvousBackend  # 定义一个私有属性 _backend，类型为 RendezvousBackend

    # Type hints，声明一些类型提示的函数签名，后续用于测试用例中的断言
    assertEqual: Callable
    assertNotEqual: Callable
    assertIsNone: Callable
    assertIsNotNone: Callable
    assertRaises: Callable

    @abstractmethod
    def _corrupt_state(self) -> None:
        """Corrupts the state stored in the backend."""
        pass

    def _set_state(
        self, state: bytes, token: Optional[Any] = None
    ) -> Tuple[bytes, Token, bool]:
        # 调用 _backend 对象的 set_state 方法，设置状态，并接收返回结果
        result = self._backend.set_state(state, token)

        # 断言结果不为 None
        self.assertIsNotNone(result)

        # 将结果强制转换为指定的元组类型
        return cast(Tuple[bytes, Token, bool], result)

    def test_get_state_returns_backend_state(self) -> None:
        # 设置后端状态为 b"x"
        self._backend.set_state(b"x")

        # 获取当前后端状态
        result = self._backend.get_state()

        # 断言结果不为 None
        self.assertIsNotNone(result)

        # 将结果解包为状态和令牌
        state, token = cast(Tuple[bytes, Token], result)

        # 断言状态为 b"x"，并且令牌不为 None
        self.assertEqual(b"x", state)
        self.assertIsNotNone(token)

    def test_get_state_returns_none_if_backend_state_does_not_exist(self) -> None:
        # 获取当前后端状态
        result = self._backend.get_state()

        # 断言结果为 None
        self.assertIsNone(result)

    def test_get_state_raises_error_if_backend_state_is_corrupt(self) -> None:
        # 模拟后端状态损坏的情况
        self._corrupt_state()

        # 断言调用 get_state 方法会引发 RendezvousStateError 异常
        with self.assertRaises(RendezvousStateError):
            self._backend.get_state()

    def test_set_state_sets_backend_state_if_it_does_not_exist(self) -> None:
        # 设置后端状态为 b"x"
        state, token, has_set = self._set_state(b"x")

        # 断言状态为 b"x"，令牌不为 None，且设置成功标志为 True
        self.assertEqual(b"x", state)
        self.assertIsNotNone(token)
        self.assertTrue(has_set)

    def test_set_state_sets_backend_state_if_token_is_current(self) -> None:
        # 设置第一个状态为 b"x"
        state1, token1, has_set1 = self._set_state(b"x")

        # 使用当前令牌设置状态为 b"y"
        state2, token2, has_set2 = self._set_state(b"y", token1)

        # 断言状态为 b"y"，令牌不同于之前的令牌，且设置成功标志为 True
        self.assertEqual(b"y", state2)
        self.assertNotEqual(token1, token2)
        self.assertTrue(has_set1)
        self.assertTrue(has_set2)

    def test_set_state_returns_current_backend_state_if_token_is_old(self) -> None:
        # 设置第一个状态为 b"x"
        state1, token1, _ = self._set_state(b"x")

        # 使用当前令牌设置状态为 b"y"
        state2, token2, _ = self._set_state(b"y", token1)

        # 使用旧令牌设置状态为 b"z"
        state3, token3, has_set = self._set_state(b"z", token1)

        # 断言状态为 b"y"，令牌与之前相同，且设置成功标志为 False
        self.assertEqual(state2, state3)
        self.assertEqual(token2, token3)
        self.assertFalse(has_set)
    # 定义一个测试方法，用于测试当 token 参数为 None 时，_set_state 方法的返回值是否正确
    def test_set_state_returns_current_backend_state_if_token_is_none(self) -> None:
        # 调用 _set_state 方法，传入 b"x" 作为参数，获取返回的 state1, token1, _
        state1, token1, _ = self._set_state(b"x")
    
        # 再次调用 _set_state 方法，传入 b"y" 作为参数，获取返回的 state2, token2, has_set
        state2, token2, has_set = self._set_state(b"y")
    
        # 断言 state1 和 state2 相等
        self.assertEqual(state1, state2)
        # 断言 token1 和 token2 相等
        self.assertEqual(token1, token2)
        # 断言 has_set 为 False
        self.assertFalse(has_set)
    
    # 定义一个测试方法，用于测试当 token 参数为 "invalid" 时，_set_state 方法的返回值是否正确
    def test_set_state_returns_current_backend_state_if_token_is_invalid(self) -> None:
        # 调用 _set_state 方法，传入 b"x" 作为参数，获取返回的 state1, token1, _
        state1, token1, _ = self._set_state(b"x")
    
        # 再次调用 _set_state 方法，传入 b"y" 和 token="invalid" 作为参数，获取返回的 state2, token2, has_set
        state2, token2, has_set = self._set_state(b"y", token="invalid")
    
        # 断言 state1 和 state2 相等
        self.assertEqual(state1, state2)
        # 断言 token1 和 token2 相等
        self.assertEqual(token1, token2)
        # 断言 has_set 为 False
        self.assertFalse(has_set)
```