# `.\pytorch\test\functorch\test_rearrange.py`

```
class TestRearrange(TestCase):
    # 定义测试类 TestRearrange，继承自 TestCase

    def test_collapsed_ellipsis_errors_out(self) -> None:
        # 定义测试方法 test_collapsed_ellipsis_errors_out，无返回值
        x = torch.zeros([1, 1, 1, 1, 1])
        # 创建一个形状为 [1, 1, 1, 1, 1] 的全零张量 x

        # 测试 rearrange 函数对包含收缩省略号的模式的处理
        rearrange(x, "a b c d ... ->  a b c ... d")
        # 尝试使用指定模式对 x 进行重排列操作

        # 验证是否会抛出 ValueError 异常
        with self.assertRaises(ValueError):
            rearrange(x, "a b c d (...) ->  a b c ... d")
        # 尝试使用另一指定模式对 x 进行重排列操作，期望抛出 ValueError 异常

        # 测试使用简化的省略号模式
        rearrange(x, "... ->  (...)")
        # 尝试使用简化的省略号模式对 x 进行重排列操作

        # 再次验证是否会抛出 ValueError 异常
        with self.assertRaises(ValueError):
            rearrange(x, "(...) -> (...)")

    def test_ellipsis_ops(self) -> None:
        # 定义测试方法 test_ellipsis_ops，无返回值
        x = torch.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])
        # 创建一个形状为 [2, 3, 4, 5, 6] 的张量 x，内容为按顺序填充的数值

        # 对于每个标识模式进行测试
        for pattern in identity_patterns:
            # 使用 rearrange 函数按当前模式重排 x，并验证是否与 x 相等
            torch.testing.assert_close(rearrange(x, pattern), x, msg=pattern)

        # 对于每对等价重排模式进行测试
        for pattern1, pattern2 in equivalent_rearrange_patterns:
            # 分别使用两个模式重排 x，并验证它们是否得到相同结果
            torch.testing.assert_close(
                rearrange(x, pattern1),
                rearrange(x, pattern2),
                msg=f"{pattern1} vs {pattern2}",
            )
    # 定义一个测试函数，用于测试重新排列函数的一致性
    def test_rearrange_consistency(self) -> None:
        # 创建一个形状为 [1, 2, 3, 5, 7, 11] 的张量 x
        shape = [1, 2, 3, 5, 7, 11]
        x = torch.arange(int(np.prod(shape, dtype=int))).reshape(shape)
        
        # 对于每个指定的重排列模式，执行重新排列操作并进行断言检查
        for pattern in [
            "a b c d e f -> a b c d e f",  # 不进行任何重排列，应该返回原始张量
            "b a c d e f -> a b d e f c",  # 交换第一维和第二维，以此类推
            "a b c d e f -> f e d c b a",  # 逆序排列
            "a b c d e f -> (f e) d (c b a)",  # 一部分维度交换
            "a b c d e f -> (f e d c b a)",  # 全部维度交换
        ]:
            # 执行重新排列函数，并检查结果与原始张量的差异是否为空集
            result = rearrange(x, pattern)
            self.assertEqual(len(np.setdiff1d(x, result)), 0)
            # 检查结果的数据类型与原始张量是否一致
            self.assertIs(result.dtype, x.dtype)

        # 对特定的重排列模式进行进一步的测试与断言
        result = rearrange(x, "a b c d e f -> a (b) (c d e) f")
        torch.testing.assert_close(x.flatten(), result.flatten())

        result = rearrange(x, "a aa aa1 a1a1 aaaa a11 -> a aa aa1 a1a1 aaaa a11")
        torch.testing.assert_close(x, result)

        # 对逆序排列的两种方式进行测试，并检查结果是否相等
        result1 = rearrange(x, "a b c d e f -> f e d c b a")
        result2 = rearrange(x, "f e d c b a -> a b c d e f")
        torch.testing.assert_close(result1, result2)

        # 进行双重重新排列，并检查结果是否与原始张量相等
        result = rearrange(
            rearrange(x, "a b c d e f -> (f d) c (e b) a"),
            "(f d) c (e b) a -> a b c d e f",
            b=2,
            d=5,
        )
        torch.testing.assert_close(x, result)

        # 根据给定的大小参数重新排列张量，并检查结果是否与原始张量相等
        sizes = dict(zip("abcdef", shape))
        temp = rearrange(x, "a b c d e f -> (f d) c (e b) a", **sizes)
        result = rearrange(temp, "(f d) c (e b) a -> a b c d e f", **sizes)
        torch.testing.assert_close(x, result)

        # 创建另一个形状为 [2, 3, 4] 的张量 x2，并进行重新排列操作与断言检查
        x2 = torch.arange(2 * 3 * 4).reshape([2, 3, 4])
        result = rearrange(x2, "a b c -> b c a")
        # 检查特定位置的元素是否相等
        self.assertEqual(x2[1, 2, 3], result[2, 3, 1])
        self.assertEqual(x2[0, 1, 2], result[1, 2, 0])
    def test_rearrange_permutations(self) -> None:
        # 测试随机轴的排列与两种独立的 numpy 方法对比
        for n_axes in range(1, 10):
            # 创建一个形状为 [2] * n_axes 的张量
            input = torch.arange(2**n_axes).reshape([2] * n_axes)
            # 生成长度为 n_axes 的随机排列
            permutation = np.random.permutation(n_axes)
            # 构建左侧表达式，例如 "i0 i1 i2 ..."
            left_expression = " ".join("i" + str(axis) for axis in range(n_axes))
            # 根据排列构建右侧表达式，例如 "i2 i0 i1 ..."
            right_expression = " ".join("i" + str(axis) for axis in permutation)
            # 组合左右表达式形成完整的表达式，例如 "i0 i1 i2 ... -> i2 i0 i1 ..."
            expression = left_expression + " -> " + right_expression
            # 使用 rearrange 函数进行重排操作
            result = rearrange(input, expression)

            # 对于随机选取的 [10, n_axes] 数组中的每个选项
            for pick in np.random.randint(0, 2, [10, n_axes]):
                # 使用索引 pick 对 input 和 result 进行检查
                self.assertEqual(input[tuple(pick)], result[tuple(pick[permutation])])

        for n_axes in range(1, 10):
            # 创建一个形状为 [2] * n_axes 的张量
            input = torch.arange(2**n_axes).reshape([2] * n_axes)
            # 生成长度为 n_axes 的随机排列
            permutation = np.random.permutation(n_axes)
            # 构建左侧表达式，例如 "i2 i1 i0 ..."
            left_expression = " ".join("i" + str(axis) for axis in range(n_axes)[::-1])
            # 根据排列构建右侧表达式，例如 "i2 i1 i0 ..."
            right_expression = " ".join("i" + str(axis) for axis in permutation[::-1])
            # 组合左右表达式形成完整的表达式，例如 "i2 i1 i0 ... -> i2 i1 i0 ..."
            expression = left_expression + " -> " + right_expression
            # 使用 rearrange 函数进行重排操作
            result = rearrange(input, expression)
            # 检查结果张量的形状与输入张量是否相同
            self.assertEqual(result.shape, input.shape)
            # 创建一个与 input 形状相同的零张量 expected_result
            expected_result = torch.zeros_like(input)
            # 对于每个原始轴和结果轴的对应关系，更新 expected_result
            for original_axis, result_axis in enumerate(permutation):
                expected_result |= ((input >> original_axis) & 1) << result_axis

            # 使用 torch.testing.assert_close 检查 result 和 expected_result 是否接近
            torch.testing.assert_close(result, expected_result)

    def test_concatenations_and_stacking(self) -> None:
        # 对于不同数量 n_arrays 的数组进行测试
        for n_arrays in [1, 2, 5]:
            # 不同形状 shape 的测试列表
            shapes: List[List[int]] = [[], [1], [1, 1], [2, 3, 5, 7], [1] * 6]
            for shape in shapes:
                # 生成 n_arrays 个张量数组，每个数组的形状由 shape 决定
                arrays1 = [
                    torch.arange(i, i + np.prod(shape, dtype=int)).reshape(shape)
                    for i in range(n_arrays)
                ]
                # 使用 torch.stack 将数组堆叠为一个新的张量
                result0 = torch.stack(arrays1)
                # 使用 rearrange 函数进行重排操作
                result1 = rearrange(arrays1, "...->...")
                # 使用 torch.testing.assert_close 检查 result0 和 result1 是否接近
                torch.testing.assert_close(result0, result1)

    def test_unsqueeze(self) -> None:
        # 创建一个形状为 (2, 3, 4, 5) 的张量 x
        x = torch.randn((2, 3, 4, 5))
        # 使用 rearrange 函数进行重排操作，添加两个新的维度
        actual = rearrange(x, "b h w c -> b 1 h w 1 c")
        # 使用 unsqueeze 在指定维度上插入新维度
        expected = x.unsqueeze(1).unsqueeze(-2)
        # 使用 torch.testing.assert_close 检查 actual 和 expected 是否接近
        torch.testing.assert_close(actual, expected)

    def test_squeeze(self) -> None:
        # 创建一个形状为 (2, 1, 3, 4, 1, 5) 的张量 x
        x = torch.randn((2, 1, 3, 4, 1, 5))
        # 使用 rearrange 函数进行重排操作，去除所有维度为 1 的轴
        actual = rearrange(x, "b 1 h w 1 c -> b h w c")
        # 使用 squeeze 去除张量中所有大小为 1 的维度
        expected = x.squeeze()
        # 使用 torch.testing.assert_close 检查 actual 和 expected 是否接近
        torch.testing.assert_close(actual, expected)

    def test_0_dim_tensor(self) -> None:
        # 创建一个标量张量 x 和其期望值 expected
        x = expected = torch.tensor(1)
        # 使用 rearrange 函数将标量张量 x 重排为自身
        actual = rearrange(x, "->")
        # 使用 torch.testing.assert_close 检查 actual 和 expected 是否接近
        torch.testing.assert_close(actual, expected)

        # 使用 rearrange 函数将标量张量 x 重排为自身（与上述相同，使用 ... 表示所有维度）
        actual = rearrange(x, "... -> ...")
        # 使用 torch.testing.assert_close 检查 actual 和 expected 是否接近
        torch.testing.assert_close(actual, expected)
    # 定义一个测试方法，用于测试在没有省略号的情况下维度不匹配的情况
    def test_dimension_mismatch_no_ellipsis(self) -> None:
        # 创建一个形状为 (1, 2, 3) 的随机张量 x
        x = torch.randn((1, 2, 3))
        # 断言调用 rearrange 函数时会抛出 ValueError 异常，因为维度不匹配
        with self.assertRaises(ValueError):
            rearrange(x, "a b -> b a")
    
        # 再次使用不同的维度转换模式，期望同样抛出 ValueError 异常
        with self.assertRaises(ValueError):
            rearrange(x, "a b c d -> c d b a")
    
    # 定义另一个测试方法，用于测试在有省略号的情况下维度不匹配的情况
    def test_dimension_mismatch_with_ellipsis(self) -> None:
        # 创建一个标量张量 x
        x = torch.tensor(1)
        # 断言调用 rearrange 函数时会抛出 ValueError 异常，因为维度不匹配
        with self.assertRaises(ValueError):
            rearrange(x, "a ... -> ... a")
# 如果当前脚本被作为主程序执行（而非被导入为模块），则执行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```