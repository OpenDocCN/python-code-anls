# `d:/src/tocomm/basic-computer-games\73_Reverse\csharp\Reverse.Tests\ReverserTests.cs`

```
using FsCheck.Xunit;  # 导入 FsCheck.Xunit 模块
using Reverse.Tests.Generators;  # 导入 Reverse.Tests.Generators 模块
using System;  # 导入 System 模块
using System.Linq;  # 导入 System.Linq 模块
using Xunit;  # 导入 Xunit 模块

namespace Reverse.Tests  # 定义 Reverse.Tests 命名空间
{
    public class ReverserTests  # 定义 ReverserTests 类
    {
        [Fact]  # 标记下面的方法为测试方法
        public void Constructor_CannotAcceptNumberLessThanZero()  # 定义测试方法 Constructor_CannotAcceptNumberLessThanZero
        {
            Action action = () => new Reverser(0);  # 创建一个匿名方法，实例化 Reverser 类并传入参数 0

            Assert.Throws<ArgumentOutOfRangeException>(action);  # 断言匿名方法会抛出 ArgumentOutOfRangeException 异常
        }

        [Property(Arbitrary = new[] { typeof(PositiveIntegerGenerator) })]  # 使用属性标记指定测试方法使用 PositiveIntegerGenerator 生成器
        public void Constructor_CreatesRandomArrayOfSpecifiedLength(int size)  # 定义测试方法 Constructor_CreatesRandomArrayOfSpecifiedLength，接受一个参数 size
        {
            // 创建一个大小为 size 的 TestReverser 对象
            var sut = new TestReverser(size);

            // 断言 TestReverser 对象的数组长度是否等于 size
            Assert.Equal(size, sut.GetArray().Length);
        }

        [Property(Arbitrary = new[] { typeof(PositiveIntegerGenerator) })]
        public void ConstructorArray_MaxElementValueIsEqualToSize(int size)
        {
            // 创建一个大小为 size 的 TestReverser 对象
            var sut = new TestReverser(size);

            // 断言 TestReverser 对象的数组中最大值是否等于 size
            Assert.Equal(size, sut.GetArray().Max());
        }

        [Property(Arbitrary = new[] { typeof(PositiveIntegerGenerator) })]
        public void ConstructorArray_ReturnsRandomArrayWithDistinctElements(int size)
        {
            // 创建一个大小为 size 的 TestReverser 对象
            var sut = new TestReverser(size);
            // 获取 TestReverser 对象的数组
            var array = sut.GetArray();
            // 将数组中的元素按照值进行分组
            var arrayGroup = array.GroupBy(x => x);
            var duplicateFound = arrayGroup.Any(x => x.Count() > 1);
            // 检查是否在数组组中存在重复元素，返回布尔值
            Assert.False(duplicateFound);
            // 断言，如果duplicateFound为true，则测试失败

        }

        [Theory]
        [InlineData(new int[] { 1 }, new int[] { 1 })]
        [InlineData(new int[] { 1, 2 }, new int[] { 2, 1 })]
        [InlineData(new int[] { 1, 2, 3 }, new int[] { 3, 2, 1 })]
        public void Reverse_WillReverseEntireArray(int[] input, int[] output)
        {
            var sut = new TestReverser(1);
            // 创建TestReverser对象，参数为1
            sut.SetArray(input);
            // 设置TestReverser对象的数组为输入数组

            sut.Reverse(input.Length);
            // 调用Reverse方法，将数组中的元素进行反转

            Assert.True(sut.GetArray().SequenceEqual(output));
            // 断言，如果反转后的数组与期望的输出数组相等，则测试通过
        }

        [Fact]
        # 定义一个公共方法，用于测试指定索引位置之前的元素进行反转
        public void Reverse_WithSpecifiedIndex_ReversesItemsUpToThatIndex()
        {
            # 初始化输入数组
            var input = new int[] { 1, 2, 3, 4 };
            # 预期的输出数组
            var output = new int[] { 2, 1, 3, 4 };
            # 创建一个测试用的反转器对象，指定索引位置为1
            var sut = new TestReverser(1);
            # 设置反转器对象的数组为输入数组
            sut.SetArray(input);

            # 调用反转方法，指定索引位置为2
            sut.Reverse(2);

            # 断言实际输出数组与预期输出数组相等
            Assert.True(sut.GetArray().SequenceEqual(output));
        }

        # 定义一个公共方法，用于测试指定索引位置为1时的情况
        [Fact]
        public void Reverse_WithIndexOne_DoesNothing()
        {
            # 初始化输入数组
            var input = new int[] { 1, 2 };
            # 预期的输出数组
            var output = new int[] { 1, 2 };
            # 创建一个测试用的反转器对象，指定索引位置为1
            var sut = new TestReverser(1);
            # 设置反转器对象的数组为输入数组
            sut.SetArray(input);
            sut.Reverse(1);  # 调用 Reverse 方法，传入参数 1

            Assert.True(sut.GetArray().SequenceEqual(output));  # 使用断言验证 sut 对象的 GetArray 方法返回的数组是否与 output 数组相等
        }

        [Fact]
        public void Reverse_WithIndexGreaterThanArrayLength_DoesNothing()
        {
            var input = new int[] { 1, 2 };  # 创建包含元素 1 和 2 的整数数组 input
            var output = new int[] { 1, 2 };  # 创建包含元素 1 和 2 的整数数组 output
            var sut = new TestReverser(1);  # 创建 TestReverser 类的实例 sut，传入参数 1
            sut.SetArray(input);  # 调用 sut 对象的 SetArray 方法，传入参数 input

            sut.Reverse(sut.GetArray().Length + 1);  # 调用 Reverse 方法，传入参数为数组长度加 1

            Assert.True(sut.GetArray().SequenceEqual(output));  # 使用断言验证 sut 对象的 GetArray 方法返回的数组是否与 output 数组相等
        }

        [Fact]
        public void Reverse_WithIndexLessThanZero_DoesNothing()
        {
            // 创建输入数组
            var input = new int[] { 1, 2 };
            // 创建预期输出数组
            var output = new int[] { 1, 2 };
            // 创建测试对象
            var sut = new TestReverser(1);
            // 设置测试对象的输入数组
            sut.SetArray(input);

            // 反转输入数组
            sut.Reverse(-1);

            // 断言测试对象的输出数组与预期输出数组相等
            Assert.True(sut.GetArray().SequenceEqual(output));
        }

        [Theory]
        // 测试当输入数组为单个元素时的情况
        [InlineData(new int[] { 1 })]
        // 测试当输入数组为两个元素时的情况
        [InlineData(new int[] { 1, 2 })]
        // 测试当输入数组中元素相等时的情况
        [InlineData(new int[] { 1, 1 })]
        public void IsArrayInAscendingOrder_WhenArrayElementsAreInNumericAscendingOrder_ReturnsTrue(int[] input)
        {
            // 创建测试对象
            var sut = new TestReverser(1);
            // 设置测试对象的输入数组
            sut.SetArray(input);
            var result = sut.IsArrayInAscendingOrder();  # 调用 sut 对象的 IsArrayInAscendingOrder 方法，将结果赋值给 result

            Assert.True(result);  # 使用断言验证 result 是否为 True
        }

        [Fact]
        public void IsArrayInOrder_WhenArrayElementsAreNotInNumericAscendingOrder_ReturnsFalse()
        {
            var sut = new TestReverser(1);  # 创建 TestReverser 对象 sut，传入参数 1
            sut.SetArray(new int[] { 2, 1 });  # 调用 sut 对象的 SetArray 方法，传入整型数组 [2, 1]

            var result = sut.IsArrayInAscendingOrder();  # 调用 sut 对象的 IsArrayInAscendingOrder 方法，将结果赋值给 result

            Assert.False(result);  # 使用断言验证 result 是否为 False
        }

        [Fact]
        public void GetArrayString_ReturnsSpaceSeparatedElementsOfArrayInStringFormat()
        {
            var sut = new TestReverser(1);  # 创建 TestReverser 对象 sut，传入参数 1
# 设置一个包含整数 1 和 2 的数组
sut.SetArray(new int[] { 1, 2 });

# 调用 sut 对象的 GetArrayString 方法，将结果保存到 result 变量中
var result = sut.GetArrayString();

# 使用断言检查 result 是否等于 " 1  2 "，如果不等则抛出异常
Assert.Equal(" 1  2 ", result);
```