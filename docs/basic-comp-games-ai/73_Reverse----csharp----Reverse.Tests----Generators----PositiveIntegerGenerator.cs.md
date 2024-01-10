# `basic-computer-games\73_Reverse\csharp\Reverse.Tests\Generators\PositiveIntegerGenerator.cs`

```
# 导入 FsCheck 模块
using FsCheck;

# 定义一个静态类 PositiveIntegerGenerator
namespace Reverse.Tests.Generators
{
    public static class PositiveIntegerGenerator
    {
        # 定义一个静态方法 Generate，返回一个任意正整数的生成器
        public static Arbitrary<int> Generate() =>
            # 使用 Arb.Default.Int32() 生成一个任意整数的生成器，并通过 Filter 方法筛选出大于 0 的整数
            Arb.Default.Int32().Filter(x => x > 0);
    }
}
```