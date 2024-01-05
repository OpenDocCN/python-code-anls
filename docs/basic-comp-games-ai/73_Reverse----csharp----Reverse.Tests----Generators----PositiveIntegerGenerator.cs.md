# `73_Reverse\csharp\Reverse.Tests\Generators\PositiveIntegerGenerator.cs`

```
# 使用FsCheck库
import FsCheck

# 在Reverse.Tests.Generators命名空间下创建PositiveIntegerGenerator类
class PositiveIntegerGenerator:
    # 生成一个大于0的任意整数
    def Generate():
        return Arb.Default.Int32().Filter(lambda x: x > 0)
```