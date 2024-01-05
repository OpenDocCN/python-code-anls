# `00_Common\dotnet\Games.Common\IO\InsufficientInputException.cs`

```
# 定义命名空间为 Games.Common.IO
namespace Games.Common.IO;

# 创建一个自定义异常类 InsufficientInputException，继承自 Exception 类
public class InsufficientInputException : Exception
{
    # 定义 InsufficientInputException 类的构造函数
    public InsufficientInputException()
        # 调用基类 Exception 的构造函数，传入错误信息 "Insufficient input was supplied"
        : base("Insufficient input was supplied")
    {
    }
}
```