# `basic-computer-games\00_Common\dotnet\Games.Common\IO\InsufficientInputException.cs`

```py
// 声明一个命名空间 Games.Common.IO
namespace Games.Common.IO;

// 声明一个自定义异常类 InsufficientInputException，继承自 Exception 类
public class InsufficientInputException : Exception
{
    // 声明一个无参构造函数，调用基类 Exception 的构造函数并传入错误信息字符串
    public InsufficientInputException()
        : base("Insufficient input was supplied")
    {
    }
}
```