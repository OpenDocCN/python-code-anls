# `basic-computer-games\00_Common\dotnet\Games.Common\IO\InsufficientInputException.cs`

```

// 声明命名空间为 Games.Common.IO
namespace Games.Common.IO;

// 声明一个自定义异常类 InsufficientInputException，继承自 Exception 类
public class InsufficientInputException : Exception
{
    // 声明一个无参构造函数，调用父类 Exception 的构造函数并传入错误信息 "Insufficient input was supplied"
    public InsufficientInputException()
        : base("Insufficient input was supplied")
    {
    }
}

```