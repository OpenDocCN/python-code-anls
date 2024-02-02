# `basic-computer-games\00_Common\dotnet\Games.Common\IO\Strings.cs`

```py
# 声明一个命名空间为 Games.Common.IO
namespace Games.Common.IO;

# 声明一个内部的静态类 Strings
internal static class Strings
{
    # 声明一个内部的常量字符串 NumberExpected，表示输入的不是数字时的错误提示信息
    internal const string NumberExpected = "!Number expected - retry input line";
    # 声明一个内部的常量字符串 ExtraInput，表示输入多余内容时的提示信息
    internal const string ExtraInput = "!Extra input ignored";
}
```