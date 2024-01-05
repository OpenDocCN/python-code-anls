# `d:/src/tocomm/basic-computer-games\53_King\csharp\Result.cs`

```
namespace King;  # 命名空间声明，定义了代码所在的命名空间

internal record struct Result (bool IsGameOver, string Message)  # 定义了一个内部的记录结构体Result，包含IsGameOver和Message两个属性

{
    internal static Result GameOver(string message) => new(true, message);  # 定义了一个静态方法GameOver，返回一个Result对象，IsGameOver为true，Message为传入的message参数
    internal static Result Continue => new(false, "");  # 定义了一个静态属性Continue，返回一个Result对象，IsGameOver为false，Message为空字符串
}
```