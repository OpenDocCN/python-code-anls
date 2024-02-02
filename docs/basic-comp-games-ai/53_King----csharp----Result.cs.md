# `basic-computer-games\53_King\csharp\Result.cs`

```py
# 命名空间 King
namespace King;

# 定义一个内部的记录结构，包含 IsGameOver 和 Message 两个属性
internal record struct Result (bool IsGameOver, string Message)
{
    # 定义一个内部的静态方法，表示游戏结束，返回一个 Result 对象
    internal static Result GameOver(string message) => new(true, message);
    # 定义一个内部的静态属性，表示游戏继续，返回一个 Result 对象
    internal static Result Continue => new(false, "");
}
```