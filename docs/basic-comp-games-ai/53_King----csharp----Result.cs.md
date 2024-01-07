# `basic-computer-games\53_King\csharp\Result.cs`

```

// 命名空间为 King
namespace King;

// 定义一个内部的记录结构，包含两个字段：IsGameOver（游戏是否结束）和Message（消息）
internal record struct Result (bool IsGameOver, string Message)
{
    // 定义一个静态方法，表示游戏结束，返回一个Result对象，IsGameOver为true，Message为传入的消息
    internal static Result GameOver(string message) => new(true, message);
    // 定义一个静态属性，表示游戏继续，返回一个Result对象，IsGameOver为false，Message为空字符串
    internal static Result Continue => new(false, "");
}

```