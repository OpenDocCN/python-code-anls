# `basic-computer-games\84_Super_Star_Trek\csharp\Commands\CommandResult.cs`

```

// 命名空间 SuperStarTrek.Commands 下的内部类 CommandResult
internal class CommandResult
{
    // 表示命令执行成功的静态只读属性
    public static readonly CommandResult Ok = new(false);
    // 表示游戏结束的静态只读属性
    public static readonly CommandResult GameOver = new(true);

    // 私有构造函数，用于创建游戏结束的 CommandResult 对象
    private CommandResult(bool isGameOver)
    {
        IsGameOver = isGameOver;
    }

    // 私有构造函数，用于创建经过时间的 CommandResult 对象
    private CommandResult(float timeElapsed)
    {
        TimeElapsed = timeElapsed;
    }

    // 表示命令执行是否导致游戏结束的属性
    public bool IsGameOver { get; }
    // 表示经过的时间的属性
    public float TimeElapsed { get; }

    // 静态方法，用于创建经过时间的 CommandResult 对象
    public static CommandResult Elapsed(float timeElapsed) => new(timeElapsed);
}

```