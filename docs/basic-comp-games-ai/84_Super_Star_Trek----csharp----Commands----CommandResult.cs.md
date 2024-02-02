# `basic-computer-games\84_Super_Star_Trek\csharp\Commands\CommandResult.cs`

```py
# 定义命名空间 SuperStarTrek 中的命令类
namespace SuperStarTrek.Commands;

# 定义命令结果类
internal class CommandResult
{
    # 定义静态只读属性 Ok，表示命令执行正常
    public static readonly CommandResult Ok = new(false);
    # 定义静态只读属性 GameOver，表示游戏结束
    public static readonly CommandResult GameOver = new(true);

    # 私有构造函数，根据是否游戏结束来初始化命令结果
    private CommandResult(bool isGameOver)
    {
        IsGameOver = isGameOver;
    }

    # 私有构造函数，根据经过的时间来初始化命令结果
    private CommandResult(float timeElapsed)
    {
        TimeElapsed = timeElapsed;
    }

    # 公共只读属性，表示游戏是否结束
    public bool IsGameOver { get; }
    # 公共只读属性，表示经过的时间
    public float TimeElapsed { get; }

    # 静态方法，返回经过一定时间后的命令结果
    public static CommandResult Elapsed(float timeElapsed) => new(timeElapsed);
}
```