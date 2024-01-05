# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Commands\CommandResult.cs`

```
namespace SuperStarTrek.Commands;  # 命名空间声明

internal class CommandResult  # 定义一个内部类 CommandResult
{
    public static readonly CommandResult Ok = new(false);  # 定义一个静态只读的 CommandResult 对象 Ok，表示游戏继续进行
    public static readonly CommandResult GameOver = new(true);  # 定义一个静态只读的 CommandResult 对象 GameOver，表示游戏结束

    private CommandResult(bool isGameOver)  # 定义一个私有构造函数，根据传入的参数判断游戏是否结束
    {
        IsGameOver = isGameOver;  # 设置 IsGameOver 属性为传入的参数值
    }

    private CommandResult(float timeElapsed)  # 定义一个私有构造函数，根据传入的参数设置游戏经过的时间
    {
        TimeElapsed = timeElapsed;  # 设置 TimeElapsed 属性为传入的时间值
    }

    public bool IsGameOver { get; }  # 定义一个公共只读属性 IsGameOver，用于获取游戏是否结束的状态
    public float TimeElapsed { get; }  # 定义一个公共只读属性 TimeElapsed，用于获取游戏经过的时间
}
# 创建一个名为Elapsed的静态方法，接受一个名为timeElapsed的参数，返回一个CommandResult对象
def Elapsed(timeElapsed):
    return CommandResult(timeElapsed)
```