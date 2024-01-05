# `12_Bombs_Away\csharp\BombsAwayGame\Side.cs`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
    /// <summary>
    /// User interface supplied to ctor.
    /// </summary>
    protected IUserInterface UI { get; }
    // 用户界面提供给构造函数的接口

    /// <summary>
    /// Random-number generator for this play-through.
    /// </summary>
    private readonly Random _random = new();
    // 为本次游戏生成随机数的随机数生成器

    /// <summary>
    /// Gets a random floating-point number greater than or equal to zero, and less than one.
    /// </summary>
    /// <returns>Random floating-point number greater than or equal to zero, and less than one.</returns>
    protected double RandomFrac() => _random.NextDouble();
    // 获取大于或等于零且小于一的随机浮点数

    /// <summary>
    /// Gets a random integer in a range.
    /// </summary>
    // 获取指定范围内的随机整数
    /// <param name="minValue">The inclusive lower bound of the number returned.</param>
    /// <param name="maxValue">The exclusive upper bound of the number returned.</param>
    /// <returns>Random integer in a range.</returns>
    // 生成指定范围内的随机整数
    protected int RandomInteger(int minValue, int maxValue) => _random.Next(minValue: minValue, maxValue: maxValue);

    /// <summary>
    /// Display messages indicating the mission succeeded.
    /// </summary>
    // 显示任务成功的消息
    protected void MissionSucceeded()
    {
        UI.Output("DIRECT HIT!!!! " + RandomInteger(0, 100) + " KILLED.");
        UI.Output("MISSION SUCCESSFUL.");
    }

    /// <summary>
    /// Gets the Guns type of enemy artillery.
    /// </summary>
    // 获取敌方火炮的类型为GUNS
    protected EnemyArtillery Guns { get; } = new("GUNS", 0);

    /// <summary>
    /// <summary>
    /// 获取敌方火炮的导弹类型。
    /// </summary>
    protected EnemyArtillery Missiles { get; } = new("MISSILES", 35);

    /// <summary>
    /// 获取敌方火炮的双枪和导弹类型。
    /// </summary>
    protected EnemyArtillery Both { get; } = new("BOTH", 35);

    /// <summary>
    /// 使用给定的火炮和命中率百分比执行敌方反击。
    /// </summary>
    /// <param name="artillery">要使用的敌方火炮。</param>
    /// <param name="hitRatePercent">敌方的命中率百分比。</param>
    protected void EnemyCounterattack(EnemyArtillery artillery, int hitRatePercent)
    {
        if (hitRatePercent + artillery.Accuracy > RandomInteger(0, 100))
        {
            MissionFailed();
        }
    }
        else
        {
            // 如果条件不满足，显示消息表明任务失败
            UI.Output("YOU MADE IT THROUGH TREMENDOUS FLAK!!");
        }
    }

    /// <summary>
    /// 显示消息表明任务失败
    /// </summary>
    protected void MissionFailed()
    {
        // 显示任务失败的消息
        UI.Output("* * * * BOOM * * * *");
        UI.Output("YOU HAVE BEEN SHOT DOWN.....");
        UI.Output("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR");
        UI.Output("LAST TRIBUTE...");
    }
}
```