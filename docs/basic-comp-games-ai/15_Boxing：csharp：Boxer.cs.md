# `15_Boxing\csharp\Boxer.cs`

```
    // 设置拳击手的姓名
    public void SetName(string prompt)
    {
        // 输出提示信息
        Console.WriteLine(prompt);
        // 声明一个可空的字符串变量
        string? name;
        // 循环直到输入的姓名不为空或者不只包含空格
        do
        {
            // 从控制台读取输入的姓名
            name = Console.ReadLine();
        } while (string.IsNullOrWhiteSpace(name));
    }
        Name = name;  # 设置属性 Name 的值为参数 name

    public int DamageTaken { get; set; }  # 定义属性 DamageTaken，可读可写

    public void ResetForNewRound() => DamageTaken = 0;  # 重置 DamageTaken 属性为 0

    public void RecordWin() => _wins += 1;  # 记录胜利次数，_wins 属性加 1

    public bool IsWinner => _wins >= 2;  # 判断是否胜利，_wins 属性大于等于 2 返回 true

    public override string ToString() => Name;  # 重写 ToString 方法，返回 Name 属性的值

}

public class Opponent : Boxer  # 创建 Opponent 类，继承自 Boxer 类
{
    public void SetRandomPunches()  # 定义方法 SetRandomPunches
    {
        do  # 开始 do-while 循环
        {
            BestPunch = (Punch) GameUtils.Roll(4); // 用 GameUtils.Roll(4) 方法随机生成一个 Punch 对象，并赋值给 BestPunch
            Vulnerability = (Punch) GameUtils.Roll(4); // 用 GameUtils.Roll(4) 方法随机生成一个 Punch 对象，并赋值给 Vulnerability
        } while (BestPunch == Vulnerability); // 当 BestPunch 等于 Vulnerability 时，继续循环
    }
}
```