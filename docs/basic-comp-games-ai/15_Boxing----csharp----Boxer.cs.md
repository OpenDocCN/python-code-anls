# `basic-computer-games\15_Boxing\csharp\Boxer.cs`

```

// 命名空间声明
namespace Boxing;

// Boxer 类
public class Boxer
{
    // 私有字段 _wins
    private int _wins;

    // 公共属性 Name，可读可写，初始值为空字符串
    private string Name { get; set; } = string.Empty;

    // 公共属性 BestPunch，可读可写
    public Punch BestPunch { get; set; }

    // 公共属性 Vulnerability，可读可写
    public Punch Vulnerability { get; set; }

    // 设置拳击手姓名的方法
    public void SetName(string prompt)
    {
        // 输出提示信息
        Console.WriteLine(prompt);
        // 声明可空字符串变量 name
        string? name;
        // 循环直到输入的姓名不为空
        do
        {
            name = Console.ReadLine();
        } while (string.IsNullOrWhiteSpace(name));
        // 将输入的姓名赋值给 Name 属性
        Name = name;
    }

    // 公共属性 DamageTaken，可读可写
    public int DamageTaken { get; set; }

    // 重置伤害值的方法
    public void ResetForNewRound() => DamageTaken = 0;

    // 记录胜利次数的方法
    public void RecordWin() => _wins += 1;

    // 判断是否获胜的属性
    public bool IsWinner => _wins >= 2;

    // 重写 ToString 方法，返回姓名
    public override string ToString() => Name;
}

// Opponent 类，继承自 Boxer 类
public class Opponent : Boxer
{
    // 设置随机拳的方法
    public void SetRandomPunches()
    {
        // 循环直到最佳拳和弱点拳不相同
        do
        {
            // 将随机数转换为 Punch 枚举类型，赋值给 BestPunch 属性
            BestPunch = (Punch) GameUtils.Roll(4); // B1
            // 将随机数转换为 Punch 枚举类型，赋值给 Vulnerability 属性
            Vulnerability = (Punch) GameUtils.Roll(4); // D1
        } while (BestPunch == Vulnerability);
    }
}

```