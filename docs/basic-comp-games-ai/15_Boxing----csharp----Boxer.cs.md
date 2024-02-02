# `basic-computer-games\15_Boxing\csharp\Boxer.cs`

```py
# 命名空间声明
namespace Boxing;

# 定义 Boxer 类
public class Boxer
{
    # 私有属性 _wins，用于记录胜利次数
    private int _wins;

    # 公共属性 Name，用于设置和获取拳击手的姓名
    private string Name { get; set; } = string.Empty;

    # 公共属性 BestPunch，用于设置和获取拳击手的最佳拳
    public Punch BestPunch { get; set; }

    # 公共属性 Vulnerability，用于设置和获取拳击手的弱点
    public Punch Vulnerability { get; set; }

    # 设置拳击手的姓名
    public void SetName(string prompt)
    {
        # 输出提示信息
        Console.WriteLine(prompt);
        # 声明一个可空的字符串变量 name
        string? name;
        # 循环直到输入的姓名不为空或者不是空白字符
        do
        {
            name = Console.ReadLine();
        } while (string.IsNullOrWhiteSpace(name));
        # 将输入的姓名赋值给 Name 属性
        Name = name;
    }

    # 公共属性 DamageTaken，用于设置和获取拳击手受到的伤害
    public int DamageTaken { get; set; }

    # 重置拳击手的受伤程度为 0
    public void ResetForNewRound() => DamageTaken = 0;

    # 记录拳击手的胜利次数
    public void RecordWin() => _wins += 1;

    # 判断拳击手是否获胜（胜利次数大于等于 2）
    public bool IsWinner => _wins >= 2;

    # 重写 ToString 方法，返回拳击手的姓名
    public override string ToString() => Name;
}

# 定义 Opponent 类，继承自 Boxer 类
public class Opponent : Boxer
{
    # 设置随机的最佳拳和弱点
    public void SetRandomPunches()
    {
        # 循环直到最佳拳不等于弱点
        do
        {
            # 设置最佳拳为随机值
            BestPunch = (Punch) GameUtils.Roll(4); // B1
            # 设置弱点为随机值
            Vulnerability = (Punch) GameUtils.Roll(4); // D1
        } while (BestPunch == Vulnerability);
    }
}
```