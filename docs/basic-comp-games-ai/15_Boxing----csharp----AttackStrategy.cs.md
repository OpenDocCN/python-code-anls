# `basic-computer-games\15_Boxing\csharp\AttackStrategy.cs`

```

// 命名空间声明
namespace Boxing;

// 抽象类 AttackStrategy
public abstract class AttackStrategy
{
    // 受保护的常量，表示击倒对手所需的伤害阈值
    protected const int KnockoutDamageThreshold = 35;
    // 受保护的只读字段，表示对手
    protected readonly Boxer Other;
    // 受保护的只读字段，表示工作栈
    protected readonly Stack<Action> Work;
    // 私有的只读字段，表示游戏结束通知
    private readonly Action _notifyGameEnded;

    // 构造函数，初始化对手、工作栈和游戏结束通知
    public AttackStrategy(Boxer other, Stack<Action> work, Action notifyGameEnded)
    {
        Other = other;
        Work = work;
        _notifyGameEnded = notifyGameEnded;
    }

    // 攻击方法
    public void Attack()
    {
        // 获取拳击动作
        var punch = GetPunch();
        // 如果是最佳拳击动作，对手受到额外伤害
        if (punch.IsBestPunch)
        {
            Other.DamageTaken += 2;
        }

        // 将拳击动作推入工作栈
        Work.Push(punch.Punch switch
        {
            Punch.FullSwing => FullSwing,
            Punch.Hook => Hook,
            Punch.Uppercut => Uppercut,
            _ => Jab
        });
    }

    // 抽象方法，获取拳击动作
    protected abstract AttackPunch GetPunch();
    // 抽象方法，全力摆动拳击动作
    protected abstract void FullSwing();
    // 抽象方法，钩拳拳击动作
    protected abstract void Hook();
    // 抽象方法，上勾拳拳击动作
    protected abstract void Uppercut();
    // 抽象方法，快速出拳拳击动作
    protected abstract void Jab();

    // 注册击倒对手
    protected void RegisterKnockout(string knockoutMessage)
    {
        // 清空工作栈
        Work.Clear();
        // 触发游戏结束通知
        _notifyGameEnded();
        // 打印击倒信息
        Console.WriteLine(knockoutMessage);
    }

    // 定义攻击拳击动作的记录类型
    protected record AttackPunch(Punch Punch, bool IsBestPunch);
}

```