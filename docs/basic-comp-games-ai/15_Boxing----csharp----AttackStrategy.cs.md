# `basic-computer-games\15_Boxing\csharp\AttackStrategy.cs`

```py
// 命名空间声明
namespace Boxing;

// 抽象类，定义了攻击策略的基本结构
public abstract class AttackStrategy
{
    // 受保护的常量，表示击倒对手所需的伤害阈值
    protected const int KnockoutDamageThreshold = 35;
    // 受保护的只读字段，表示对手的信息
    protected readonly Boxer Other;
    // 受保护的只读字段，表示工作栈
    protected readonly Stack<Action> Work;
    // 私有的只读字段，表示游戏结束时的通知动作
    private readonly Action _notifyGameEnded;

    // 构造函数，初始化攻击策略的实例
    public AttackStrategy(Boxer other, Stack<Action> work, Action notifyGameEnded)
    {
        Other = other;
        Work = work;
        _notifyGameEnded = notifyGameEnded;
    }

    // 攻击方法
    public void Attack()
    {
        // 获取攻击拳法
        var punch = GetPunch();
        // 如果是最佳拳法，则对对手造成额外伤害
        if (punch.IsBestPunch)
        {
            Other.DamageTaken += 2;
        }

        // 将攻击拳法对应的动作推入工作栈
        Work.Push(punch.Punch switch
        {
            Punch.FullSwing => FullSwing,
            Punch.Hook => Hook,
            Punch.Uppercut => Uppercut,
            _ => Jab
        });
    }

    // 抽象方法，获取攻击拳法
    protected abstract AttackPunch GetPunch();
    // 抽象方法，执行全力摆拳动作
    protected abstract void FullSwing();
    // 抽象方法，执行钩拳动作
    protected abstract void Hook();
    // 抽象方法，执行上勾拳动作
    protected abstract void Uppercut();
    // 抽象方法，执行快拳动作
    protected abstract void Jab();

    // 受保护的方法，注册击倒对手的动作
    protected void RegisterKnockout(string knockoutMessage)
    {
        // 清空工作栈
        Work.Clear();
        // 发送游戏结束的通知
        _notifyGameEnded();
        // 打印击倒信息
        Console.WriteLine(knockoutMessage);
    }

    // 定义攻击拳法的记录类型
    protected record AttackPunch(Punch Punch, bool IsBestPunch);
}
```