# `15_Boxing\csharp\AttackStrategy.cs`

```
    // 定义一个名为 Boxing 的命名空间
    namespace Boxing;

    // 定义一个名为 AttackStrategy 的抽象类
    public abstract class AttackStrategy
    {
        // 定义一个名为 KnockoutDamageThreshold 的常量，表示击倒对手所需的伤害阈值
        protected const int KnockoutDamageThreshold = 35;
        // 定义一个名为 Other 的只读字段，表示另一个拳击手
        protected readonly Boxer Other;
        // 定义一个名为 Work 的只读字段，表示拳击手的工作栈
        protected readonly Stack<Action> Work;
        // 定义一个名为 _notifyGameEnded 的只读字段，表示游戏结束时的通知动作
        private readonly Action _notifyGameEnded;

        // 定义一个构造函数，接受另一个拳击手、工作栈和游戏结束通知动作作为参数
        public AttackStrategy(Boxer other, Stack<Action> work, Action notifyGameEnded)
        {
            Other = other;
            Work = work;
            _notifyGameEnded = notifyGameEnded;
        }

        // 定义一个名为 Attack 的方法
        public void Attack()
        {
            // 获取拳击动作
            var punch = GetPunch();
            // 如果拳击动作是最佳拳击动作
        {
            # 增加其他伤害值
            Other.DamageTaken += 2;
        }

        # 将拳击动作推入工作队列
        Work.Push(punch.Punch switch
        {
            Punch.FullSwing => FullSwing,  # 如果是全力摆动，则执行 FullSwing 方法
            Punch.Hook => Hook,  # 如果是钩拳，则执行 Hook 方法
            Punch.Uppercut => Uppercut,  # 如果是上勾拳，则执行 Uppercut 方法
            _ => Jab  # 如果是其他情况，则执行 Jab 方法
        });
    }

    # 获取拳击动作
    protected abstract AttackPunch GetPunch();
    # 执行全力摆动动作
    protected abstract void FullSwing();
    # 执行钩拳动作
    protected abstract void Hook();
    # 执行上勾拳动作
    protected abstract void Uppercut();
    # 执行普通直拳动作
    protected abstract void Jab();

    # 注册击倒信息
    protected void RegisterKnockout(string knockoutMessage)
    {
        Work.Clear();  # 清空工作内容
        _notifyGameEnded();  # 通知游戏结束
        Console.WriteLine(knockoutMessage);  # 在控制台打印击倒信息
    }

    protected record AttackPunch(Punch Punch, bool IsBestPunch);  # 定义攻击拳击动作的记录类型，包括拳击动作和是否是最佳拳击动作的标志
```