# `basic-computer-games\15_Boxing\csharp\PlayerAttackStrategy.cs`

```
// 使用静态导入 GameUtils 类中的方法和 Console 类中的 WriteLine 方法
using static Boxing.GameUtils;
using static System.Console;
// 命名空间 Boxing
namespace Boxing;

// 定义 PlayerAttackStrategy 类，继承自 AttackStrategy 类
public class PlayerAttackStrategy : AttackStrategy
{
    // 私有字段，存储玩家角色
    private readonly Boxer _player;

    // 构造函数，接受玩家角色、对手、游戏结束通知和工作栈作为参数
    public PlayerAttackStrategy(Boxer player, Opponent opponent, Action notifyGameEnded, Stack<Action> work)
        : base(opponent, work, notifyGameEnded) => _player = player;

    // 重写父类的 GetPunch 方法
    protected override AttackPunch GetPunch()
    {
        // 获取玩家角色的拳击动作
        var punch = GameUtils.GetPunch($"{_player}'S PUNCH");
        // 返回一个 AttackPunch 对象，包含拳击动作和是否为玩家角色的最佳拳击动作
        return new AttackPunch(punch, punch == _player.BestPunch);
    }

    // 重写父类的 FullSwing 方法
    protected override void FullSwing() // 340
    {
        // 输出玩家角色进行全力摆动的信息
        Write($"{_player} SWINGS AND ");
        // 如果对手的脆弱性为 FullSwing，则得分
        if (Other.Vulnerability == Punch.FullSwing)
        {
            ScoreFullSwing();
        }
        else
        {
            // 如果随机数满足条件，则得分
            if (RollSatisfies(30, x => x < 10))
            {
                ScoreFullSwing();
            }
            else
            {
                // 否则输出“HE MISSES”
                WriteLine("HE MISSES");
            }
        }

        // 定义内部方法 ScoreFullSwing
        void ScoreFullSwing()
        {
            // 输出“HE CONNECTS!”
            WriteLine("HE CONNECTS!");
            // 如果对手受到的伤害大于击倒阈值，则注册击倒事件
            if (Other.DamageTaken > KnockoutDamageThreshold)
            {
                Work.Push(() => RegisterKnockout($"{Other} IS KNOCKED COLD AND {_player} IS THE WINNER AND CHAMP!"));
            }
            // 对手受到的伤害增加 15 点
            Other.DamageTaken += 15;
        }
    }

    // 重写父类的 Uppercut 方法
    protected override void Uppercut() // 520
    {
        // 输出玩家角色尝试上勾拳的信息
        Write($"{_player} TRIES AN UPPERCUT ");
        // 如果对手的脆弱性为 Uppercut，则得分
        if (Other.Vulnerability == Punch.Uppercut)
        {
            ScoreUpperCut();
        }
        else
        {
            // 如果随机数满足条件，则得分
            if (RollSatisfies(100, x => x < 51))
            {
                ScoreUpperCut();
            }
            else
            {
                // 否则输出“AND IT'S BLOCKED (LUCKY BLOCK!)”
                WriteLine("AND IT'S BLOCKED (LUCKY BLOCK!)");
            }
        }

        // 定义内部方法 ScoreUpperCut
        void ScoreUpperCut()
        {
            // 输出“AND HE CONNECTS!”
            WriteLine("AND HE CONNECTS!");
            // 对手受到的伤害增加 4 点
            Other.DamageTaken += 4;
        }
    }

    // 重写父类的 Hook 方法
    protected override void Hook() // 450
    {
        // 输出玩家进行钩拳的动作
        Write($"{_player} GIVES THE HOOK... ");
        // 如果对手的脆弱性是钩拳，则得分
        if (Other.Vulnerability == Punch.Hook)
        {
            ScoreHookOnOpponent();
        }
        else
        {
            // 如果掷骰子的结果满足条件，则输出被阻挡的信息，否则得分
            if (RollSatisfies(2, x => x == 1))
            {
                WriteLine("BUT IT'S BLOCKED!!!!!!!!!!!!!");
            }
            else
            {
                ScoreHookOnOpponent();
            }
        }

        // 定义得分钩拳的方法
        void ScoreHookOnOpponent()
        {
            // 输出连接成功的信息
            WriteLine("CONNECTS...");
            // 对手受到7点伤害
            Other.DamageTaken += 7;
        }
    }

    // 重写父类的Jab方法
    protected override void Jab()
    {
        // 输出玩家进行快拳的动作
        WriteLine($"{_player} JABS AT {Other}'S HEAD");
        // 如果对手的脆弱性是快拳，则得分
        if (Other.Vulnerability == Punch.Jab)
        {
            ScoreJabOnOpponent();
        }
        else
        {
            // 如果掷骰子的结果满足条件，则输出被阻挡的信息，否则得分
            if (RollSatisfies(8, x => x < 4))
            {
                WriteLine("IT'S BLOCKED.");
            }
            else
            {
                ScoreJabOnOpponent();
            }
        }

        // 定义得分快拳的方法，对手受到3点伤害
        void ScoreJabOnOpponent() => Other.DamageTaken += 3;
    }
# 闭合前面的函数定义
```