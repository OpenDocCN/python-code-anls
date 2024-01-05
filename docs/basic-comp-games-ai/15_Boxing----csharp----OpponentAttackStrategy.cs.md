# `15_Boxing\csharp\OpponentAttackStrategy.cs`

```
// 使用静态类 GameUtils 中的方法和属性，无需通过类名引用
using static Boxing.GameUtils;
// 使用静态类 System.Console 中的方法和属性，无需通过类名引用
using static System.Console;

// 命名空间 Boxing
namespace Boxing
{
    // 定义 OpponentAttackStrategy 类，继承自 AttackStrategy 类
    public class OpponentAttackStrategy : AttackStrategy
    {
        // 声明私有字段 _opponent，类型为 Opponent
        private readonly Opponent _opponent;

        // 定义 OpponentAttackStrategy 类的构造函数，接受 Opponent 对象、Boxer 对象、Action 委托和 Stack<Action> 对象作为参数
        // 调用基类 AttackStrategy 的构造函数
        public OpponentAttackStrategy(Opponent opponent, Boxer player,  Action notifyGameEnded, Stack<Action> work) : base(player, work, notifyGameEnded)
        {
            // 将传入的 opponent 参数赋值给 _opponent 字段
            _opponent = opponent;
        }

        // 重写基类的 GetPunch 方法
        protected override AttackPunch GetPunch()
        {
            // 生成一个 1 到 4 之间的随机数，并将其转换为 Punch 枚举类型
            var punch = (Punch)Roll(4);
            // 返回一个新的 AttackPunch 对象，包含随机生成的 punch 和判断是否为对手的最佳拳击的布尔值
            return new AttackPunch(punch, punch == _opponent.BestPunch);
        }
    }
}
    protected override void FullSwing() // 720
    {
        // 输出对手进行全力挥拳的动作
        Write($"{_opponent}  TAKES A FULL SWING AND");
        // 如果对手的脆弱度为 Punch.FullSwing，则进行得分
        if (Other.Vulnerability == Punch.FullSwing)
        {
            ScoreFullSwing();
        }
        else
        {
            // 如果随机数满足条件（60%的概率小于30），则输出挡住了全力挥拳的信息
            if (RollSatisfies(60, x => x < 30))
            {
                WriteLine(" IT'S BLOCKED!");
            }
            else
            {
                // 否则进行得分
                ScoreFullSwing();
            }
        }

        // 定义得分的方法
        void ScoreFullSwing()
        {
            WriteLine(" POW!!!!! HE HITS HIM RIGHT IN THE FACE!");  // 输出字符串" POW!!!!! HE HITS HIM RIGHT IN THE FACE!"
            if (Other.DamageTaken > KnockoutDamageThreshold)  // 如果对手受到的伤害大于击倒阈值
            {
                Work.Push(RegisterOtherKnockedOut);  // 将RegisterOtherKnockedOut推入Work栈
            }
            Other.DamageTaken += 15;  // 对手受到的伤害增加15点
        }
    }

    protected override void Hook() // 810
    {
        Write($"{_opponent} GETS {Other} IN THE JAW (OUCH!)");  // 输出字符串"{_opponent} GETS {Other} IN THE JAW (OUCH!)"
        Other.DamageTaken += 7;  // 对手受到的伤害增加7点
        WriteLine("....AND AGAIN!");  // 输出字符串"....AND AGAIN!"
        Other.DamageTaken += 5;  // 对手受到的伤害增加5点
        if (Other.DamageTaken > KnockoutDamageThreshold)  // 如果对手受到的伤害大于击倒阈值
        {
            Work.Push(RegisterOtherKnockedOut);  // 将RegisterOtherKnockedOut推入Work栈
        }
    } // 结束 Uppercut 方法的定义

    protected override void Uppercut() // 860
    {
        Write($"{Other} IS ATTACKED BY AN UPPERCUT (OH,OH)..."); // 打印对手被上勾拳攻击的信息
        if (Other.Vulnerability == Punch.Uppercut) // 如果对手的弱点是上勾拳
        {
            ScoreUppercut(); // 调用 ScoreUppercut 方法
        }
        else
        {
            if (RollSatisfies(200, x => x > 75)) // 如果满足条件：200以内的随机数大于75
            {
                WriteLine($" BLOCKS AND HITS {_opponent} WITH A HOOK."); // 打印阻挡并用钩拳攻击对手的信息
                _opponent.DamageTaken += 5; // 对手受到5点伤害
            }
            else
            {
                ScoreUppercut(); // 调用 ScoreUppercut 方法
            }
        }

        void ScoreUppercut()
        {
            WriteLine($"AND {_opponent} CONNECTS...");  // 输出对手连接的消息
            Other.DamageTaken += 8;  // 对手受到8点伤害
        }
    }

    protected override void Jab() // 640
    {
        Write($"{_opponent}  JABS AND ");  // 输出对手的快拳攻击
        if (Other.Vulnerability == Punch.Jab)  // 如果对手的弱点是快拳攻击
        {
            ScoreJab();  // 计分
        }
        else
        {
            if (RollSatisfies(7, x => x > 4))  // 如果满足条件
            {
                WriteLine("BLOOD SPILLS !!!");  // 输出字符串 "BLOOD SPILLS !!!"
                ScoreJab();  // 调用 ScoreJab() 方法
            }
            else
            {
                WriteLine("IT'S BLOCKED!");  // 输出字符串 "IT'S BLOCKED!"
            }
        }

        void ScoreJab() => Other.DamageTaken += 5;  // 定义 ScoreJab() 方法，使 Other.DamageTaken 值增加 5
    }

    private void RegisterOtherKnockedOut()
        => RegisterKnockout($"{Other} IS KNOCKED COLD AND {_opponent} IS THE WINNER AND CHAMP!");  // 调用 RegisterKnockout() 方法，传入字符串参数
}
```