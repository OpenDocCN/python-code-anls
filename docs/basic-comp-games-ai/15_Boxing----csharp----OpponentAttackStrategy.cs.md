# `basic-computer-games\15_Boxing\csharp\OpponentAttackStrategy.cs`

```
// 使用静态导入来引入 GameUtils 和 Console 类
using static Boxing.GameUtils;
using static System.Console;

// 命名空间 Boxing
namespace Boxing
{
    // OpponentAttackStrategy 类继承自 AttackStrategy 类
    public class OpponentAttackStrategy : AttackStrategy
    {
        // Opponent 对象的只读字段
        private readonly Opponent _opponent;

        // OpponentAttackStrategy 类的构造函数
        public OpponentAttackStrategy(Opponent opponent, Boxer player,  Action notifyGameEnded, Stack<Action> work) : base(player, work, notifyGameEnded)
        {
            _opponent = opponent;
        }

        // 重写父类的 GetPunch 方法
        protected override AttackPunch GetPunch()
        {
            // 生成一个随机的 Punch 枚举值
            var punch = (Punch)Roll(4);
            // 返回一个新的 AttackPunch 对象
            return new AttackPunch(punch, punch == _opponent.BestPunch);
        }

        // 重写父类的 FullSwing 方法
        protected override void FullSwing() // 720
        {
            // 输出对手进行全力挥拳的信息
            Write($"{_opponent}  TAKES A FULL SWING AND");
            // 如果对手的脆弱性与 FullSwing 相同
            if (Other.Vulnerability == Punch.FullSwing)
            {
                // 计分并执行相关操作
                ScoreFullSwing();
            }
            else
            {
                // 如果满足条件，输出挥拳被挡住的信息，否则计分并执行相关操作
                if (RollSatisfies(60, x => x < 30))
                {
                    WriteLine(" IT'S BLOCKED!");
                }
                else
                {
                    ScoreFullSwing();
                }
            }

            // 内部方法，计分并执行相关操作
            void ScoreFullSwing()
            {
                WriteLine(" POW!!!!! HE HITS HIM RIGHT IN THE FACE!");
                // 如果对手受到的伤害超过击倒阈值，将 RegisterOtherKnockedOut 方法推入工作栈
                if (Other.DamageTaken > KnockoutDamageThreshold)
                {
                    Work.Push(RegisterOtherKnockedOut);
                }
                Other.DamageTaken += 15;
            }
        }

        // 重写父类的 Hook 方法
        protected override void Hook() // 810
        {
            // 输出对手击中 Other 的信息
            Write($"{_opponent} GETS {Other} IN THE JAW (OUCH!)");
            // 增加 Other 受到的伤害值
            Other.DamageTaken += 7;
            // 输出再次击中的信息
            WriteLine("....AND AGAIN!");
            // 增加 Other 受到的伤害值
            Other.DamageTaken += 5;
            // 如果对手受到的伤害超过击倒阈值，将 RegisterOtherKnockedOut 方法推入工作栈
            if (Other.DamageTaken > KnockoutDamageThreshold)
            {
                Work.Push(RegisterOtherKnockedOut);
            }
        }

        // 重写父类的 Uppercut 方法
        protected override void Uppercut() // 860
    {
        // 输出信息，表示另一个角色被上勾击中
        Write($"{Other} IS ATTACKED BY AN UPPERCUT (OH,OH)...");
        // 如果另一个角色的脆弱性是上勾击，则调用ScoreUppercut函数
        if (Other.Vulnerability == Punch.Uppercut)
        {
            ScoreUppercut();
        }
        else
        {
            // 如果随机数满足条件，则输出信息并给对手造成伤害
            if (RollSatisfies(200, x => x > 75))
            {
                WriteLine($" BLOCKS AND HITS {_opponent} WITH A HOOK.");
                _opponent.DamageTaken += 5;
            }
            else
            {
                // 否则调用ScoreUppercut函数
                ScoreUppercut();
            }
        }
    
        // 定义ScoreUppercut函数，输出信息并给另一个角色造成伤害
        void ScoreUppercut()
        {
            WriteLine($"AND {_opponent} CONNECTS...");
            Other.DamageTaken += 8;
        }
    }
    
    // 重写Jab方法
    protected override void Jab() // 640
    {
        // 输出信息，表示对手使用左直拳
        Write($"{_opponent}  JABS AND ");
        // 如果另一个角色的脆弱性是左直拳，则调用ScoreJab函数
        if (Other.Vulnerability == Punch.Jab)
        {
            ScoreJab();
        }
        else
        {
            // 如果随机数满足条件，则输出信息并调用ScoreJab函数
            if (RollSatisfies(7, x => x > 4))
            {
                WriteLine("BLOOD SPILLS !!!");
                ScoreJab();
            }
            else
            {
                // 否则输出信息表示左直拳被格挡
                WriteLine("IT'S BLOCKED!");
            }
        }
    
        // 定义ScoreJab函数，给另一个角色造成伤害
        void ScoreJab() => Other.DamageTaken += 5;
    }
    
    // 注册另一个角色被击倒的事件
    private void RegisterOtherKnockedOut()
        => RegisterKnockout($"{Other} IS KNOCKED COLD AND {_opponent} IS THE WINNER AND CHAMP!");
# 闭合前面的函数定义
```