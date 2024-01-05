# `15_Boxing\csharp\PlayerAttackStrategy.cs`

```
// 使用静态类 GameUtils 中的 GetPunch 方法
var punch = GameUtils.GetPunch($"{_player}'S PUNCH");
// 创建一个 AttackPunch 对象，传入获取到的拳击动作和判断是否为玩家最佳拳击动作的布尔值
return new AttackPunch(punch, punch == _player.BestPunch);
// 输出玩家的名字和动作
Write($"{_player} SWINGS AND ");
        # 如果对手的 Vulnerability 是 FullSwing，则调用 ScoreFullSwing 函数
        if (Other.Vulnerability == Punch.FullSwing)
        {
            ScoreFullSwing();
        }
        # 如果不是 FullSwing，则判断 RollSatisfies 函数的返回值
        else
        {
            # 如果 RollSatisfies 函数返回 true，则调用 ScoreFullSwing 函数
            if (RollSatisfies(30, x => x < 10))
            {
                ScoreFullSwing();
            }
            # 如果 RollSatisfies 函数返回 false，则输出 "HE MISSES"
            else
            {
                WriteLine("HE MISSES");
            }
        }

        # 定义 ScoreFullSwing 函数
        void ScoreFullSwing()
        {
            # 输出 "HE CONNECTS!"
            WriteLine("HE CONNECTS!");
            # 如果对手受到的伤害大于 KnockoutDamageThreshold，则执行以下代码
            if (Other.DamageTaken > KnockoutDamageThreshold)
            {
                // 将一个动作推入工作队列，该动作是注册一个击倒事件的回调函数
                Work.Push(() => RegisterKnockout($"{Other} IS KNOCKED COLD AND {_player} IS THE WINNER AND CHAMP!"));
            }
            // 增加对手的受伤程度
            Other.DamageTaken += 15;
        }
    }

    protected override void Uppercut() // 520
    {
        // 输出玩家尝试进行上勾拳的信息
        Write($"{_player} TRIES AN UPPERCUT ");
        // 如果对手的脆弱性是上勾拳，则得分
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
                WriteLine("AND IT'S BLOCKED (LUCKY BLOCK!)"); // 如果条件不满足，则输出信息“AND IT'S BLOCKED (LUCKY BLOCK!)”
            }
        }

        void ScoreUpperCut() // 定义一个名为ScoreUpperCut的函数
        {
            WriteLine("AND HE CONNECTS!"); // 输出信息“AND HE CONNECTS!”
            Other.DamageTaken += 4; // 对Other对象的DamageTaken属性增加4
        }
    }

    protected override void Hook() // 450 // 重写基类的Hook方法，注释为450
    {
        Write($"{_player} GIVES THE HOOK... "); // 输出信息“_player GIVES THE HOOK...”
        if (Other.Vulnerability == Punch.Hook) // 如果Other对象的Vulnerability属性等于Punch.Hook
        {
            ScoreHookOnOpponent(); // 调用ScoreHookOnOpponent方法
        }
        else
        {
            // 如果骰子满足条件，输出信息
            if (RollSatisfies(2, x => x == 1))
            {
                WriteLine("BUT IT'S BLOCKED!!!!!!!!!!!!!");
            }
            // 如果不满足条件，执行ScoreHookOnOpponent函数
            else
            {
                ScoreHookOnOpponent();
            }
        }

        // 定义ScoreHookOnOpponent函数
        void ScoreHookOnOpponent()
        {
            // 输出信息
            WriteLine("CONNECTS...");
            // 对手受到7点伤害
            Other.DamageTaken += 7;
        }
    }

    // 重写Jab方法
    protected override void Jab()
        # 输出玩家对其他人头部进行刺击的动作
        WriteLine($"{_player} JABS AT {Other}'S HEAD");
        # 如果对方的脆弱性是刺击，则得分
        if (Other.Vulnerability == Punch.Jab)
        {
            ScoreJabOnOpponent();
        }
        else
        {
            # 如果滚动满足条件，则显示“被阻挡”
            if (RollSatisfies(8, x => x < 4))
            {
                WriteLine("IT'S BLOCKED.");
            }
            else
            {
                # 否则得分
                ScoreJabOnOpponent();
            }
        }

        # 对方受到3点伤害
        void ScoreJabOnOpponent() => Other.DamageTaken += 3;
    }
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```