# `17_Bullfight\csharp\BullFight.cs`

```
using System;  // 引入 System 命名空间
using System.Collections.Generic;  // 引入 System.Collections.Generic 命名空间

namespace Game  // 命名空间 Game
{
    /// <summary>
    /// 提供模拟斗牛比赛的方法。
    /// </summary>
    public static class BullFight  // 定义名为 BullFight 的静态类
    {
        /// <summary>
        /// 开始一场新的比赛。
        /// </summary>
        /// <param name="mediator">
        /// 用于与玩家进行通信的对象。
        /// </param>
        /// <returns>
        /// 比赛期间发生的事件序列。
        /// </returns>
        /// <remarks>
```
```csharp
        # 创建一个随机数生成器对象
        random = Random()
        # 初始化结果为战斗继续
        result = ActionResult.FightContinues

        # 获取斗牛士的质量
        bullQuality = GetBullQuality()
        # 获取斗牛士的表现质量
        toreadorePerformance = GetHelpQuality(bullQuality)
        # 获取斗牛士的帮助质量
        picadorePerformance = GetHelpQuality(bullQuality)

        # 计算斗牛的强度
        bullStrength = 6 - int(bullQuality)
        # 计算帮助水平
        assistanceLevel = (12 - int(toreadorePerformance) - int(picadorePerformance)) * 0.1
        # 初始化勇气值
        bravery = 1.0
        # 初始化风格值
        style = 1.0
        # 初始化通过次数
        passNumber = 0

        # 返回比赛开始事件
        yield Events.MatchStarted(
                bullQuality,  # 定义变量 bullQuality，表示斗牛的质量
                toreadorePerformance,  # 定义变量 toreadorePerformance，表示斗牛士的表现
                picadorePerformance,  # 定义变量 picadorePerformance，表示刺马士的表现
                GetHumanCasualties(toreadorePerformance),  # 调用函数 GetHumanCasualties，计算斗牛士表现导致的人员伤亡
                GetHumanCasualties(picadorePerformance),  # 调用函数 GetHumanCasualties，计算刺马士表现导致的人员伤亡
                GetHorseCasualties(picadorePerformance));  # 调用函数 GetHorseCasualties，计算刺马士表现导致的马匹伤亡

            while (result == ActionResult.FightContinues)  # 当结果为继续战斗时进入循环
            {
                yield return new Events.BullCharging(++passNumber);  # 返回一个新的事件 BullCharging，并递增 passNumber

                var (action, riskLevel) = mediator.GetInput<(Action, RiskLevel)>();  # 从中介者获取输入的行动和风险级别
                result = action switch  # 根据行动类型进行判断
                {
                    Action.Dodge => TryDodge(riskLevel),  # 如果行动是躲避，则调用 TryDodge 函数
                    Action.Kill  => TryKill(riskLevel),  # 如果行动是杀死，则调用 TryKill 函数
                    _            => Panic()  # 其他情况则调用 Panic 函数
                };

                var first = true;  # 定义变量 first，表示是否第一次循环
                while (result == ActionResult.BullGoresPlayer)  # 当结果为ActionResult.BullGoresPlayer时，进入循环
                {
                    yield return new Events.PlayerGored(action == Action.Panic, first);  # 返回一个PlayerGored事件，根据action是否为Panic和是否是第一次
                    first = false;  # 将first设置为false

                    result = TrySurvive();  # 调用TrySurvive方法，将结果赋给result
                    if (result == ActionResult.FightContinues)  # 如果结果为ActionResult.FightContinues
                    {
                        yield return new Events.PlayerSurvived();  # 返回一个PlayerSurvived事件

                        var runFromRing = mediator.GetInput<bool>();  # 从mediator获取一个bool类型的输入，赋给runFromRing
                        if (runFromRing)  # 如果runFromRing为true
                            result = Flee();  # 调用Flee方法，将结果赋给result
                        else
                            result = IgnoreInjury(action);  # 调用IgnoreInjury方法，将结果赋给result
                    }
                }
            }

            yield return new Events.MatchCompleted(  # 返回一个MatchCompleted事件
                result,  # 变量 result 未定义，需要添加注释说明其作用
                bravery == 2,  # 检查 bravery 是否等于 2，需要添加注释说明其作用
                GetReward());  # 调用 GetReward() 函数，需要添加注释说明其作用

            Quality GetBullQuality() =>  # 定义一个函数 GetBullQuality()，返回类型为 Quality，需要添加注释说明其作用
                (Quality)random.Next(1, 6);  # 使用随机数生成牛的质量，需要添加注释说明其作用

            Quality GetHelpQuality(Quality bullQuality) =>  # 定义一个函数 GetHelpQuality()，参数为 bullQuality，返回类型为 Quality，需要添加注释说明其作用
                ((3.0 / (int)bullQuality) * random.NextDouble()) switch  # 根据牛的质量计算帮助的质量，使用 switch 语句，需要添加注释说明其作用
                {
                    < 0.37 => Quality.Superb,  # 如果计算结果小于 0.37，则返回 Quality.Superb
                    < 0.50 => Quality.Good,  # 如果计算结果小于 0.50，则返回 Quality.Good
                    < 0.63 => Quality.Fair,  # 如果计算结果小于 0.63，则返回 Quality.Fair
                    < 0.87 => Quality.Poor,  # 如果计算结果小于 0.87，则返回 Quality.Poor
                    _      => Quality.Awful  # 其他情况返回 Quality.Awful
                };

            int GetHumanCasualties(Quality performance) =>  # 定义一个函数 GetHumanCasualties()，参数为 performance，返回类型为 int，需要添加注释说明其作用
                performance switch  # 根据 performance 的值进行不同的处理，需要添加注释说明其作用
                {
                    Quality.Poor  => random.Next(0, 2),  // 如果表演质量为Poor，返回0或1
                    Quality.Awful => random.Next(1, 3),  // 如果表演质量为Awful，返回1或2
                    _             => 0  // 其他情况返回0
                };

            int GetHorseCasualties(Quality performance) =>
                performance switch
                {
                    // 注意：在原始的BASIC版本中，对于Poor表演质量导致的马匹伤亡的代码是无法到达的。我假设这是一个bug。
                    Quality.Poor  => 1,  // 如果表演质量为Poor，返回1
                    Quality.Awful => random.Next(1, 3),  // 如果表演质量为Awful，返回1或2
                    _             => 0  // 其他情况返回0
                };

            ActionResult TryDodge(RiskLevel riskLevel)
            {
                var difficultyModifier = riskLevel switch
                {
                    RiskLevel.High   => 3.0,  # 如果风险级别为高，则返回3.0
                    RiskLevel.Medium => 2.0,  # 如果风险级别为中等，则返回2.0
                    _                => 0.5   # 其他情况返回0.5
                };

                var outcome = (bullStrength + (difficultyModifier / 10)) * random.NextDouble() /
                    ((assistanceLevel + (passNumber / 10.0)) * 5);  # 计算结果的公式

                if (outcome < 0.51)  # 如果结果小于0.51
                {
                    style += difficultyModifier;  # 根据难度修饰器增加风格
                    return ActionResult.FightContinues;  # 返回继续战斗的结果
                }
                else
                    return ActionResult.BullGoresPlayer;  # 否则返回公牛攻击玩家的结果
            }

            ActionResult TryKill(RiskLevel riskLevel)  # 尝试杀死的方法，参数为风险级别
                # 计算运气值，根据公式：牛的力量 * 10 * 一个随机小数 / （辅助水平 * 5 * 通过次数）
                var luck = bullStrength * 10 * random.NextDouble() / (assistanceLevel * 5 * passNumber);

                # 根据运气值和风险级别判断结果，如果风险级别为高且运气值大于0.2，或者运气值大于0.8，则返回牛攻击玩家，否则返回玩家击败牛
                return ((riskLevel == RiskLevel.High && luck > 0.2) || luck > 0.8) ?
                    ActionResult.BullGoresPlayer : ActionResult.PlayerKillsBull;
            }

            # 当处于恐慌状态时的行为，直接返回牛攻击玩家
            ActionResult Panic() =>
                ActionResult.BullGoresPlayer;

            # 尝试生存时的行为
            ActionResult TrySurvive()
            {
                # 如果随机数为0，则勇气值为1.5，返回牛杀死玩家；否则返回战斗继续
                if (random.Next(2) == 0)
                {
                    bravery = 1.5;
                    return ActionResult.BullKillsPlayer;
                }
                else
                    return ActionResult.FightContinues;
            }
            # 定义逃跑行为，将勇气值设为0.0，然后返回玩家逃跑的结果
            ActionResult Flee()
            {
                bravery = 0.0;
                return ActionResult.PlayerFlees;
            }

            # 定义忽视伤害行为，如果随机数为0，则将勇气值设为2.0，并根据行为类型返回不同的结果；否则返回公牛刺伤玩家的结果
            ActionResult IgnoreInjury(Action action)
            {
                if (random.Next(2) == 0)
                {
                    bravery = 2.0;
                    return action == Action.Dodge ? ActionResult.FightContinues : ActionResult.Draw;
                }
                else
                    return ActionResult.BullGoresPlayer;
            }

            # 获取奖励，计算得分并返回
            Reward GetReward()
            {
                var score = CalculateScore();
                if (score * random.NextDouble() < 2.4)  # 如果得分乘以随机数小于2.4
                    return Reward.Nothing;  # 返回奖励为Nothing
                else
                if (score * random.NextDouble() < 4.9)  # 否则，如果得分乘以随机数小于4.9
                    return Reward.OneEar;  # 返回奖励为OneEar
                else
                if (score * random.NextDouble() < 7.4)  # 否则，如果得分乘以随机数小于7.4
                    return Reward.TwoEars;  # 返回奖励为TwoEars
                else
                    return Reward.CarriedFromRing;  # 否则，返回奖励为CarriedFromRing
            }

            double CalculateScore()  # 定义一个函数CalculateScore，返回类型为double
            {
                var score = 4.5;  # 初始化得分为4.5

                // Style
                score += style / 6;  # 根据style的值增加得分
                // 减去辅助程度的影响
                score -= assistanceLevel * 2.5;

                // 勇气加成
                score += 4 * bravery;

                // 击杀奖励
                score += (result == ActionResult.PlayerKillsBull) ? 4 : 2;

                // 比赛时长影响
                score -= Math.Pow(passNumber, 2) / 120;

                // 难度影响
                score -= (int)bullQuality;

                return score;
            }
        }
    }
}
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```