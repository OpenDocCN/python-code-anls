# `27_Civil_War\csharp\GameOptions.cs`

```
            if (Yes("Do you want instructions?"))
            {
                Console.WriteLine(instructions);
            }

            return new GameOptions(Yes("Do you want to play a two player game?"),
                                   Yes("Do you want to see the descriptions of the armies?"));
        }
    }
}
# The object of the game is to win as many battles as possible.
# 游戏的目标是赢得尽可能多的战斗。

# Your choices for defensive strategy are:
# (1) artillery attack
# (2) fortification against frontal attack
# (3) fortification against flanking maneuvers
# (4) falling back
# 防御策略的选择包括：
# (1) 炮击
# (2) 针对正面进攻的防御工事
# (3) 针对侧翼进攻的防御工事
# (4) 撤退

# Your choices for offensive strategy are:
# (1) artillery attack
# (2) frontal attack
# (3) flanking maneuvers
# (4) encirclement
# 进攻策略的选择包括：
# (1) 炮击
# (2) 正面进攻
# (3) 侧翼进攻
# (4) 包围
# You may surrender by typing a '5' for your strategy.";
# 你可以通过输入'5'来投降。

if (InputYesOrNo())
    WriteWordWrap(instructions);
# 如果玩家选择输入Yes或No，则显示游戏说明。

Console.WriteLine("\n\nAre there two generals present?");
bool twoPlayers = InputYesOrNo();
# 控制台输出“是否有两位将军在场？”并根据输入判断是否有两位玩家参与游戏。
if (!twoPlayers)
# 输出提示信息
Console.WriteLine("\nYou are the confederacy.  Good luck!\n");

# 调用函数输出提示信息
WriteWordWrap(
@"Select a battle by typing a number from 1 to 14 on request.  Type any other number to end the simulation. But '0' brings back exact previous battle situation allowing you to replay it.

Note: a negative Food$ entry causes the program to use the entries from the previous battle

After requesting a battle, do you wish battle descriptions (answer yes or no)");

# 调用函数获取用户输入，判断是否需要显示战斗描述
bool showDescriptions = InputYesOrNo();

# 返回一个包含两个参数的 GameOptions 对象
return new GameOptions(twoPlayers, showDescriptions);
```