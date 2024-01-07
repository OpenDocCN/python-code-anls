# `basic-computer-games\04_Awari\csharp\Game.cs`

```

namespace Awari; // 声明命名空间为 Awari

public class Game // 声明一个名为 Game 的公共类
}

public enum GameState // 声明一个名为 GameState 的公共枚举
{
    PlayerMove, // 玩家移动状态
    PlayerSecondMove, // 玩家第二次移动状态
    ComputerMove, // 计算机移动状态
    ComputerSecondMove, // 计算机第二次移动状态
    Done, // 游戏结束状态
}

public enum GameWinner // 声明一个名为 GameWinner 的公共枚举
{
    Player, // 玩家获胜
    Computer, // 计算机获胜
    Draw, // 平局
}

public record struct GameOutcome(GameWinner Winner, int Difference); // 声明一个名为 GameOutcome 的记录结构，包含获胜者和差值

```