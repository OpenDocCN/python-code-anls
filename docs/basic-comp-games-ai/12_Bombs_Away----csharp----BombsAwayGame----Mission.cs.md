# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\Mission.cs`

```
# 声明一个命名空间为BombsAwayGame
namespace BombsAwayGame;

# 声明一个内部的记录类Mission，表示可以由MissionSide飞行的任务
# 该类包含两个属性：Name（任务名称）和Description（任务描述）
internal record class Mission(string Name, string Description);
```