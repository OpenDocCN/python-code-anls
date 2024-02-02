# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\EnemyArtillery.cs`

```py
// 命名空间声明，表示该类属于BombsAwayGame命名空间
namespace BombsAwayGame;

/// <summary>
/// 表示敌方炮兵
/// </summary>
/// <param name="Name">炮兵类型的名称</param>
/// <param name="Accuracy">炮兵的准确度。这是原始BASIC中的变量`T`。</param>
// 内部记录类声明，表示EnemyArtillery类是一个记录类，包含名称和准确度两个属性
internal record class EnemyArtillery(string Name, int Accuracy);
```