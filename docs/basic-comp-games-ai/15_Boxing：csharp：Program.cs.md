# `d:/src/tocomm/basic-computer-games\15_Boxing\csharp\Program.cs`

```
// 引入 Boxing 命名空间
using Boxing;
// 使用 Boxing.GameUtils 中的静态成员
using static Boxing.GameUtils;
// 使用 System.Console 中的 WriteLine 方法
using static System.Console;

// 打印标题
WriteLine(new string('\t', 33) + "BOXING");
// 打印创意计算的信息
WriteLine(new string('\t', 15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
// 打印游戏规则
WriteLine("{0}{0}{0}BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS){0}", Environment.NewLine);

// 创建对手对象
var opponent = new Opponent();
// 设置对手的名字
opponent.SetName("WHAT IS YOUR OPPONENT'S NAME"); // J$
// 创建玩家对象
var player = new Boxer();
// 设置玩家的名字
player.SetName("INPUT YOUR MAN'S NAME"); // L$

// 打印拳击描述
PrintPunchDescription();
// 设置玩家的最佳拳
player.BestPunch = GetPunch("WHAT IS YOUR MANS BEST"); // B
// 设置玩家的脆弱性
player.Vulnerability = GetPunch("WHAT IS HIS VULNERABILITY"); // D
// 对手随机选择拳
opponent.SetRandomPunches();
// 打印对手的优势和脆弱性
WriteLine($"{opponent}'S ADVANTAGE IS {opponent.BestPunch.ToFriendlyString()} AND VULNERABILITY IS SECRET.");
for (var i = 1; i <= 3; i ++) // 使用循环执行以下代码块3次，i从1开始，每次增加1
{
    var round = new Round(player, opponent, i); // 创建一个名为round的新回合对象，传入player、opponent和i作为参数
    round.Start(); // 开始回合
    round.CheckOpponentWin(); // 检查对手是否获胜
    round.CheckPlayerWin(); // 检查玩家是否获胜
    if (round.GameEnded) break; // 如果游戏结束，跳出循环
}
WriteLine("{0}{0}AND NOW GOODBYE FROM THE OLYMPIC ARENA.{0}", Environment.NewLine); // 在控制台输出指定的字符串
```