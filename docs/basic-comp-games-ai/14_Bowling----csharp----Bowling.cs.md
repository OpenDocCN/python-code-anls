# `14_Bowling\csharp\Bowling.cs`

```
# 创建一个名为Bowling的命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bowling
{
    # 创建Bowling类
    public class Bowling
    {
        # 创建一个名为pins的私有变量，并初始化为一个新的Pins对象
        private readonly Pins pins = new();

        # 创建一个名为players的私有变量
        private int players;

        # 创建一个名为Play的公共方法
        public void Play()
        {
            # 调用ShowBanner方法
            ShowBanner();
            # 调用MaybeShowInstructions方法
            MaybeShowInstructions();
            # 调用Setup方法
            Setup();
            # 调用GameLoop方法
            GameLoop();
        }

        private static void ShowBanner()
        {
            // 打印游戏的横幅
            Utility.PrintString(34, "BOWL");
            Utility.PrintString(15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Utility.PrintString();
            Utility.PrintString();
            Utility.PrintString();
            Utility.PrintString("WELCOME TO THE ALLEY");
            Utility.PrintString("BRING YOUR FRIENDS");
            Utility.PrintString("OKAY LET'S FIRST GET ACQUAINTED");
            Utility.PrintString();
        }
        private static void MaybeShowInstructions()
        {
            // 可能显示游戏说明
            Utility.PrintString("THE INSTRUCTIONS (Y/N)");
            // 如果用户输入为"N"，则返回
            if (Utility.InputString() == "N") return;
            // 打印游戏说明
            Utility.PrintString("THE GAME OF BOWLING TAKES MIND AND SKILL.DURING THE GAME");
            Utility.PrintString("THE COMPUTER WILL KEEP SCORE.YOU MAY COMPETE WITH");
            # 打印游戏规则提示信息
            Utility.PrintString("OTHER PLAYERS[UP TO FOUR].YOU WILL BE PLAYING TEN FRAMES");
            Utility.PrintString("ON THE PIN DIAGRAM 'O' MEANS THE PIN IS DOWN...'+' MEANS THE");
            Utility.PrintString("PIN IS STANDING.AFTER THE GAME THE COMPUTER WILL SHOW YOUR");
            Utility.PrintString("SCORES .");
        }
        # 设置游戏
        private void Setup()
        {
            # 打印提示信息，询问玩家数量
            Utility.PrintString("FIRST OF ALL...HOW MANY ARE PLAYING", false);
            # 获取玩家输入的数量
            var input = Utility.InputInt();
            # 如果输入小于1，则默认设置为1
            players = input < 1 ? 1 : input;
            # 打印提示信息
            Utility.PrintString();
            Utility.PrintString("VERY GOOD...");
        }
        # 游戏循环
        private void GameLoop()
        {
            # 初始化游戏结果数组
            GameResults[] gameResults = InitGameResults();
            # 设置循环标志为false
            var done = false;
            # 循环直到游戏结束
            while (!done)
            {
                # 重置游戏结果
                ResetGameResults(gameResults);
                for (int frame = 0; frame < GameResults.FramesPerGame; ++frame)
                {
                    // 循环每一局比赛的帧数
                    for (int player = 0; player < players; ++player)
                    {
                        // 循环每个玩家
                        pins.Reset();
                        // 重置击倒的瓶子数量
                        int pinsDownThisFrame = pins.GetPinsDown();
                        // 获取当前帧击倒的瓶子数量

                        int ball = 1;
                        // 初始化球的数量为1
                        while (ball == 1 || ball == 2) // One or two rolls
                        {
                            // 当球的数量为1或2时，进行循环，表示一次或两次投球
                            Utility.PrintString("TYPE ROLL TO GET THE BALL GOING.");
                            // 打印提示信息，要求输入ROLL来进行投球
                            _ = Utility.InputString();
                            // 获取输入的字符串，但不使用

                            int pinsDownAfterRoll = pins.Roll();
                            // 进行投球，获取击倒的瓶子数量
                            ShowPins(player, frame, ball);
                            // 显示当前玩家、帧数和球数的瓶子状态

                            if (pinsDownAfterRoll == pinsDownThisFrame)
                            {
                                // 如果投球后的击倒瓶子数量等于当前帧的击倒瓶子数量
                                Utility.PrintString("GUTTER!!");
                                // 打印提示信息，表示击倒了零瓶
                            }
                            if (ball == 1)
                            {
                                // 如果当前是第一次投球

                                // 存储当前击倒的瓶数
                                gameResults[player].Results[frame].PinsBall1 = pinsDownAfterRoll;

                                // 如果是全中（击倒全部瓶子）
                                if (pinsDownAfterRoll == Pins.TotalPinCount)
                                {
                                    // 打印STRIKE的提示音
                                    Utility.PrintString("STRIKE!!!!!\a\a\a\a");
                                    // 没有第二次投球
                                    ball = 0;
                                    gameResults[player].Results[frame].PinsBall2 = pinsDownAfterRoll;
                                    gameResults[player].Results[frame].Score = FrameResult.Points.Strike;
                                }
                                else
                                {
                                    ball = 2; // 再投一次
                                    Utility.PrintString("ROLL YOUR SECOND BALL");
                                }
                            }
                            else if (ball == 2) // 如果是第二次投球
                            {
                                // 存储当前击倒的瓶数
                                gameResults[player].Results[frame].PinsBall2 = pinsDownAfterRoll;
                                ball = 0; // 重置投球次数

                                // 确定该轮的得分
                                if (pinsDownAfterRoll == Pins.TotalPinCount) // 如果全部击倒
                                {
                                    Utility.PrintString("SPARE!!!!"); // 打印“SPARE!!!!”
                                    gameResults[player].Results[frame].Score = FrameResult.Points.Spare; // 设置该轮得分为SPARE
                                }
                                else // 如果未全部击倒
                                {
                                    Utility.PrintString("ERROR!!!"); // 打印“ERROR!!!”
                                    gameResults[player].Results[frame].Score = FrameResult.Points.Error; // 设置该轮得分为ERROR
                                }
                            }
                            Utility.PrintString(); // 打印空行
            }
        }
    }
}
ShowGameResults(gameResults);  // 调用 ShowGameResults 函数，传入 gameResults 参数
Utility.PrintString("DO YOU WANT ANOTHER GAME");  // 调用 Utility 类的 PrintString 方法，打印提示信息
var a = Utility.InputString();  // 调用 Utility 类的 InputString 方法，获取用户输入并赋值给变量 a
done = a.Length == 0 || a[0] != 'Y';  // 判断用户输入是否为空或者首字母不是 'Y'，将结果赋值给 done 变量
```
```csharp
private GameResults[] InitGameResults()
{
    var gameResults = new GameResults[players];  // 创建长度为 players 的 GameResults 数组
    for (int i = 0; i < gameResults.Length; i++)  // 循环遍历 gameResults 数组
    {
        gameResults[i] = new GameResults();  // 对每个元素赋值为新的 GameResults 对象
    }
    return gameResults;  // 返回初始化后的 gameResults 数组
}
        // 显示球员、帧数和球数信息
        private void ShowPins(int player, int frame, int ball)
        {
            // 打印帧数、球员和球数信息
            Utility.PrintString($"FRAME: {frame + 1} PLAYER: {player + 1} BALL: {ball}");
            // 定义击倒瓶子的布尔数组
            var breakPins = new bool[] { true, false, false, false, true, false, false, true, false, true };
            // 初始化缩进
            var indent = 0;
            // 遍历瓶子
            for (int pin = 0; pin < Pins.TotalPinCount; ++pin)
            {
                // 如果需要换行
                if (breakPins[pin])
                {
                    Utility.PrintString(); // 结束当前行
                    Utility.PrintString(indent++, false); // 缩进下一行
                }
                // 根据瓶子状态打印相应符号
                var s = pins[pin] == Pins.State.Down ? "+ " : "o ";
                Utility.PrintString(s, false);
            }
            Utility.PrintString(); // 打印空行
            Utility.PrintString(); // 打印空行
        }
        
        // 重置游戏结果
        private void ResetGameResults(GameResults[] gameResults)
        {
            // 遍历游戏结果数组
            foreach (var gameResult in gameResults)
            {
                // 遍历每个游戏结果中的帧结果
                foreach (var frameResult in gameResult.Results)
                {
                    // 重置每个帧结果
                    frameResult.Reset();
                }
            }
        }
        // 显示游戏结果
        private void ShowGameResults(GameResults[] gameResults)
        {
            // 打印"FRAMES"字符串
            Utility.PrintString("FRAMES");
            // 循环打印帧数
            for (int i = 0; i < GameResults.FramesPerGame; ++i)
            {
                // 打印帧数，并且不换行
                Utility.PrintString(Utility.PadInt(i, 3), false);
            }
            // 打印空行
            Utility.PrintString();
            // 遍历游戏结果数组
            foreach (var gameResult in gameResults)
            {
                // 遍历每个游戏结果中的帧结果
                foreach (var frameResult in gameResult.Results)
                {
# 打印第一球击倒的瓶数，不换行
Utility.PrintString(Utility.PadInt(frameResult.PinsBall1, 3), false);
# 打印换行
Utility.PrintString();
# 遍历每个帧的结果，打印第二球击倒的瓶数，不换行
foreach (var frameResult in gameResult.Results)
{
    Utility.PrintString(Utility.PadInt(frameResult.PinsBall2, 3), false);
}
# 打印换行
Utility.PrintString();
# 遍历每个帧的结果，打印分数，不换行
foreach (var frameResult in gameResult.Results)
{
    Utility.PrintString(Utility.PadInt((int)frameResult.Score, 3), false);
}
# 打印换行
Utility.PrintString();
# 打印两个空行
Utility.PrintString();
Utility.PrintString();
```