# `08_Batnum\csharp\BatnumGame.cs`

```
// 引入命名空间
using Batnum.Properties;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// 声明命名空间
namespace Batnum
{
    // 定义枚举类型
    public enum WinOptions
    {
        /// <summary>
        /// 最后一个玩家赢
        /// </summary>
        WinWithTakeLast = 1,
        /// <summary>
        /// 避免成为最后一个玩家
        /// </summary>
        WinWithAvoidLast = 2
    }
}
    # 定义一个枚举类型 Players，包含两个成员：Computer 和 Human
    public enum Players
    {
        Computer = 1,
        Human = 2
    }

    # 定义一个名为 BatnumGame 的类
    public class BatnumGame
    {
        # 定义 BatnumGame 类的构造函数，接受 pileSize（堆大小）、winCriteria（胜利条件）、minTake（最小取数）、maxTake（最大取数）、firstPlayer（先手玩家）、askPlayerCallback（询问玩家的回调函数）作为参数
        public BatnumGame(int pileSize, WinOptions winCriteria, int minTake, int maxtake, Players firstPlayer, Func<string, int>askPlayerCallback)
        {
            # 初始化 BatnumGame 类的成员变量
            this.pileSize = pileSize;
            this.winCriteria = winCriteria;
            this.minTake = minTake;
            this.maxTake = maxtake;
            this.currentPlayer = firstPlayer;
            this.askPlayerCallback = askPlayerCallback;
        }

        # 定义 BatnumGame 类的私有成员变量 pileSize
        private int pileSize;
        private WinOptions winCriteria; // 定义一个私有的变量用于存储游戏的胜利条件
        private int minTake; // 定义一个私有的变量用于存储最小取走数量
        private int maxTake; // 定义一个私有的变量用于存储最大取走数量
        private Players currentPlayer; // 定义一个私有的变量用于存储当前玩家
        private Func<string, int> askPlayerCallback; // 定义一个私有的委托类型变量，用于询问玩家的回调函数

        /// <summary>
        /// Returns true if the game is running
        /// </summary>
        public bool IsRunning => pileSize > 0; // 当堆的大小大于0时，返回true，表示游戏正在进行中

        /// <summary>
        /// Takes the next turn
        /// </summary>
        /// <returns>A message to be displayed to the player</returns>
        public string TakeTurn()
        {
            //Edge condition - can occur when minTake is more > 1
            if (pileSize < minTake) // 如果堆的大小小于最小取走数量
            {
                pileSize = 0;  # 重置变量 pileSize 的值为 0
                return string.Format(Resources.END_DRAW, minTake);  # 返回一个格式化的字符串，包含 Resources.END_DRAW 和 minTake 的值
            }
            return currentPlayer == Players.Computer ? ComputerTurn() : PlayerTurn();  # 如果当前玩家是电脑，则调用 ComputerTurn() 方法，否则调用 PlayerTurn() 方法
        }

        private string PlayerTurn()  # 定义一个名为 PlayerTurn 的私有方法
        {
            int draw = askPlayerCallback(Resources.INPUT_TURN);  # 调用 askPlayerCallback 方法，传入 Resources.INPUT_TURN，获取玩家输入的值
            if (draw == 0)  # 如果玩家输入的值为 0
            {
                pileSize = 0;  # 将变量 pileSize 的值设为 0
                return Resources.INPUT_ZERO;  # 返回字符串 Resources.INPUT_ZERO
            }
            if (draw < minTake || draw > maxTake || draw > pileSize)  # 如果玩家输入的值小于最小取数、大于最大取数或者大于当前堆的数量
            {
                return Resources.INPUT_ILLEGAL;  # 返回字符串 Resources.INPUT_ILLEGAL
            }
            pileSize = pileSize - draw;  # 将当前堆的数量减去玩家输入的值
            if (pileSize == 0)  # 如果当前堆的数量为 0
            {
                return winCriteria == WinOptions.WinWithTakeLast ? Resources.END_PLAYERWIN : Resources.END_PLAYERLOSE;
            }
```
这段代码是一个条件语句，根据winCriteria的值来决定返回不同的字符串。如果winCriteria等于WinOptions.WinWithTakeLast，则返回Resources.END_PLAYERWIN，否则返回Resources.END_PLAYERLOSE。

```
            currentPlayer = Players.Computer;
            return "";
        }
```
这段代码将currentPlayer设置为Players.Computer，然后返回一个空字符串。

```
        private string ComputerTurn()
        {
            //first calculate the move to play
            int sumTake = minTake + maxTake;
            int draw = pileSize - sumTake * (int)(pileSize / (float)sumTake);
            draw = Math.Clamp(draw, minTake, maxTake);
```
这段代码首先计算要进行的移动。它计算了sumTake的值，然后计算了draw的值，并使用Math.Clamp方法将其限制在minTake和maxTake之间。

```
            //detect win/lose conditions
            switch (winCriteria)
            {
                case WinOptions.WinWithAvoidLast when (pileSize == minTake): //lose condition
                    pileSize = 0;
                    return string.Format(Resources.END_COMPLOSE, minTake);
```
这段代码使用switch语句来检测赢或输的条件。在这个特定的case中，当winCriteria等于WinOptions.WinWithAvoidLast并且pileSize等于minTake时，将pileSize设置为0，并返回一个格式化的字符串。
                case WinOptions.WinWithAvoidLast when (pileSize <= maxTake): // 当堆大小小于等于最大取走数量时，避免在下一轮自动输掉游戏
                    draw = Math.Clamp(draw, minTake, pileSize - 1); // 限制取走的数量在最小取走数量和堆大小减一之间
                    break;
                case WinOptions.WinWithTakeLast when pileSize <= maxTake: // 当堆大小小于等于最大取走数量时，满足获胜条件
                    draw = Math.Min(pileSize, maxTake); // 取走的数量为堆大小和最大取走数量中的较小值
                    pileSize = 0; // 堆大小设为零
                    return string.Format(Resources.END_COMPWIN, draw); // 返回电脑获胜的消息
            }
            pileSize -= draw; // 堆大小减去取走的数量
            currentPlayer = Players.Human; // 当前玩家设为人类玩家
            return string.Format(Resources.COMPTURN, draw, pileSize); // 返回电脑取走的数量和剩余堆大小的消息
        }
    }
}
```