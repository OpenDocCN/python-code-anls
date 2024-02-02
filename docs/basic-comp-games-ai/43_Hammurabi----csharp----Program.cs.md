# `basic-computer-games\43_Hammurabi\csharp\Program.cs`

```py
// 引入必要的命名空间
using System;
using System.Collections.Immutable;

// 定义 Hammurabi 类
namespace Hammurabi
{
    // 定义 Program 类
    public static class Program
    {
        // 定义游戏长度常量
        public const int GameLength = 10;

        // 定义程序入口点
        public static void Main(string[] args)
        {
            // 创建随机数生成器
            var random  = new Random() ;
            // 初始化游戏状态
            var state   = Rules.BeginGame();
            // 创建空的游戏状态历史记录
            var history = ImmutableList<GameState>.Empty;

            // 显示游戏横幅
            View.ShowBanner();

            try
            {
                // 当玩家未被弹劾时循环执行游戏
                while (!state.IsPlayerImpeached)
                {
                    // 开始新的回合
                    state = Rules.BeginTurn(state, random);
                    // 显示城市摘要
                    View.ShowCitySummary(state);

                    // 如果游戏年限超过设定的长度，则跳出循环
                    if (state.Year > GameLength)
                        break;

                    // 显示土地价格
                    View.ShowLandPrice(state);
                    // 更新游戏状态，提示购买土地
                    var newState = Controller.UpdateGameState(state, View.PromptBuyLand, Rules.BuyLand);
                    // 如果购买了土地，则更新游戏状态，否则提示出售土地
                    state = newState.Acres != state.Acres ?
                        newState : Controller.UpdateGameState(state, View.PromptSellLand, Rules.SellLand);

                    // 显示分隔线
                    View.ShowSeparator();
                    // 更新游戏状态，提示喂养人口
                    state = Controller.UpdateGameState(state, View.PromptFeedPeople, Rules.FeedPeople);

                    // 显示分隔线
                    View.ShowSeparator();
                    // 更新游戏状态，提示种植庄稼
                    state = Controller.UpdateGameState(state, View.PromptPlantCrops, Rules.PlantCrops);

                    // 结束回合
                    state = Rules.EndTurn(state, random);
                    // 将当前游戏状态添加到历史记录中
                    history = history.Add(state);
                }

                // 获取游戏结果
                var result = Rules.GetGameResult(history, random);
                // 显示游戏结果
                View.ShowGameResult(result);
            }
            // 捕获 GreatOffence 异常
            catch (GreatOffence)
            {
                // 显示 GreatOffence 异常信息
                View.ShowGreatOffence();
            }

            // 显示游戏结束信息
            View.ShowFarewell();
        }
    }
}
```