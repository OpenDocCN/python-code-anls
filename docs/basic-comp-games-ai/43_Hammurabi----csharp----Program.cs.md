# `43_Hammurabi\csharp\Program.cs`

```
# 导入必要的命名空间
using System;
using System.Collections.Immutable;

namespace Hammurabi
{
    # 创建一个静态类 Program
    public static class Program
    {
        # 定义常量 GameLength 并赋值为 10
        public const int GameLength = 10;

        # 定义 Main 方法，入参为字符串数组 args
        public static void Main(string[] args)
        {
            # 创建一个 Random 对象 random
            var random  = new Random() ;
            # 调用 BeginGame 方法初始化游戏状态并赋值给 state
            var state   = Rules.BeginGame();
            # 创建一个空的不可变列表 history 用于存储游戏状态历史记录
            var history = ImmutableList<GameState>.Empty;

            # 调用 View 类的 ShowBanner 方法显示游戏横幅
            View.ShowBanner();

            # 使用 while 循环，直到玩家被弹劾为止
            while (!state.IsPlayerImpeached)
# 开始一个新的回合，并显示城市摘要
state = Rules.BeginTurn(state, random);
View.ShowCitySummary(state);

# 如果游戏年数超过了规定的长度，就结束游戏
if (state.Year > GameLength)
    break;

# 显示土地价格
View.ShowLandPrice(state);

# 更新游戏状态，提示玩家购买土地，并根据规则进行处理
var newState = Controller.UpdateGameState(state, View.PromptBuyLand, Rules.BuyLand);
state = newState.Acres != state.Acres ?
    newState : Controller.UpdateGameState(state, View.PromptSellLand, Rules.SellLand);

# 显示分隔线
View.ShowSeparator();

# 更新游戏状态，提示玩家喂养人口，并根据规则进行处理
state = Controller.UpdateGameState(state, View.PromptFeedPeople, Rules.FeedPeople);

# 显示分隔线
View.ShowSeparator();

# 更新游戏状态，提示玩家种植作物，并根据规则进行处理
state = Controller.UpdateGameState(state, View.PromptPlantCrops, Rules.PlantCrops);

# 结束回合，并记录历史状态
state = Rules.EndTurn(state, random);
history = history.Add(state);
                }
                // 根据历史记录和随机数获取游戏结果
                var result = Rules.GetGameResult(history, random);
                // 显示游戏结果
                View.ShowGameResult(result);
            }
            // 捕获 GreatOffence 异常并显示对应信息
            catch (GreatOffence)
            {
                View.ShowGreatOffence();
            }
            // 显示道别信息
            View.ShowFarewell();
        }
    }
}
```