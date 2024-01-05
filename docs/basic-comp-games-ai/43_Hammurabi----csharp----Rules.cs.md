# `43_Hammurabi\csharp\Rules.cs`

```
using System;  // 导入 System 命名空间
using System.Collections.Generic;  // 导入 System.Collections.Generic 命名空间
using System.Linq;  // 导入 System.Linq 命名空间

namespace Hammurabi  // 定义 Hammurabi 命名空间
{
    public static class Rules  // 定义名为 Rules 的静态类
    {
        /// <summary>
        /// 创建新游戏的初始状态。
        /// </summary>
        public static GameState BeginGame() =>  // 定义名为 BeginGame 的公共静态方法，返回类型为 GameState
            new GameState  // 创建 GameState 对象
            {
                Year                = 0,  // 设置 Year 属性为 0
                Population          = 95,  // 设置 Population 属性为 95
                PopulationIncrease  = 5,  // 设置 PopulationIncrease 属性为 5
                Starvation          = 0,  // 设置 Starvation 属性为 0
                Acres               = 1000,  // 设置 Acres 属性为 1000
                Stores              = 0,  // 设置 Stores 属性为 0
                AcresPlanted        = 1000,  // 设置初始种植面积为1000
                Productivity        = 3,     // 设置初始生产力为3
                Spoilage            = 200,   // 设置初始损耗为200
                IsPlagueYear        = false, // 设置是否为瘟疫年为假
                IsPlayerImpeached   = false  // 设置玩家是否被弹劾为假
            };

        /// <summary>
        /// Updates the game state to start a new turn.
        /// </summary>
        public static GameState BeginTurn(GameState state, Random random) =>
            state with
            {
                Year            = state.Year + 1,  // 将年份加1
                Population      = (state.Population + state.PopulationIncrease - state.Starvation) / (state.IsPlagueYear ? 2 : 1),  // 根据是否为瘟疫年更新人口数量
                LandPrice       = random.Next(10) + 17,  // 根据随机数更新土地价格
                Stores          = state.Stores + (state.AcresPlanted * state.Productivity) - state.Spoilage,  // 更新存储量
                AcresPlanted    = 0,  // 将种植面积重置为0
                FoodDistributed = 0   // 将分发食物数量重置为0
            };
/// <summary>
/// Attempts to purchase the given number of acres.
/// </summary>
/// <returns>
/// The updated game state and action result.
/// </returns>
public static (GameState newState, ActionResult result) BuyLand(GameState state, int amount)
{
    // 计算购买指定数量土地的总价格
    var price = state.LandPrice * amount;

    // 如果价格为负数，返回当前状态和攻击结果
    if (price < 0)
        return (state, ActionResult.Offense);
    // 如果价格大于当前存储量，返回当前状态和存储不足的结果
    else if (price > state.Stores)
        return (state, ActionResult.InsufficientStores);
    // 否则更新游戏状态，扣除相应存储量，返回新状态和成功的结果
    else
        return (state with { Acres = state.Acres + amount, Stores = state.Stores - price }, ActionResult.Success);
}
        /// <summary>
        /// Attempts to sell the given number of acres.
        /// </summary>
        /// <returns>
        /// The updated game state and action result.
        /// </returns>
        public static (GameState newState, ActionResult result) SellLand(GameState state, int amount)
        {
            // 计算出售土地的总价
            var price = state.LandPrice * amount;

            // 如果价格为负数，返回当前状态和“Offense”操作结果
            if (price < 0)
                return (state, ActionResult.Offense);
            else
            // 如果要出售的土地数量大于等于当前拥有的土地数量，返回当前状态和“InsufficientLand”操作结果
            if (amount >= state.Acres)
                return (state, ActionResult.InsufficientLand);
            else
            // 更新游戏状态，减少土地数量，增加金库数量，返回更新后的状态和“Success”操作结果
                return (state with { Acres = state.Acres - amount, Stores = state.Stores + price }, ActionResult.Success);
        }

        /// <summary>
        /// Attempts to feed the people the given number of buschels.
        /// </summary>
        /// <returns>
        /// The updated game state and action result.
        /// </returns>
        public static (GameState newState, ActionResult result) FeedPeople(GameState state, int amount)
        {
            // 检查输入的喂食数量是否小于0，如果是则返回当前状态和进攻结果
            if (amount < 0)
                return (state, ActionResult.Offense);
            else
            // 检查输入的喂食数量是否大于当前存储量，如果是则返回当前状态和存储不足结果
            if (amount > state.Stores)
                return (state, ActionResult.InsufficientStores);
            else
            // 更新游戏状态，减少存储量，增加分发食物量，并返回更新后的状态和成功结果
                return (state with { Stores = state.Stores - amount, FoodDistributed = state.FoodDistributed + amount }, ActionResult.Success);
        }

        /// <summary>
        /// Attempts to plant crops on the given number of acres.
        /// <returns>
        /// 返回更新后的游戏状态和行动结果。
        /// </returns>
        public static (GameState newState, ActionResult result) PlantCrops(GameState state, int amount)
        {
            // 计算所需的存储量
            var storesRequired = amount / 2;
            // 计算最大耕种土地面积
            var maxAcres       = state.Population * 10;

            // 如果种植数量小于0，则返回当前状态和进攻行动结果
            if (amount < 0)
                return (state, ActionResult.Offense);
            else
            // 如果种植数量大于当前土地面积，则返回当前状态和土地不足行动结果
            if (amount > state.Acres)
                return (state, ActionResult.InsufficientLand);
            else
            // 如果所需存储量大于当前存储量，则返回当前状态和存储不足行动结果
            if (storesRequired > state.Stores)
                return (state, ActionResult.InsufficientStores);
            else
            // 如果（当前已种植土地面积 + 种植数量）大于最大耕种土地面积，则返回当前状态和人口不足行动结果
            if ((state.AcresPlanted + amount) > maxAcres)
                return (state, ActionResult.InsufficientPopulation);
            else
                return (state with  # 返回更新后的游戏状态
                {
                    AcresPlanted = state.AcresPlanted + amount,  # 更新种植的农田面积
                    Stores       = state.Stores - storesRequired,  # 更新存储的粮食数量
                }, ActionResult.Success);  # 返回更新后的游戏状态和操作结果为成功
        }

        /// <summary>
        /// Ends the current turn and returns the updated game state.
        /// </summary>
        public static GameState EndTurn(GameState state, Random random)  # 结束当前回合并返回更新后的游戏状态
        {
            var productivity = random.Next(1, 6);  # 生成1到5之间的随机数作为产量
            var harvest = productivity * state.AcresPlanted;  # 计算收获量

            var spoilage = random.Next(1, 6) switch  # 根据随机数的不同情况计算损耗量
            {
                2 => state.Stores / 2,  # 当随机数为2时，粮食损耗为存储粮食的一半
                4 => state.Stores / 4,  # 当随机数为4时，粮食损耗为存储粮食的四分之一
                _ => 0  # 其他情况下，粮食损耗为0
            };

            // 计算人口增长
            var populationIncrease= (int)((double)random.Next(1, 6) * (20 * state.Acres + state.Stores + harvest - spoilage) / state.Population / 100 + 1);

            // 判断是否为瘟疫年
            var plagueYear = random.Next(20) < 3;

            // 计算人口是否得到充分喂养
            var peopleFed  = state.FoodDistributed / 20;
            var starvation = peopleFed < state.Population ? state.Population - peopleFed : 0;
            var impeached  = starvation > state.Population * 0.45;

            // 返回更新后的州对象
            return state with
            {
                Productivity       = productivity,
                Spoilage           = spoilage,
                PopulationIncrease = populationIncrease,
                Starvation         = starvation,
                IsPlagueYear       = plagueYear,
                IsPlayerImpeached  = impeached
            };
        }
                PerformanceRating.Outstanding;

            if (averageStarvationRate > 20 || totalStarvation > 1000)
            {
                rating = PerformanceRating.Terrible;
            }
            else if (acresPerPerson < 7)
            {
                rating = PerformanceRating.Poor;
            }
            else if (acresPerPerson > 10)
            {
                rating = PerformanceRating.Good;
            }

            return new GameResult
            {
                FinalState = finalState,
                Rating = rating
            };
        }
                (averageStarvationRate, acresPerPerson) switch  # 使用 switch 语句根据 averageStarvationRate 和 acresPerPerson 的值进行条件判断
                {
                    (> 33, _) => PerformanceRating.Disgraceful,  # 如果 averageStarvationRate 大于 33 或者 acresPerPerson 为任意值，则返回 PerformanceRating.Disgraceful
                    (_, < 7)  => PerformanceRating.Disgraceful,  # 如果 averageStarvationRate 为任意值或者 acresPerPerson 小于 7，则返回 PerformanceRating.Disgraceful
                    (> 10, _) => PerformanceRating.Bad,  # 如果 averageStarvationRate 大于 10 或者 acresPerPerson 为任意值，则返回 PerformanceRating.Bad
                    (_, < 9)  => PerformanceRating.Bad,  # 如果 averageStarvationRate 为任意值或者 acresPerPerson 小于 9，则返回 PerformanceRating.Bad
                    (> 3, _)  => PerformanceRating.Ok,  # 如果 averageStarvationRate 大于 3 或者 acresPerPerson 为任意值，则返回 PerformanceRating.Ok
                    (_, < 10) => PerformanceRating.Ok,  # 如果 averageStarvationRate 为任意值或者 acresPerPerson 小于 10，则返回 PerformanceRating.Ok
                    _         => PerformanceRating.Terrific  # 其他情况返回 PerformanceRating.Terrific
                };

            var assassins = rating == PerformanceRating.Ok ?  # 如果 rating 等于 PerformanceRating.Ok，则将 assassins 设置为一个介于 0 和 finalState.Population * 0.8 之间的随机数，否则设置为 0
                random.Next(0, (int)(finalState.Population * 0.8)) : 0;

            return new GameResult  # 返回一个新的 GameResult 对象
            {
                Rating                = rating,  # 将 rating 赋值给 GameResult 对象的 Rating 属性
                AcresPerPerson        = acresPerPerson,  # 将 acresPerPerson 赋值给 GameResult 对象的 AcresPerPerson 属性
                FinalStarvation       = finalState.Starvation,  # 将 finalState.Starvation 赋值给 GameResult 对象的 FinalStarvation 属性
                TotalStarvation       = totalStarvation,  # 将 totalStarvation 赋值给 GameResult 对象的 TotalStarvation 属性
                AverageStarvationRate = averageStarvationRate,  # 设置平均饥饿率
                Assassins             = assassins,  # 设置刺客
                WasPlayerImpeached    = finalState.IsPlayerImpeached  # 设置玩家是否被弹劾
            };
        }
    }
}
```