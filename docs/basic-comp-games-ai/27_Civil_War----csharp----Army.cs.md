# `27_Civil_War\csharp\Army.cs`

```
# 引入所需的模块
import zipfile
from io import BytesIO

# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
        public Side Side { get; } // 定义一个公共属性，表示一方的身份

        // Cumulative
        public int Wins { get; private set; } // W, L // 定义一个公共属性，表示累计胜利次数
        public int Losses { get; private set; } // L, W // 定义一个公共属性，表示累计失败次数
        public int Draws { get; private set; } // W0 // 定义一个公共属性，表示平局次数
        public int BattlesFought => Wins + Draws + Losses; // 定义一个只读属性，表示累计战斗次数
        public bool Surrendered { get; private set; } // Y, Y2 == 5 // 定义一个公共属性，表示是否投降

        public int CumulativeHistoricCasualties { get; private set; } // P1, P2 // 定义一个公共属性，表示累计历史伤亡人数
        public int CumulativeSimulatedCasualties { get; private set; } // T1, T2 // 定义一个公共属性，表示累计模拟伤亡人数
        public int CumulativeHistoricMen { get; private set; } // M3, M4 // 定义一个公共属性，表示累计历史人数

        private int income; // R1, R2 // 定义一个私有属性，表示收入
        private int moneySpent; // Q1, Q2 // 定义一个私有属性，表示花费的金钱

        private bool IsFirstBattle => income == 0; // 定义一个私有属性，表示是否是第一次战斗

        // This battle
        private int historicMen; // M1, M2 // 定义一个私有属性，表示本次战斗的历史人数
        public int HistoricCasualties { get; private set; } // 历史伤亡人数

        public int Money { get; private set; } // 资金
        public int Men { get; private set; } // 士兵数量
        public int Inflation { get; private set; } // 通货膨胀
        public int InflationDisplay => Side == Side.Confederate ? Inflation + 15 : Inflation; // 显示通货膨胀，南方联盟的通货膨胀显示时加上15，原因不明！

        private readonly Dictionary<Resource, int> allocations = new(); // 分配资源的字典，包括食物、工资、弹药

        public int Strategy { get; protected set; } // 战略指数

        public int Casualties { get; protected set; } // 伤亡人数
        public int Desertions { get; protected set; } // 叛逃人数
        public int MenLost => Casualties + Desertions; // 丢失的士兵数量
        public bool AllLost { get; private set; } // 是否全部丢失

        private double reducedAvailableMen; // 减少的可用士兵数量

        protected virtual double FractionUnspent => (income - moneySpent) / (income + 1.0); // 未花费的比例
        // 准备战斗，设置历史人数和伤亡人数
        public void PrepareBattle(int men, int casualties)
        {
            historicMen = men; // 设置历史人数
            HistoricCasualties = casualties; // 设置历史伤亡人数
            Inflation = 10 + (Losses - Wins) * 2; // 根据损失和胜利计算通货膨胀
            Money = 100 * (int)(men * (100 - Inflation) / 2000.0 * (1 + FractionUnspent) + 0.5); // 计算可用资金
            Men = (int)(men * 1 + (CumulativeHistoricCasualties - CumulativeSimulatedCasualties) / (CumulativeHistoricMen + 1.0)); // 计算可用人数
            reducedAvailableMen = men * 5.0 / 6.0; // 计算减少后的可用人数
        }

        // 分配资源
        public virtual void AllocateResources()
        {
            Console.WriteLine($"{Side} General ---\nHow much do you wish to spend for"); // 输出提示信息
            while (true) // 进入循环，直到条件满足退出
            {
                foreach (Resource resource in Enum.GetValues<Resource>()) // 遍历资源类型
                {
                    if (EnterResource(resource)) // 调用EnterResource方法，如果返回true则跳出循环
                        break;
                }
                if (allocations.Values.Sum() <= Money)
                    return;  # 如果分配的资源总和小于等于拥有的钱，则返回
                Console.WriteLine($"Think again! You have only ${Money}");  # 打印拥有的钱的金额
            }
        }

        private bool EnterResource(Resource resource)
        {
            while (true)
            {
                Console.WriteLine($" - {resource}");  # 打印资源的名称
                switch ((int.TryParse(Console.ReadLine(), out int val), val))  # 尝试将用户输入的内容转换为整数，并将结果存储在val中
                {
                    case (false, _):  # 如果转换失败
                        Console.WriteLine("Not a valid number");  # 打印错误消息
                        break;
                    case (_, < 0):  # 如果转换成功但值小于0
                        Console.WriteLine("Negative values not allowed");  # 打印错误消息
                        break;
                    case (_, 0) when IsFirstBattle:  # 当前情况下，如果是第一场战斗且没有先前的记录
                        Console.WriteLine("No previous entries");  # 输出提示信息
                        break;  # 跳出当前循环
                    case (_, 0):  # 当前情况下，如果没有先前的记录
                        Console.WriteLine("Assume you want to keep same allocations");  # 输出提示信息
                        return true;  # 返回 true
                    case (_, > 0):  # 当前情况下，如果有先前的记录
                        allocations[resource] = val;  # 将资源分配值更新到字典中
                        return false;  # 返回 false
                }
            }
        }

        public virtual void DisplayMorale()  # 定义虚拟方法 DisplayMorale
        {
            Console.WriteLine($"{Side} morale is {Morale switch { < 5 => "Poor", < 10 => "Fair", _ => "High" }}");  # 输出阵营士气情况
        }

        public virtual bool ChooseStrategy(bool isReplay) => EnterStrategy(true, "(1-5)");  # 定义虚拟方法 ChooseStrategy，调用 EnterStrategy 方法并返回结果
        # 定义一个名为 EnterStrategy 的方法，接受两个参数：canSurrender 和 hint
        protected bool EnterStrategy(bool canSurrender, string hint)
        {
            # 打印出当前策略的提示信息
            Console.WriteLine($"{Side} strategy {hint}");
            # 进入一个无限循环
            while (true)
            {
                # 从控制台读取用户输入的值，并尝试将其转换为整数
                switch ((int.TryParse(Console.ReadLine(), out int val), val))
                {
                    # 如果转换失败，则打印出错误信息
                    case (false, _):
                        Console.WriteLine("Not a valid number");
                        break;
                    # 如果用户输入的值为5，并且可以投降，则设置 Surrendered 为 true，并打印出投降的信息，然后返回 true
                    case (_, 5) when canSurrender:
                        Surrendered = true;
                        Console.WriteLine($"The {Side} general has surrendered");
                        return true;
                    # 如果用户输入的值小于1或者大于等于5，则打印出不允许该策略的信息
                    case (_, < 1 or >= 5):
                        Console.WriteLine($"Strategy {val} not allowed.");
                        break;
                    # 其他情况下，设置 Strategy 为用户输入的值，并返回 false
                    default:
                        Strategy = val;
                        return false;
        public virtual void CalculateLosses(Army opponent)
        {
            // 设置标志位为 false
            AllLost = false;
            // 计算战略因素
            int stratFactor = 2 * (Math.Abs(Strategy - opponent.Strategy) + 1);
            // 计算伤亡人数
            Casualties = (int)Math.Round(HistoricCasualties * 0.4 * (1 + 1.0 / stratFactor) * (1 + 1 / Morale) * (1.28 + reducedAvailableMen / (allocations[Resource.Ammunition] + 1)));
            // 计算叛逃人数
            Desertions = (int)Math.Round(100 / Morale);

            // 如果伤亡人数大于当前人数，重新计算伤亡人数和叛逃人数
            if (MenLost > Men)
            {
                Casualties = 13 * Men / 20;
                Desertions = Men - Casualties;
                // 设置标志位为 true
                AllLost = true;
            }
        }
# 记录比赛结果的方法，根据获胜方记录胜利次数，平局次数和失败次数
public void RecordResult(Side winner)
{
    # 如果获胜方等于当前方，胜利次数加一
    if (winner == Side)
        Wins++;
    # 如果获胜方是双方都是，平局次数加一
    else if (winner == Side.Both)
        Draws++;
    # 否则，失败次数加一
    else
        Losses++;

    # 累计模拟伤亡人数加上当前战斗中损失的人数
    CumulativeSimulatedCasualties += MenLost;
    # 累计历史伤亡人数加上历史伤亡人数
    CumulativeHistoricCasualties += HistoricCasualties;
    # 花费的金钱加上分配值的总和
    moneySpent += allocations.Values.Sum();
    # 收入加上历史人数乘以（100-通货膨胀率）除以20
    income += historicMen * (100 - Inflation) / 20;
    # 累计历史人数加上历史人数
    CumulativeHistoricMen += historicMen;

    # 学习战略
    LearnStrategy();
}

# 学习战略的虚拟方法
protected virtual void LearnStrategy() { }
        public void DisplayWarResult(Army opponent)
        {
            // 在控制台打印空行
            Console.WriteLine("\n\n\n\n");
            // 打印当前将军的胜利和失败次数
            Console.WriteLine($"The {Side} general has won {Wins} battles and lost {Losses}");
            // 根据条件判断胜利方，并赋值给变量winner
            Side winner = (Surrendered, opponent.Surrendered, Wins < Losses) switch
            {
                (_, true, _) => Side,  // 如果对手投降，则当前将军获胜
                (true, _, _) or (_, _, true) => opponent.Side,  // 如果当前将军投降或者战败次数多于胜利次数，则对手获胜
                _ => Side  // 其他情况当前将军获胜
            };
            // 打印战争结果
            Console.WriteLine($"The {winner} general has won the war\n");
        }

        // 虚方法，用于展示战略
        public virtual void DisplayStrategies() { }
    }

    class ComputerArmy : Army
    {
        // 电脑军队的战略概率数组
        public int[] StrategyProb { get; } = { 25, 25, 25, 25 }; // S(n)
        // 用于生成随机数的实例
        private readonly Random strategyRng = new();
        public ComputerArmy(Side side) : base(side) { }  // 创建一个名为ComputerArmy的公共类，继承自基类Side

        protected override double FractionUnspent => 0.0;  // 重写基类的FractionUnspent属性，将其值设为0.0

        public override void AllocateResources() { }  // 重写基类的AllocateResources方法

        public override void DisplayMorale() { }  // 重写基类的DisplayMorale方法

        public override bool ChooseStrategy(bool isReplay)  // 重写基类的ChooseStrategy方法，接受一个布尔类型的参数isReplay
        {
            if (isReplay)  // 如果isReplay为真
                return EnterStrategy(false, $"(1-4; usually previous {Side} strategy)");  // 调用EnterStrategy方法并返回结果

            // Basic code comments say "If actual strategy info is in  data then r-100 is extra weight given to that strategy" but there's no data or code to do it.
            int strategyChosenProb = strategyRng.Next(100); // 0-99  // 生成一个0到99之间的随机数
            int sumProbs = 0;  // 初始化sumProbs为0
            for (int i = 0; i < 4; i++)  // 循环4次
            {
                sumProbs += StrategyProb[i];  // 将StrategyProb数组中第i个元素的值加到sumProbs上
                if (strategyChosenProb < sumProbs)
                {
                    Strategy = i + 1;  # 如果随机选择的概率小于总概率，则选择当前策略
                    break;  # 跳出循环
                }
            }
            Console.WriteLine($"{Side} strategy is {Strategy}");  # 打印当前策略
            return false;  # 返回 false
        }

        protected override void LearnStrategy()
        {
            // 学习当前策略，开始遗忘旧的策略
            // - 当前策略增加 3 * s 的概率点，其他策略失去 s 的概率点，除非某个策略下降到 5% 以下。
            const int s = 3;  # 定义常量 s 为 3
            int presentGain = 0;  # 初始化当前增益为 0
            for (int i = 0; i < 4; i++)  # 循环 4 次
            {
                if (StrategyProb[i] >= 5)  # 如果策略概率大于等于 5
                {
                    StrategyProb[i] -= s;  // 从策略概率数组中减去指定索引位置的值
                    presentGain += s;  // 将指定索引位置的值加到当前收益中
                }
            }
            StrategyProb[Strategy - 1] += presentGain;  // 将当前收益加到策略概率数组中指定索引位置的值上
        }

        public override void CalculateLosses(Army opponent)
        {
            Casualties = (int)(17.0 * HistoricCasualties * opponent.HistoricCasualties / (opponent.Casualties * 20));  // 计算损失人数
            Desertions = (int)(5 * opponent.Morale);  // 计算叛逃人数
        }

        public override void DisplayStrategies()
        {
            ConsoleUtils.WriteWordWrap($"\nIntelligence suggests that the {Side} general used strategies 1, 2, 3, 4 in the following percentages:");  // 显示情报提示
            Console.WriteLine(string.Join(", ", StrategyProb));  // 在控制台上显示策略概率数组的值
        }
    }
}
bio = BytesIO(open(fname, 'rb').read())
```
这行代码创建了一个字节流对象`bio`，并将以二进制形式读取的文件内容封装到这个字节流中。

```python
zip = zipfile.ZipFile(bio, 'r')
```
这行代码使用字节流里面的内容创建了一个ZIP对象`zip`，并指定了以只读模式打开。

```python
fdict = {n:zip.read(n) for n in zip.namelist()}
```
这行代码遍历ZIP对象中所包含的文件名，然后使用`zip.read(n)`读取文件数据，并将文件名和数据组成字典`fdict`。

```python
zip.close()
```
这行代码关闭了ZIP对象，释放了相关资源。

```python
return fdict
```
这行代码返回了结果字典`fdict`。
```