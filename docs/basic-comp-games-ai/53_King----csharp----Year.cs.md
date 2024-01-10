# `basic-computer-games\53_King\csharp\Year.cs`

```
using System.Text;

namespace King;
// 定义了一个名为 King 的命名空间

internal class Year
{
    private readonly Country _country;
    private readonly IRandom _random;
    private readonly IReadWrite _io;
    private readonly int _plantingCost;
    private readonly int _landValue;
    // 定义了 Year 类，包含了一些私有字段和构造函数

    private float _landSold;
    private float _rallodsDistributed;
    private float _landPlanted;
    private float _pollutionControlCost;
    // 定义了一些私有字段，用于存储一些数值

    private float _citizenSupport;
    private int _deaths;
    private float _starvationDeaths;
    private int _pollutionDeaths;
    private int _migration;
    // 定义了一些私有字段，用于存储一些数值

    public Year(Country country, IRandom random, IReadWrite io)
    {
        _country = country;
        _random = random;
        _io = io;
        // Year 类的构造函数，接受 Country、IRandom 和 IReadWrite 三个参数，并初始化相应的字段
        
        _plantingCost = random.Next(10, 15);
        _landValue = random.Next(95, 105);
        // 使用 random 对象生成种植成本和土地价值的随机数，并赋值给相应的字段
    }

    public string Status => _country.GetStatus(_landValue, _plantingCost);
    // 定义了一个只读属性 Status，返回调用 _country.GetStatus 方法的结果

    public Result? GetPlayerActions()
    {
        var playerSoldLand = _country.SellLand(_landValue, out _landSold);
        var playerDistributedRallods = _country.DistributeRallods(out _rallodsDistributed);
        var playerPlantedLand = _country.HasRallods && _country.PlantLand(_plantingCost, out _landPlanted);
        var playerControlledPollution = _country.HasRallods && _country.ControlPollution(out _pollutionControlCost);
        // 调用 _country 对象的方法，获取玩家的行动结果

        return playerSoldLand || playerDistributedRallods || playerPlantedLand || playerControlledPollution
            ? null
            : Result.GameOver(Goodbye);
        // 如果玩家有任何一种行动结果为真，则返回 null，否则返回游戏结束的结果
    }

    public Result? EvaluateResults()
    {
        var rallodsUnspent = _country.Rallods;

        _io.WriteLine();
        _io.WriteLine();
        // 输出空行

        return EvaluateDeaths() 
            ?? EvaluateMigration() 
            ?? EvaluateAgriculture()
            ?? EvaluateTourism()
            ?? DetermineResult(rallodsUnspent);
        // 调用一系列评估方法，根据结果返回相应的结果
    }

    public Result? EvaluateDeaths()
    // 定义了一个评估死亡情况的方法
    {
        // 计算受支持的国民数量
        var supportedCountrymen = _rallodsDistributed / 100;
        // 计算支持国民与实际国民数量之差
        _citizenSupport = supportedCountrymen - _country.Countrymen;
        // 计算因饥饿导致的死亡人数
        _starvationDeaths = -_citizenSupport;
        // 如果有饥饿死亡人数，则进行相应处理
        if (_starvationDeaths > 0)
        {
            // 如果受支持的国民数量小于50，则游戏结束
            if (supportedCountrymen < 50) { return Result.GameOver(EndOneThirdDead(_random)); }
            // 输出饥饿死亡人数
            _io.WriteLine(DeathsStarvation(_starvationDeaths));
        }
    
        // 计算污染控制对死亡人数的影响
        var pollutionControl = _pollutionControlCost >= 25 ? _pollutionControlCost / 25 : 1;
        _pollutionDeaths = (int)(_random.Next((int)_country.IndustryLand) / pollutionControl);
        // 如果有污染导致的死亡人数，则进行相应处理
        if (_pollutionDeaths > 0)
        {
            // 输出污染导致的死亡人数
            _io.WriteLine(DeathsPollution(_pollutionDeaths));
        }
    
        // 计算总死亡人数
        _deaths = (int)(_starvationDeaths + _pollutionDeaths);
        // 如果有死亡人数，则进行相应处理
        if (_deaths > 0)
        {
            // 计算葬礼费用
            var funeralCosts = _deaths * 9;
            // 输出葬礼费用
            _io.WriteLine(FuneralExpenses(funeralCosts));
            // 如果国家财政无法支付葬礼费用，则输出相应信息
            if (!_country.TrySpend(funeralCosts, _landValue))
            {
                _io.WriteLine(InsufficientReserves);
            }
            // 移除死亡人数
            _country.RemoveTheDead(_deaths);
        }
    
        // 返回空结果
        return null;
    }
    
    // 评估移民情况
    private Result? EvaluateMigration()
    {
        // 如果有土地出售，则计算新工人数量
        if (_landSold > 0)
        {
            var newWorkers = (int)(_landSold + _random.NextFloat(10) - _random.NextFloat(20));
            // 如果国家没有工人，则增加新工人数量
            if (!_country.HasWorkers) { newWorkers += 20; }
            // 输出工人迁移信息
            _io.Write(WorkerMigration(newWorkers));
            // 增加新工人数量
            _country.AddWorkers(newWorkers);
        }
    
        // 计算移民数量
        _migration = (int)(_citizenSupport / 10 + _pollutionControlCost / 25 - _country.IndustryLand / 50 - _pollutionDeaths / 2);
        // 输出移民信息
        _io.WriteLine(Migration(_migration));
        // 进行移民处理
        _country.Migration(_migration);
    
        // 返回空结果
        return null;
    }
    
    // 评估农业情况
    {
        // 计算受损庄稼数量，取工业用地和随机数的最小值，再除以2，再取与种植的土地数量的最小值
        var ruinedCrops = (int)Math.Min(_country.IndustryLand * (_random.NextFloat() + 1.5f) / 2, _landPlanted);
        // 计算收成，为种植的土地数量减去受损庄稼数量
        var yield = (int)(_landPlanted - ruinedCrops);
        // 计算收入，为收成乘以土地价值的一半
        var income = (int)(yield * _landValue / 2f);

        // 输出种植的土地数量
        _io.Write(LandPlanted(_landPlanted));
        // 输出收成和收入，以及是否有工业用地
        _io.Write(Harvest(yield, income, _country.IndustryLand > 0));

        // 出售庄稼，获得收入
        _country.SellCrops(income);

        // 返回空值
        return null;
    }

    // 评估旅游业
    private Result? EvaluateTourism()
    {
        // 计算声望值，为（国民数量减去迁移数量）乘以22，再加上随机数（范围在500以内）
        var reputationValue = (int)((_country.Countrymen - _migration) * 22 + _random.NextFloat(500));
        // 计算工业调整值，为工业用地数量乘以15
        var industryAdjustment = (int)(_country.IndustryLand * 15);
        // 计算旅游收入，为声望值和工业调整值的绝对值
        var tourismIncome = Math.Abs(reputationValue - industryAdjustment);

        // 输出旅游收入
        _io.WriteLine(TourismEarnings(tourismIncome));
        // 如果工业调整值大于0且旅游收入小于上次的旅游收入，则输出旅游减少
        if (industryAdjustment > 0 && tourismIncome < _country.PreviousTourismIncome)
        {
            _io.Write(TourismDecrease(_random));
        }

        // 娱乐游客，获得旅游收入
        _country.EntertainTourists(tourismIncome);

        // 返回空值
        return null;
    }

    // 确定结果
    private Result? DetermineResult(float rallodsUnspent)
    {
        // 如果死亡人数大于200，则返回游戏结束，输出死亡人数和随机数
        if (_deaths > 200) { return Result.GameOver(EndManyDead(_deaths, _random)); }
        // 如果国民数量小于343，则返回游戏结束，输出随机数
        if (_country.Countrymen < 343) { return Result.GameOver(EndOneThirdDead(_random)); }
        // 如果剩余的货币/100大于5且饥饿死亡人数大于等于2，则返回游戏结束，输出剩余的货币
        if (rallodsUnspent / 100 > 5 && _starvationDeaths >= 2) { return Result.GameOver(EndMoneyLeftOver()); }
        // 如果工人数量大于国民数量，则返回游戏结束，输出随机数
        if (_country.Workers > _country.Countrymen) { return Result.GameOver(EndForeignWorkers(_random)); }
        // 返回空值
        return null;
    }
# 闭合前面的函数定义
```