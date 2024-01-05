# `53_King\csharp\Year.cs`

```
using System.Text;  # 导入 System.Text 模块

namespace King;  # 定义命名空间 King

internal class Year  # 定义内部类 Year
{
    private readonly Country _country;  # 声明私有只读字段 _country，类型为 Country
    private readonly IRandom _random;  # 声明私有只读字段 _random，类型为 IRandom
    private readonly IReadWrite _io;  # 声明私有只读字段 _io，类型为 IReadWrite
    private readonly int _plantingCost;  # 声明私有只读字段 _plantingCost，类型为 int
    private readonly int _landValue;  # 声明私有只读字段 _landValue，类型为 int

    private float _landSold;  # 声明私有字段 _landSold，类型为 float
    private float _rallodsDistributed;  # 声明私有字段 _rallodsDistributed，类型为 float
    private float _landPlanted;  # 声明私有字段 _landPlanted，类型为 float
    private float _pollutionControlCost;  # 声明私有字段 _pollutionControlCost，类型为 float

    private float _citizenSupport;  # 声明私有字段 _citizenSupport，类型为 float
    private int _deaths;  # 声明私有字段 _deaths，类型为 int
    private float _starvationDeaths;  # 声明私有字段 _starvationDeaths，类型为 float
```
```python
    private int _pollutionDeaths; // 声明一个私有整型变量 _pollutionDeaths，用于存储污染导致的死亡人数
    private int _migration; // 声明一个私有整型变量 _migration，用于存储迁移人口数量

    public Year(Country country, IRandom random, IReadWrite io)
    {
        _country = country; // 将传入的 country 参数赋值给私有变量 _country
        _random = random; // 将传入的 random 参数赋值给私有变量 _random
        _io = io; // 将传入的 io 参数赋值给私有变量 _io
        
        _plantingCost = random.Next(10, 15); // 使用 random 对象生成一个介于 10 到 15 之间的随机数，赋值给 _plantingCost
        _landValue = random.Next(95, 105); // 使用 random 对象生成一个介于 95 到 105 之间的随机数，赋值给 _landValue
    }

    public string Status => _country.GetStatus(_landValue, _plantingCost); // 使用 _country 对象的 GetStatus 方法获取国家的状态信息，并返回该信息

    public Result? GetPlayerActions()
    {
        var playerSoldLand = _country.SellLand(_landValue, out _landSold); // 调用 _country 对象的 SellLand 方法，将 _landValue 作为参数，将结果赋值给 playerSoldLand，并将卖出的土地数量赋值给 _landSold
        var playerDistributedRallods = _country.DistributeRallods(out _rallodsDistributed); // 调用 _country 对象的 DistributeRallods 方法，将结果赋值给 playerDistributedRallods，并将分发的粮食数量赋值给 _rallodsDistributed
        var playerPlantedLand = _country.HasRallods && _country.PlantLand(_plantingCost, out _landPlanted); // 检查 _country 对象是否有足够的粮食，如果有则调用 _country 对象的 PlantLand 方法，将 _plantingCost 作为参数，将结果赋值给 playerPlantedLand，并将种植的土地数量赋值给 _landPlanted
        // 检查玩家是否控制了污染，如果是，则获取控制污染的成本
        var playerControlledPollution = _country.HasRallods && _country.ControlPollution(out _pollutionControlCost);

        // 如果玩家卖出土地、分配资源、种植土地或控制了污染，则返回空，否则返回游戏结束
        return playerSoldLand || playerDistributedRallods || playerPlantedLand || playerControlledPollution
            ? null
            : Result.GameOver(Goodbye);
    }

    public Result? EvaluateResults()
    {
        // 获取国家未使用的资源
        var rallodsUnspent = _country.Rallods;

        // 执行评估死亡人数、移民、农业、旅游，并确定最终结果
        return EvaluateDeaths() 
            ?? EvaluateMigration() 
            ?? EvaluateAgriculture()
            ?? EvaluateTourism()
            ?? DetermineResult(rallodsUnspent);
    }
    # 计算饥饿死亡人数
    public Result? EvaluateDeaths()
    {
        # 计算受支持的国民数量
        var supportedCountrymen = _rallodsDistributed / 100;
        # 计算受支持的国民数量与国家总人口的差值
        _citizenSupport = supportedCountrymen - _country.Countrymen;
        # 计算因饥饿导致的死亡人数
        _starvationDeaths = -_citizenSupport;
        # 如果有饥饿死亡人数
        if (_starvationDeaths > 0)
        {
            # 如果受支持的国民数量小于50，则游戏结束
            if (supportedCountrymen < 50) { return Result.GameOver(EndOneThirdDead(_random)); }
            # 输出饥饿死亡人数
            _io.WriteLine(DeathsStarvation(_starvationDeaths));
        }

        # 计算污染导致的死亡人数
        var pollutionControl = _pollutionControlCost >= 25 ? _pollutionControlCost / 25 : 1;
        _pollutionDeaths = (int)(_random.Next((int)_country.IndustryLand) / pollutionControl);
        # 如果有污染死亡人数
        if (_pollutionDeaths > 0)
        {
            # 输出污染死亡人数
            _io.WriteLine(DeathsPollution(_pollutionDeaths));
        }

        # 计算总死亡人数
        _deaths = (int)(_starvationDeaths + _pollutionDeaths);
        if (_deaths > 0)  # 如果死亡人数大于0
        {
            var funeralCosts = _deaths * 9;  # 计算葬礼费用
            _io.WriteLine(FuneralExpenses(funeralCosts));  # 输出葬礼费用

            if (!_country.TrySpend(funeralCosts, _landValue))  # 如果国家尝试花费葬礼费用
            {
                _io.WriteLine(InsufficientReserves);  # 输出储备不足
            }

            _country.RemoveTheDead(_deaths);  # 移除死者
        }

        return null;  # 返回空值
    }

    private Result? EvaluateMigration()  # 评估迁移
    {
        if (_landSold > 0)  # 如果卖出土地大于0
        {
            // 计算新的工人数量，包括已售出的土地和随机波动
            var newWorkers = (int)(_landSold + _random.NextFloat(10) - _random.NextFloat(20));
            // 如果国家没有工人，则新增工人数量再加上20
            if (!_country.HasWorkers) { newWorkers += 20; }
            // 输出工人迁移情况
            _io.Write(WorkerMigration(newWorkers));
            // 向国家添加新的工人
            _country.AddWorkers(newWorkers);
        }

        // 计算迁移人口数量，包括公民支持度、污染控制成本、工业用地和污染死亡人数的影响
        _migration = 
            (int)(_citizenSupport / 10 + _pollutionControlCost / 25 - _country.IndustryLand / 50 - _pollutionDeaths / 2);
        // 输出迁移人口情况
        _io.WriteLine(Migration(_migration));
        // 更新国家的迁移人口数量
        _country.Migration(_migration);

        // 返回空结果
        return null;
    }

    // 评估农业情况
    private Result? EvaluateAgriculture()
    {
        // 计算受损作物数量，取工业用地和种植土地的最小值
        var ruinedCrops = (int)Math.Min(_country.IndustryLand * (_random.NextFloat() + 1.5f) / 2, _landPlanted);
        // 计算农作物产量，减去受损作物数量
        var yield = (int)(_landPlanted - ruinedCrops);
        // 计算农业收入，取农作物产量乘以土地价值的一半
        var income = (int)(yield * _landValue / 2f);
        _io.Write(LandPlanted(_landPlanted)); // 将 _landPlanted 的值写入输出流中
        _io.Write(Harvest(yield, income, _country.IndustryLand > 0)); // 将收获的产量、收入和国家工业用地是否大于0的布尔值写入输出流中

        _country.SellCrops(income); // 将收入作为参数调用 _country 对象的 SellCrops 方法

        return null; // 返回空值
    }

    private Result? EvaluateTourism() // 定义一个返回 Result 类型或空值的私有方法 EvaluateTourism
    {
        var reputationValue = (int)((_country.Countrymen - _migration) * 22 + _random.NextFloat(500)); // 计算声誉值
        var industryAdjustment = (int)(_country.IndustryLand * 15); // 计算工业调整值
        var tourismIncome = Math.Abs(reputationValue - industryAdjustment); // 计算旅游收入

        _io.WriteLine(TourismEarnings(tourismIncome)); // 将旅游收入传入 TourismEarnings 方法并将结果写入输出流中
        if (industryAdjustment > 0 && tourismIncome < _country.PreviousTourismIncome) // 如果工业调整值大于0且旅游收入小于国家之前的旅游收入
        {
            _io.Write(TourismDecrease(_random)); // 调用 TourismDecrease 方法并将随机数传入输出流中
        }
        _country.EntertainTourists(tourismIncome);  # 调用 _country 对象的 EntertainTourists 方法，传入 tourismIncome 参数

        return null;  # 返回空值
    }

    private Result? DetermineResult(float rallodsUnspent)  # 定义一个私有方法 DetermineResult，接受一个浮点数参数 rallodsUnspent
    {
        if (_deaths > 200) { return Result.GameOver(EndManyDead(_deaths, _random)); }  # 如果 _deaths 大于 200，则返回一个 Result 对象，调用 EndManyDead 方法
        if (_country.Countrymen < 343) { return Result.GameOver(EndOneThirdDead(_random)); }  # 如果 _country.Countrymen 小于 343，则返回一个 Result 对象，调用 EndOneThirdDead 方法
        if (rallodsUnspent / 100 > 5 && _starvationDeaths >= 2) { return Result.GameOver(EndMoneyLeftOver()); }  # 如果 rallodsUnspent 除以 100 大于 5 并且 _starvationDeaths 大于等于 2，则返回一个 Result 对象，调用 EndMoneyLeftOver 方法
        if (_country.Workers > _country.Countrymen) { return Result.GameOver(EndForeignWorkers(_random)); }  # 如果 _country.Workers 大于 _country.Countrymen，则返回一个 Result 对象，调用 EndForeignWorkers 方法
        return null;  # 返回空值
    }
}
```