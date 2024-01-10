# `basic-computer-games\53_King\csharp\Country.cs`

```
namespace King;

internal class Country
{
    private const int InitialLand = 1000;  // 设置初始土地数量为1000

    private readonly IReadWrite _io;  // 读写接口
    private readonly IRandom _random;  // 随机数接口
    private float _rallods;  // 农产品数量
    private float _countrymen;  // 国民数量
    private float _foreigners;  // 外国人数量
    private float _arableLand;  // 可耕种土地数量
    private float _industryLand;  // 工业用地数量

    public Country(IReadWrite io, IRandom random)  // 构造函数，接受读写接口和随机数接口
        : this(
            io,
            random,
            (int)(60000 + random.NextFloat(1000) - random.NextFloat(1000)),  // 初始化农产品数量
            (int)(500 + random.NextFloat(10) - random.NextFloat(10)),  // 初始化国民数量
            0,  // 初始化外国人数量
            InitialLand)  // 初始化土地数量
    {
    }

    public Country(IReadWrite io, IRandom random, float rallods, float countrymen, float foreigners, float land)  // 构造函数，接受读写接口、随机数接口、农产品数量、国民数量、外国人数量、土地数量
    {
        _io = io;  // 初始化读写接口
        _random = random;  // 初始化随机数接口
        _rallods = rallods;  // 初始化农产品数量
        _countrymen = countrymen;  // 初始化国民数量
        _foreigners = foreigners;  // 初始化外国人数量
        _arableLand = land;  // 初始化可耕种土地数量
    }

    public string GetStatus(int landValue, int plantingCost)  // 获取国家状态的方法，接受土地价值和种植成本
        => Resource.Status(_rallods, _countrymen, _foreigners, _arableLand, landValue, plantingCost);  // 调用Resource类的Status方法，传入农产品数量、国民数量、外国人数量、可耕种土地数量、土地价值和种植成本

    public float Countrymen => _countrymen;  // 返回国民数量
    public float Workers => _foreigners;  // 返回外国人数量
    public bool HasWorkers => _foreigners > 0;  // 判断是否有外国人
    private float FarmLand => _arableLand;  // 返回可耕种土地数量
    public bool HasRallods => _rallods > 0;  // 判断是否有农产品
    public float Rallods => _rallods;  // 返回农产品数量
    public float IndustryLand => InitialLand - _arableLand;  // 返回工业用地数量
    public int PreviousTourismIncome { get; private set; }  // 上次旅游收入

    public bool SellLand(int landValue, out float landSold)  // 出售土地的方法，接受土地价值和出售的土地数量
    {
        if (_io.TryReadValue(
                SellLandPrompt,  // 出售土地的提示
                out landSold,  // 出售的土地数量
                new ValidityTest(v => v <= FarmLand, () => SellLandError(FarmLand))))  // 验证出售土地数量是否合法
        {
            _arableLand = (int)(_arableLand - landSold);  // 更新可耕种土地数量
            _rallods = (int)(_rallods + landSold * landValue);  // 更新农产品数量
            return true;  // 返回出售成功
        }

        return false;  // 返回出售失败
    }

    public bool DistributeRallods(out float rallodsGiven)  // 分发农产品的方法，接受分发的农产品数量
    {
        // 尝试从输入输出对象中读取给定的 rallods 数量，如果成功则执行下面的代码
        if (_io.TryReadValue(
                GiveRallodsPrompt,
                out rallodsGiven, 
                new ValidityTest(v => v <= _rallods, () => GiveRallodsError(_rallods))))
        {
            // 减去已经给出的 rallods 数量
            _rallods = (int)(_rallods - rallodsGiven);
            // 返回 true 表示成功执行
            return true;
        }
        // 返回 false 表示执行失败
        return false;
    }
    
    // 种植土地的方法，返回是否成功以及种植的土地数量
    public bool PlantLand(int plantingCost, out float landPlanted)
    {
        // 尝试从输入输出对象中读取种植土地的数量，如果成功则执行下面的代码
        if (_io.TryReadValue(
                PlantLandPrompt, 
                out landPlanted, 
                new ValidityTest(v => v <= _countrymen * 2, PlantLandError1),
                new ValidityTest(v => v <= FarmLand, PlantLandError2(FarmLand)),
                new ValidityTest(v => v * plantingCost <= _rallods, PlantLandError3(_rallods))))
        {
            // 减去花费的 rallods 数量
            _rallods -= (int)(landPlanted * plantingCost);
            // 返回 true 表示成功执行
            return true;
        }
        // 返回 false 表示执行失败
        return false;
    }
    
    // 控制污染的方法，返回是否成功以及消耗的 rallods 数量
    public bool ControlPollution(out float rallodsSpent)
    {
        // 尝试从输入输出对象中读取消耗的 rallods 数量，如果成功则执行下面的代码
        if (_io.TryReadValue(
                PollutionPrompt,
                out rallodsSpent, 
                new ValidityTest(v => v <= _rallods, () => PollutionError(_rallods))))
        {
            // 减去消耗的 rallods 数量
            _rallods = (int)(_rallods - rallodsSpent);
            // 返回 true 表示成功执行
            return true;
        }
        // 返回 false 表示执行失败
        return false;
    }
    
    // 尝试花费 rallods 的方法，返回是否成功
    public bool TrySpend(float amount, float landValue)
    {
        // 如果当前 rallods 数量大于等于花费的数量
        if (_rallods >= amount)
        {
            // 减去花费的 rallods 数量
            _rallods -= amount;
            // 返回 true 表示成功执行
            return true;
        }
        // 如果当前 rallods 数量不足以支付花费
        _arableLand = (int)(_arableLand - (int)(amount - _rallods) / landValue);
        _rallods = 0;
        // 返回 false 表示执行失败
        return false;
    }
    
    // 移除死亡人口的方法
    public void RemoveTheDead(int deaths) => _countrymen = (int)(_countrymen - deaths);
    
    // 迁移人口的方法
    public void Migration(int migration) => _countrymen = (int)(_countrymen + migration);
    
    // 增加外来工人的方法
    public void AddWorkers(int newWorkers) => _foreigners = (int)(_foreigners + newWorkers);
    
    // 出售农作物的方法
    public void SellCrops(int income) => _rallods = (int)(_rallods + income);
    
    // 招待游客的方法
    public void EntertainTourists(int income)
    {
        # 将上一次的旅游收入赋值给变量PreviousTourismIncome
        PreviousTourismIncome = income;
        # 将总旅游收入_rallods与本次收入income相加，并转换为整数
        _rallods = (int)(_rallods + income);
    }
# 闭合前面的函数定义
```