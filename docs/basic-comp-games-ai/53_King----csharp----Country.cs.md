# `basic-computer-games\53_King\csharp\Country.cs`

```

// 命名空间 King 下的内部类 Country
namespace King;

internal class Country
{
    // 初始土地数量
    private const int InitialLand = 1000;

    // 读写接口和随机数生成器
    private readonly IReadWrite _io;
    private readonly IRandom _random;
    
    // 国民、外国人、可耕种土地、工业用地的数量
    private float _rallods;
    private float _countrymen;
    private float _foreigners;
    private float _arableLand;
    private float _industryLand;

    // 构造函数，使用随机数生成器生成初始值
    public Country(IReadWrite io, IRandom random)
        : this(
            io,
            random,
            (int)(60000 + random.NextFloat(1000) - random.NextFloat(1000)),
            (int)(500 + random.NextFloat(10) - random.NextFloat(10)),
            0,
            InitialLand)
    {
    }

    // 构造函数，接受指定的值
    public Country(IReadWrite io, IRandom random, float rallods, float countrymen, float foreigners, float land)
    {
        _io = io;
        _random = random;
        _rallods = rallods;
        _countrymen = countrymen;
        _foreigners = foreigners;
        _arableLand = land;
    }

    // 获取国家状态
    public string GetStatus(int landValue, int plantingCost) 
        => Resource.Status(_rallods, _countrymen, _foreigners, _arableLand, landValue, plantingCost);
    
    // 国民数量
    public float Countrymen => _countrymen;
    // 外国工人数量
    public float Workers => _foreigners;
    // 是否有外国工人
    public bool HasWorkers => _foreigners > 0;
    // 农田数量
    private float FarmLand => _arableLand;
    // 是否有粮食
    public bool HasRallods => _rallods > 0;
    // 粮食数量
    public float Rallods => _rallods;
    // 工业用地数量
    public float IndustryLand => InitialLand - _arableLand;
    // 上一次旅游收入
    public int PreviousTourismIncome { get; private set; }

    // 出售土地
    public bool SellLand(int landValue, out float landSold)
    {
        // 尝试读取出售土地的数量
        if (_io.TryReadValue(
                SellLandPrompt, 
                out landSold, 
                new ValidityTest(v => v <= FarmLand, () => SellLandError(FarmLand))))
        {
            // 更新土地和粮食数量
            _arableLand = (int)(_arableLand - landSold);
            _rallods = (int)(_rallods + landSold * landValue);
            return true;
        }

        return false;
    }

    // 分配粮食
    public bool DistributeRallods(out float rallodsGiven)
    {
        // 尝试读取分配粮食的数量
        if (_io.TryReadValue(
                GiveRallodsPrompt,
                out rallodsGiven, 
                new ValidityTest(v => v <= _rallods, () => GiveRallodsError(_rallods))))
        {
            // 更新粮食数量
            _rallods = (int)(_rallods - rallodsGiven);
            return true;
        }

        return false;
    }

    // 种植土地
    public bool PlantLand(int plantingCost, out float landPlanted)
    {
        // 尝试读取种植土地的数量
        if (_io.TryReadValue(
                PlantLandPrompt, 
                out landPlanted, 
                new ValidityTest(v => v <= _countrymen * 2, PlantLandError1),
                new ValidityTest(v => v <= FarmLand, PlantLandError2(FarmLand)),
                new ValidityTest(v => v * plantingCost <= _rallods, PlantLandError3(_rallods))))
        {
            // 更新粮食数量
            _rallods -= (int)(landPlanted * plantingCost);
            return true;
        }

        return false;
    }

    // 控制污染
    public bool ControlPollution(out float rallodsSpent)
    {
        // 尝试读取控制污染的数量
        if (_io.TryReadValue(
                PollutionPrompt,
                out rallodsSpent, 
                new ValidityTest(v => v <= _rallods, () => PollutionError(_rallods))))
        {
            // 更新粮食数量
            _rallods = (int)(_rallods - rallodsSpent);
            return true;
        }

        return false;
    }

    // 尝试花费粮食
    public bool TrySpend(float amount, float landValue)
    {
        // 如果粮食足够，直接花费
        if (_rallods >= amount)
        {
            _rallods -= amount;
            return true;
        }
        
        // 如果粮食不够，用土地抵消
        _arableLand = (int)(_arableLand - (int)(amount - _rallods) / landValue);
        _rallods = 0;
        return false;
    }

    // 移除死亡人口
    public void RemoveTheDead(int deaths) => _countrymen = (int)(_countrymen - deaths);

    // 迁移人口
    public void Migration(int migration) => _countrymen = (int)(_countrymen + migration);

    // 增加外国工人
    public void AddWorkers(int newWorkers) => _foreigners = (int)(_foreigners + newWorkers);

    // 出售农作物
    public void SellCrops(int income) => _rallods = (int)(_rallods + income);

    // 招待游客
    public void EntertainTourists(int income)
    {
        // 更新上一次旅游收入和粮食数量
        PreviousTourismIncome = income;
        _rallods = (int)(_rallods + income);
    }
}

```