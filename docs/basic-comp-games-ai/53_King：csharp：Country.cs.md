# `53_King\csharp\Country.cs`

```
namespace King;  # 命名空间声明

internal class Country  # 定义一个内部类 Country
{
    private const int InitialLand = 1000;  # 声明一个私有常量 InitialLand 并赋值为 1000

    private readonly IReadWrite _io;  # 声明一个只读的 IReadWrite 类型的私有字段 _io
    private readonly IRandom _random;  # 声明一个只读的 IRandom 类型的私有字段 _random
    private float _rallods;  # 声明一个私有的浮点型字段 _rallods
    private float _countrymen;  # 声明一个私有的浮点型字段 _countrymen
    private float _foreigners;  # 声明一个私有的浮点型字段 _foreigners
    private float _arableLand;  # 声明一个私有的浮点型字段 _arableLand
    private float _industryLand;  # 声明一个私有的浮点型字段 _industryLand

    public Country(IReadWrite io, IRandom random)  # 定义一个公共的构造函数，接受 IReadWrite 和 IRandom 类型的参数
        : this(  # 调用当前类的另一个构造函数
            io,  # 传入 io 参数
            random,  # 传入 random 参数
            (int)(60000 + random.NextFloat(1000) - random.NextFloat(1000)),  # 计算初始值并传入
            (int)(500 + random.NextFloat(10) - random.NextFloat(10)),  # 计算初始值并传入
            0,  // 初始化为0
            InitialLand)  // 初始土地
    {
    }

    public Country(IReadWrite io, IRandom random, float rallods, float countrymen, float foreigners, float land)
    {
        _io = io;  // 将传入的io赋值给成员变量_io
        _random = random;  // 将传入的random赋值给成员变量_random
        _rallods = rallods;  // 将传入的rallods赋值给成员变量_rallods
        _countrymen = countrymen;  // 将传入的countrymen赋值给成员变量_countrymen
        _foreigners = foreigners;  // 将传入的foreigners赋值给成员变量_foreigners
        _arableLand = land;  // 将传入的land赋值给成员变量_arableLand
    }

    public string GetStatus(int landValue, int plantingCost) 
        => Resource.Status(_rallods, _countrymen, _foreigners, _arableLand, landValue, plantingCost);  // 调用Resource类的Status方法并返回结果
    
    public float Countrymen => _countrymen;  // 返回_countrymen的值
    public float Workers => _foreigners;  // 返回_foreigners的值
    // 检查是否有外国工人
    public bool HasWorkers => _foreigners > 0;
    // 获取农田面积
    private float FarmLand => _arableLand;
    // 检查是否有铁路
    public bool HasRallods => _rallods > 0;
    // 获取铁路数量
    public float Rallods => _rallods;
    // 获取工业用地面积
    public float IndustryLand => InitialLand - _arableLand;
    // 获取上一次旅游收入
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
            // 更新农田面积和铁路数量
            _arableLand = (int)(_arableLand - landSold);
            _rallods = (int)(_rallods + landSold * landValue);
            return true;
        }

        return false;
    }
        out landPlanted, 
        new ValidityTest(v => v <= _land, () => PlantLandError(_land, plantingCost))))
        {
            _land = (int)(_land - landPlanted);
            return true;
        }

        return false;
    }
```

注释：

1. public bool DistributeRallods(out float rallodsGiven) - 定义一个公共方法，用于分发资源，返回一个布尔值和一个浮点数作为输出参数。
2. if (_io.TryReadValue( - 如果输入输出对象尝试读取值
3. GiveRallodsPrompt - 给出资源的提示信息
4. out rallodsGiven - 输出参数，表示给出的资源数量
5. new ValidityTest(v => v <= _rallods, () => GiveRallodsError(_rallods)) - 创建一个新的有效性测试，检查给出的资源数量是否小于等于可用资源数量，如果不是则调用错误处理方法
6. _rallods = (int)(_rallods - rallodsGiven); - 更新可用资源数量
7. return true; - 返回true表示资源分发成功
8. public bool PlantLand(int plantingCost, out float landPlanted) - 定义一个公共方法，用于种植土地，返回一个布尔值和一个浮点数作为输出参数。
9. PlantLandPrompt - 种植土地的提示信息
10. out landPlanted - 输出参数，表示种植的土地数量
11. new ValidityTest(v => v <= _land, () => PlantLandError(_land, plantingCost)) - 创建一个新的有效性测试，检查种植的土地数量是否小于等于可用土地数量，如果不是则调用错误处理方法
12. _land = (int)(_land - landPlanted); - 更新可用土地数量
13. return false; - 返回false表示土地种植失败
                out landPlanted,  # 定义变量 landPlanted，用于存储种植的土地数量
                new ValidityTest(v => v <= _countrymen * 2, PlantLandError1),  # 使用 ValidityTest 检查条件，确保种植的土地数量不超过国民数量的两倍，如果条件不满足则返回 PlantLandError1
                new ValidityTest(v => v <= FarmLand, PlantLandError2(FarmLand)),  # 使用 ValidityTest 检查条件，确保种植的土地数量不超过农田的数量，如果条件不满足则返回 PlantLandError2(FarmLand)
                new ValidityTest(v => v * plantingCost <= _rallods, PlantLandError3(_rallods))))  # 使用 ValidityTest 检查条件，确保种植所需的成本不超过可用的资源数量，如果条件不满足则返回 PlantLandError3(_rallods)
        {
            _rallods -= (int)(landPlanted * plantingCost);  # 如果条件满足，则减去种植所需的资源数量
            return true;  # 返回 true
        }

        return false;  # 如果条件不满足，则返回 false
    }

    public bool ControlPollution(out float rallodsSpent)
    {
        if (_io.TryReadValue(
                PollutionPrompt,  # 提示用户输入污染控制所需的资源数量
                out rallodsSpent,  # 定义变量 rallodsSpent，用于存储用户输入的资源数量
                new ValidityTest(v => v <= _rallods, () => PollutionError(_rallods))))  # 使用 ValidityTest 检查条件，确保用户输入的资源数量不超过可用的资源数量，如果条件不满足则返回 PollutionError(_rallods)
        {
            _rallods = (int)(_rallods - rallodsSpent);  # 如果条件满足，则减去用户输入的资源数量
        // 如果剩余资源大于等于指定数量，则扣除指定数量的资源并返回 true
        public bool TrySpend(float amount, float landValue)
        {
            if (_rallods >= amount)
            {
                _rallods -= amount;
                return true;
            }
            // 如果剩余资源不足以扣除指定数量，则根据土地价值计算可耕种土地的减少量，并返回 false
            _arableLand = (int)(_arableLand - (int)(amount - _rallods) / landValue);
            _rallods = 0;
            return false;
        }

        // 减少人口数量
        public void RemoveTheDead(int deaths) => _countrymen = (int)(_countrymen - deaths);
```
在这段代码中，我添加了注释来解释每个方法的作用。第一个方法是TrySpend，它用于尝试花费资源，如果资源足够则扣除指定数量的资源并返回true，如果资源不足则根据土地价值计算可耕种土地的减少量，并返回false。第二个方法是RemoveTheDead，它用于减少人口数量。
    # 将移民人数加到国民人口中
    public void Migration(int migration) => _countrymen = (int)(_countrymen + migration);

    # 增加外来工人数量
    public void AddWorkers(int newWorkers) => _foreigners = (int)(_foreigners + newWorkers);

    # 出售农作物，增加收入
    public void SellCrops(int income) => _rallods = (int)(_rallods + income);

    # 接待游客，增加收入，并记录上一次的旅游收入
    public void EntertainTourists(int income)
    {
        PreviousTourismIncome = income;
        _rallods = (int)(_rallods + income);
    }
}
```