# `53_King\csharp\Resources\Resource.cs`

```
using System.Reflection;  // 导入 System.Reflection 命名空间，用于获取程序集信息
using System.Runtime.CompilerServices;  // 导入 System.Runtime.CompilerServices 命名空间，用于访问程序集中的资源

namespace King.Resources;  // 声明 King.Resources 命名空间

internal static class Resource  // 声明 Resource 类，限定只能在当前程序集内部访问
{
    private static bool _sellLandErrorShown;  // 声明私有静态布尔变量 _sellLandErrorShown

    public static Stream Title => GetStream();  // 声明公共静态属性 Title，返回 GetStream() 方法的结果
    
    public static string InstructionsPrompt => GetString();  // 声明公共静态属性 InstructionsPrompt，返回 GetString() 方法的结果
    public static string InstructionsText(int years) => string.Format(GetString(), years);  // 声明公共静态方法 InstructionsText，接受一个整数参数 years，返回格式化后的字符串

    public static string Status(  // 声明公共静态方法 Status，接受五个浮点数参数
        float rallods,
        float countrymen,
        float workers,
        float land,
        float landValue,
```
```csharp
        float plantingCost)  # 定义一个名为 plantingCost 的浮点型参数
        => string.Format(  # 使用 string.Format 方法格式化字符串
            workers == 0 ? StatusWithWorkers : StatusSansWorkers,  # 根据 workers 的值选择不同的字符串模板
            rallods,  # 使用 rallods 变量的值作为参数
            (int)countrymen,  # 将 countrymen 变量的值转换为整数作为参数
            (int)workers,  # 将 workers 变量的值转换为整数作为参数
            (int)land,  # 将 land 变量的值转换为整数作为参数
            landValue,  # 使用 landValue 变量的值作为参数
            plantingCost);  # 使用 plantingCost 变量的值作为参数

    private static string StatusWithWorkers => GetString();  # 定义一个名为 StatusWithWorkers 的静态字符串属性
    private static string StatusSansWorkers => GetString();  # 定义一个名为 StatusSansWorkers 的静态字符串属性

    public static string SellLandPrompt => GetString();  # 定义一个名为 SellLandPrompt 的公共静态字符串属性
    public static string SellLandError(float farmLand)  # 定义一个名为 SellLandError 的公共静态字符串方法，接受一个名为 farmLand 的浮点型参数
    {
        var error = string.Format(GetString(), farmLand, _sellLandErrorShown ? "" : SellLandErrorReason);  # 使用 string.Format 方法格式化字符串
        _sellLandErrorShown = true;  # 将 _sellLandErrorShown 变量的值设为 true
        return error;  # 返回格式化后的字符串
    }
    // 获取土地出售错误原因的字符串
    private static string SellLandErrorReason => GetString();

    // 获取赠送资源的提示字符串
    public static string GiveRallodsPrompt => GetString();
    // 获取赠送资源错误的字符串，包含资源数量
    public static string GiveRallodsError(float rallods) => string.Format(GetString(), rallods);

    // 获取种植土地的提示字符串
    public static string PlantLandPrompt => GetString();
    // 获取种植土地错误1的字符串
    public static string PlantLandError1 => GetString();
    // 获取种植土地错误2的字符串，包含农田数量
    public static string PlantLandError2(float farmLand) => string.Format(GetString(), farmLand);
    // 获取种植土地错误3的字符串，包含资源数量
    public static string PlantLandError3(float rallods) => string.Format(GetString(), rallods);

    // 获取污染提示字符串
    public static string PollutionPrompt => GetString();
    // 获取污染错误的字符串，包含资源数量
    public static string PollutionError(float rallods) => string.Format(GetString(), rallods);

    // 获取饥饿死亡的字符串，包含死亡人数
    public static string DeathsStarvation(float deaths) => string.Format(GetString(), (int)deaths);
    // 获取污染死亡的字符串，包含死亡人数
    public static string DeathsPollution(int deaths) => string.Format(GetString(), deaths);
    // 获取葬礼费用的字符串，包含费用
    public static string FuneralExpenses(int expenses) => string.Format(GetString(), expenses);
    // 获取储备不足的字符串
    public static string InsufficientReserves => GetString();

    // 获取工人迁移的字符串，包含新工人数量
    public static string WorkerMigration(int newWorkers) => string.Format(GetString(), newWorkers);
    // 获取迁移的字符串，包含迁移人数
    public static string Migration(int migration) 
    => string.Format(migration < 0 ? Emigration : Immigration, Math.Abs(migration));
    # 根据迁移人口的正负情况选择对应的字符串模板，并使用迁移人口的绝对值进行格式化

    public static string Emigration => GetString();
    # 返回一个表示移民的字符串

    public static string Immigration => GetString();
    # 返回一个表示移民的字符串

    public static string LandPlanted(float landPlanted) 
        => landPlanted > 0 ? string.Format(GetString(), (int)landPlanted) : "";
    # 如果种植的土地面积大于0，则使用对应的字符串模板进行格式化，否则返回空字符串

    public static string Harvest(int yield, int income, bool hasIndustry) 
        => string.Format(GetString(), yield, HarvestReason(hasIndustry), income);
    # 使用收获量、收入和是否有工业的情况来格式化字符串

    private static string HarvestReason(bool hasIndustry) => hasIndustry ? GetString() : "";
    # 根据是否有工业的情况返回对应的字符串

    public static string TourismEarnings(int income) => string.Format(GetString(), income);
    # 使用旅游收入来格式化字符串

    public static string TourismDecrease(IRandom random) => string.Format(GetString(), TourismReason(random));
    # 使用随机生成的旅游减少原因来格式化字符串

    private static string TourismReason(IRandom random) => GetStrings()[random.Next(5)];
    # 从预定义的旅游减少原因中随机选择一个返回

    private static string EndAlso(IRandom random)
        => random.Next(10) switch
        {
            <= 3 => GetStrings()[0],
            <= 6 => GetStrings()[1],
            _ => GetStrings()[2]
    # 根据随机数的范围选择对应的字符串模板进行格式化
    };

    public static string EndCongratulations(int termLength) => string.Format(GetString(), termLength);
    // 返回一条祝贺消息，包含任期长度
    private static string EndConsequences(IRandom random) => GetStrings()[random.Next(2)];
    // 返回一条任期结束的后果消息，根据随机数选择
    public static string EndForeignWorkers(IRandom random) => string.Format(GetString(), EndConsequences(random));
    // 返回一条关于外国工人的消息，包含任期结束的后果
    public static string EndManyDead(int deaths, IRandom random) => string.Format(GetString(), deaths, EndAlso(random));
    // 返回一条关于死亡人数的消息，包含死亡人数和其他信息
    public static string EndMoneyLeftOver() => GetString();
    // 返回一条剩余资金的消息
    public static string EndOneThirdDead(IRandom random) => string.Format(GetString(), EndConsequences(random));
    // 返回一条关于三分之一人口死亡的消息，包含任期结束的后果

    public static string SavedYearsPrompt => GetString();
    // 返回一条保存年限的提示消息
    public static string SavedYearsError(int years) => string.Format(GetString(), years);
    // 返回一条保存年限错误的消息，包含年限
    public static string SavedTreasuryPrompt => GetString();
    // 返回一条保存国库的提示消息
    public static string SavedCountrymenPrompt => GetString();
    // 返回一条保存同胞的提示消息
    public static string SavedWorkersPrompt => GetString();
    // 返回一条保存工人的提示消息
    public static string SavedLandPrompt => GetString();
    // 返回一条保存土地的提示消息
    public static string SavedLandError => GetString();
    // 返回一条保存土地错误的消息

    public static string Goodbye => GetString();
    // 返回一条告别消息

    private static string[] GetStrings([CallerMemberName] string? name = null) => GetString(name).Split(';');
    // 获取字符串数组，根据调用者的名称获取对应的字符串并以分号分割
    # 从调用者的成员名获取字符串
    def GetString(name=None):
        # 使用获取的成员名获取流
        stream = GetStream(name)
        # 使用流创建读取器
        reader = StreamReader(stream)
        # 读取并返回流中的所有内容
        return reader.ReadToEnd()

    # 从调用者的成员名获取流
    def GetStream(name=None):
        # 从当前程序集获取嵌入资源流
        stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(f"{typeof(Resource).Namespace}.{name}.txt")
        # 如果找不到资源流，则抛出异常
        if stream is None:
            raise Exception(f"Could not find embedded resource stream '{name}'.")
        return stream
```