# `basic-computer-games\53_King\csharp\Resources\Resource.cs`

```
// 声明一个命名空间 King.Resources
namespace King.Resources
{
    // 声明一个内部静态类 Resource
    internal static class Resource
    {
        // 声明一个私有静态布尔变量 _sellLandErrorShown
        private static bool _sellLandErrorShown;

        // 声明一个公共静态属性 Title，返回一个流对象
        public static Stream Title => GetStream();

        // 声明一个公共静态属性 InstructionsPrompt，返回一个字符串
        public static string InstructionsPrompt => GetString();
        
        // 声明一个公共静态方法 InstructionsText，接受一个整数参数 years，返回一个格式化后的字符串
        public static string InstructionsText(int years) => string.Format(GetString(), years);

        // 声明一个公共静态方法 Status，接受多个浮点数参数，返回一个格式化后的字符串
        public static string Status(
            float rallods,
            float countrymen,
            float workers,
            float land,
            float landValue,
            float plantingCost)
            => string.Format(
                workers == 0 ? StatusWithWorkers : StatusSansWorkers,
                rallods,
                (int)countrymen,
                (int)workers,
                (int)land,
                landValue,
                plantingCost);

        // 声明一个私有静态属性 StatusWithWorkers，返回一个字符串
        private static string StatusWithWorkers => GetString();
        
        // 声明一个私有静态属性 StatusSansWorkers，返回一个字符串
        private static string StatusSansWorkers => GetString();

        // 声明一个公共静态属性 SellLandPrompt，返回一个字符串
        public static string SellLandPrompt => GetString();
        
        // 声明一个公共静态方法 SellLandError，接受一个浮点数参数 farmLand，返回一个格式化后的字符串
        public static string SellLandError(float farmLand)
        {
            var error = string.Format(GetString(), farmLand, _sellLandErrorShown ? "" : SellLandErrorReason);
            _sellLandErrorShown = true;
            return error;
        }
        
        // 声明一个私有静态属性 SellLandErrorReason，返回一个字符串
        private static string SellLandErrorReason => GetString();

        // 声明一个公共静态属性 GiveRallodsPrompt，返回一个字符串
        public static string GiveRallodsPrompt => GetString();
        
        // 声明一个公共静态方法 GiveRallodsError，接受一个浮点数参数 rallods，返回一个格式化后的字符串
        public static string GiveRallodsError(float rallods) => string.Format(GetString(), rallods);

        // 声明一个公共静态属性 PlantLandPrompt，返回一个字符串
        public static string PlantLandPrompt => GetString();
        
        // 声明一个公共静态属性 PlantLandError1，返回一个字符串
        public static string PlantLandError1 => GetString();
        
        // 声明一个公共静态方法 PlantLandError2，接受一个浮点数参数 farmLand，返回一个格式化后的字符串
        public static string PlantLandError2(float farmLand) => string.Format(GetString(), farmLand);
        
        // 声明一个公共静态方法 PlantLandError3，接受一个浮点数参数 rallods，返回一个格式化后的字符串
        public static string PlantLandError3(float rallods) => string.Format(GetString(), rallods);

        // 声明一个公共静态属性 PollutionPrompt，返回一个字符串
        public static string PollutionPrompt => GetString();
        
        // 声明一个公共静态方法 PollutionError，接受一个浮点数参数 rallods，返回一个格式化后的字符串
        public static string PollutionError(float rallods) => string.Format(GetString(), rallods);

        // 声明一个公共静态方法 DeathsStarvation，接受一个浮点数参数 deaths，返回一个格式化后的字符串
        public static string DeathsStarvation(float deaths) => string.Format(GetString(), (int)deaths);
    }
}
    // 根据死亡人数返回死亡污染字符串
    public static string DeathsPollution(int deaths) => string.Format(GetString(), deaths);
    // 根据葬礼费用返回葬礼费用字符串
    public static string FuneralExpenses(int expenses) => string.Format(GetString(), expenses);
    // 返回资金不足字符串
    public static string InsufficientReserves => GetString();

    // 根据新工人数量返回工人迁移字符串
    public static string WorkerMigration(int newWorkers) => string.Format(GetString(), newWorkers);
    // 根据迁移人数返回迁移字符串
    public static string Migration(int migration) 
        => string.Format(migration < 0 ? Emigration : Immigration, Math.Abs(migration));
    // 返回移民字符串
    public static string Emigration => GetString();
    // 返回移民字符串
    public static string Immigration => GetString();

    // 根据种植土地数量返回种植土地字符串
    public static string LandPlanted(float landPlanted) 
        => landPlanted > 0 ? string.Format(GetString(), (int)landPlanted) : "";
    // 根据产量、收入和是否有工业返回收获字符串
    public static string Harvest(int yield, int income, bool hasIndustry) 
        => string.Format(GetString(), yield, HarvestReason(hasIndustry), income);
    // 根据是否有工业返回收获原因字符串
    private static string HarvestReason(bool hasIndustry) => hasIndustry ? GetString() : "";

    // 根据旅游收入返回旅游收入字符串
    public static string TourismEarnings(int income) => string.Format(GetString(), income);
    // 根据随机数返回旅游减少原因字符串
    public static string TourismDecrease(IRandom random) => string.Format(GetString(), TourismReason(random));
    // 根据随机数返回旅游减少原因字符串
    private static string TourismReason(IRandom random) => GetStrings()[random.Next(5)];

    // 根据随机数返回结束字符串
    private static string EndAlso(IRandom random)
        => random.Next(10) switch
        {
            <= 3 => GetStrings()[0],
            <= 6 => GetStrings()[1],
            _ => GetStrings()[2]
        };

    // 根据任期长度返回结束祝贺字符串
    public static string EndCongratulations(int termLength) => string.Format(GetString(), termLength);
    // 根据随机数返回结束后果字符串
    private static string EndConsequences(IRandom random) => GetStrings()[random.Next(2)];
    // 根据随机数返回外国工人结束字符串
    public static string EndForeignWorkers(IRandom random) => string.Format(GetString(), EndConsequences(random));
    // 根据死亡人数和随机数返回结束死亡字符串
    public static string EndManyDead(int deaths, IRandom random) => string.Format(GetString(), deaths, EndAlso(random));
    // 返回剩余资金字符串
    public static string EndMoneyLeftOver() => GetString();
    // 返回一个包含随机结局的字符串，结局的一部分已经死亡
    public static string EndOneThirdDead(IRandom random) => string.Format(GetString(), EndConsequences(random));
    
    // 返回一个包含保存年数提示的字符串
    public static string SavedYearsPrompt => GetString();
    // 返回一个包含保存年数错误提示的字符串，包含年数参数
    public static string SavedYearsError(int years) => string.Format(GetString(), years);
    // 返回一个包含保存国库提示的字符串
    public static string SavedTreasuryPrompt => GetString();
    // 返回一个包含保存国民提示的字符串
    public static string SavedCountrymenPrompt => GetString();
    // 返回一个包含保存工人提示的字符串
    public static string SavedWorkersPrompt => GetString();
    // 返回一个包含保存土地提示的字符串
    public static string SavedLandPrompt => GetString();
    // 返回一个包含保存土地错误提示的字符串
    public static string SavedLandError => GetString();

    // 返回一个包含告别提示的字符串
    public static string Goodbye => GetString();

    // 根据调用成员的名称获取字符串数组
    private static string[] GetStrings([CallerMemberName] string? name = null) => GetString(name).Split(';');

    // 根据调用成员的名称获取字符串
    private static string GetString([CallerMemberName] string? name = null)
    {
        // 使用资源流获取字符串
        using var stream = GetStream(name);
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    // 根据调用成员的名称获取资源流
    private static Stream GetStream([CallerMemberName] string? name = null) =>
        // 获取嵌入资源流
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            // 如果资源流不存在，则抛出异常
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
# 闭合大括号，表示代码块的结束
```