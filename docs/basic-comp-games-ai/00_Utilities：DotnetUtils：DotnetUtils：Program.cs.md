# `d:/src/tocomm/basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\Program.cs`

```
// 使用 System.Xml.Linq 命名空间
using System.Xml.Linq;
// 使用 DotnetUtils 命名空间
using DotnetUtils;
// 使用 System.Console 类的静态成员
using static System.Console;
// 使用 System.IO.Path 类的静态成员
using static System.IO.Path;
// 使用 DotnetUtils.Methods 类的静态成员
using static DotnetUtils.Methods;
// 使用 DotnetUtils.Functions 类的静态成员
using static DotnetUtils.Functions;

// 获取端口信息
var infos = PortInfos.Get;

// 创建包含动作和描述的元组数组
var actions = new (Action action, string description)[] {
    // 输出信息 -- 解决方案、项目和代码文件
    (printInfos, "Output information -- solution, project, and code files"),
    // 输出缺失的解决方案
    (missingSln, "Output missing sln"),
    // 输出错误命名的解决方案
    (unexpectedSlnName, "Output misnamed sln"),
    // 输出多个解决方案文件
    (multipleSlns, "Output multiple sln files"),
    // 输出缺失的项目文件
    (missingProj, "Output missing project file"),
    // 输出错误命名的项目文件
    (unexpectedProjName, "Output misnamed project files"),
    // 输出多个项目文件
    (multipleProjs, "Output multiple project files"),
    // 检查 .csproj/.vbproj 文件的目标框架、可空性等
    (checkProjects, "Check .csproj/.vbproj files for target framework, nullability etc."),
    // 检查每个端口至少有一个可执行项目
    (checkExecutableProject, "Check that there is at least one executable project per port"),
    // 输出没有任何代码文件的端口
    (noCodeFiles, "Output ports without any code files"),
    (printPortInfo, "Print info about a single port"),  # 创建一个包含函数和描述的元组，用于打印单个端口的信息

    (generateMissingSlns, "Generate solution files when missing"),  # 创建一个包含函数和描述的元组，用于在缺失时生成解决方案文件
    (generateMissingProjs, "Generate project files when missing")  # 创建一个包含函数和描述的元组，用于在缺失时生成项目文件
};

foreach (var (_, description, index) in actions.WithIndex()) {  # 遍历包含函数和描述的元组列表，并获取索引和描述
    WriteLine($"{index}: {description}");  # 打印索引和描述
}

WriteLine();

actions[getChoice(actions.Length - 1)].action();  # 调用根据用户选择的索引执行相应函数的函数

void printSlns(PortInfo pi) {  # 定义一个函数，用于打印解决方案文件
    switch (pi.Slns.Length) {  # 根据解决方案文件的数量进行判断
        case 0:  # 如果没有解决方案文件
            WriteLine("No sln");  # 打印提示信息
            break;  # 结束判断
        case 1:  # 如果只有一个解决方案文件
// 输出解决方案的相对路径
WriteLine($"Solution: {pi.Slns[0].RelativePath(pi.LangPath)}");
// 终止 switch 语句
break;
// 当项目数量大于1时
case > 1:
    // 输出多个解决方案的提示
    WriteLine("Solutions:");
    // 遍历每个解决方案并输出其相对路径
    foreach (var sln in pi.Slns) {
        Write(sln.RelativePath(pi.LangPath));
        WriteLine();
    }
    // 终止 switch 语句
    break;
}

// 定义一个打印项目的函数
void printProjs(PortInfo pi) {
    // 根据项目数量进行不同的操作
    switch (pi.Projs.Length) {
        // 当项目数量为0时
        case 0:
            // 输出没有项目的提示
            WriteLine("No project");
            // 终止 switch 语句
            break;
        // 当项目数量为1时
        case 1:
            // 输出单个项目的相对路径
            WriteLine($"Project: {pi.Projs[0].RelativePath(pi.LangPath)}");
            // 终止 switch 语句
            break;
        case > 1:  # 如果项目数量大于1
            WriteLine("Projects:");  # 输出"Projects:"到控制台
            foreach (var proj in pi.Projs) {  # 遍历pi.Projs中的每个项目
                Write(proj.RelativePath(pi.LangPath));  # 输出项目的相对路径到控制台
                WriteLine();  # 输出换行符到控制台
            }
            break;  # 跳出循环
    }
}

void printInfos() {  # 定义一个名为printInfos的函数
    foreach (var item in infos) {  # 遍历infos中的每个元素
        WriteLine(item.LangPath);  # 输出item的LangPath属性到控制台
        WriteLine();  # 输出换行符到控制台

        printSlns(item);  # 调用printSlns函数并传入item作为参数
        WriteLine();  # 输出换行符到控制台

        printProjs(item);  # 调用printProjs函数并传入item作为参数
        WriteLine();  # 输出换行符到控制台
        // 获取代码文件
        foreach (var file in item.CodeFiles) {
            WriteLine(file.RelativePath(item.LangPath)); // 输出文件相对路径
        }
        WriteLine(new string('-', 50)); // 输出分隔线
    }
}

void missingSln() {
    var data = infos.Where(x => x.Slns.None()).ToArray(); // 从infos中筛选出Slns为空的数据并转换为数组
    foreach (var item in data) {
        WriteLine(item.LangPath); // 输出LangPath
    }
    WriteLine(); // 输出空行
    WriteLine($"Count: {data.Length}"); // 输出数据长度
}

void unexpectedSlnName() {
    var counter = 0; // 初始化计数器为0
    foreach (var item in infos) {  # 遍历名为infos的集合中的每个元素
        if (item.Slns.None()) { continue; }  # 如果item的Slns属性为空，则跳过当前循环

        var expectedSlnName = $"{item.GameName}.sln";  # 根据item的GameName属性拼接出预期的解决方案文件名
        if (item.Slns.Contains(Combine(item.LangPath, expectedSlnName), StringComparer.InvariantCultureIgnoreCase)) { continue; }  # 如果item的Slns属性包含预期的解决方案文件名，则跳过当前循环

        counter += 1;  # 计数器加1
        WriteLine(item.LangPath);  # 输出item的LangPath属性
        WriteLine($"Expected: {expectedSlnName}");  # 输出预期的解决方案文件名

        printSlns(item);  # 调用printSlns函数打印item的Slns属性

        WriteLine();  # 输出空行
    }
    WriteLine($"Count: {counter}");  # 输出计数器的值
}

void multipleSlns() {  # 定义名为multipleSlns的函数
    var data = infos.Where(x => x.Slns.Length > 1).ToArray();  # 从infos集合中筛选出Slns属性长度大于1的元素，存入data数组
    foreach (var item in data) {  # 遍历data数组中的每个元素
        WriteLine(item.LangPath);  # 输出item的LangPath属性值
        printSlns(item);  # 调用printSlns函数，传入item作为参数
    }
    WriteLine();  # 输出空行
    WriteLine($"Count: {data.Length}");  # 输出data数组的长度

void missingProj() {
    var data = infos.Where(x => x.Projs.None()).ToArray();  # 从infos中筛选出Projs属性为空的元素，转换为数组存储在data变量中
    foreach (var item in data) {  # 遍历data数组中的元素
        WriteLine(item.LangPath);  # 输出item的LangPath属性值
    }
    WriteLine();  # 输出空行
    WriteLine($"Count: {data.Length}");  # 输出data数组的长度
}

void unexpectedProjName() {
    var counter = 0;  # 初始化counter变量为0
    foreach (var item in infos) {  # 遍历infos数组中的元素
        if (item.Projs.None()) { continue; }  # 如果item的Projs属性为空，则跳过当前循环
        # 构建预期的项目名称，格式为“游戏名.项目扩展名”
        var expectedProjName = $"{item.GameName}.{item.ProjExt}";
        # 如果项目列表中包含预期的项目路径，则跳过当前循环
        if (item.Projs.Contains(Combine(item.LangPath, expectedProjName))) { continue; }

        # 计数器加一
        counter += 1;
        # 输出语言路径
        WriteLine(item.LangPath);
        # 输出预期的项目名称
        WriteLine($"Expected: {expectedProjName}");

        # 打印项目信息
        printProjs(item);

        # 输出空行
        WriteLine();
    }
    # 输出多项目的数量
    WriteLine($"Count: {counter}");
}

# 处理多项目的函数
void multipleProjs() {
    # 从infos中筛选出项目数量大于1的数据
    var data = infos.Where(x => x.Projs.Length > 1).ToArray();
    # 遍历筛选出的数据
    foreach (var item in data) {
        # 输出语言路径
        WriteLine(item.LangPath);
        # 输出空行
        WriteLine();
        printProjs(item);  # 调用printProjs函数，传入item参数
    }
    WriteLine();  # 输出空行
    WriteLine($"Count: {data.Length}");  # 输出data长度的信息
}

void generateMissingSlns() {  # 定义generateMissingSlns函数
    foreach (var item in infos.Where(x => x.Slns.None())) {  # 遍历infos中满足条件的item
        var result = RunProcess("dotnet", $"new sln -n {item.GameName} -o {item.LangPath}");  # 运行dotnet命令，生成sln文件
        WriteLine(result);  # 输出结果

        var slnFullPath = Combine(item.LangPath, $"{item.GameName}.sln");  # 组合sln文件的完整路径
        foreach (var proj in item.Projs) {  # 遍历item的Projs
            result = RunProcess("dotnet", $"sln {slnFullPath} add {proj}");  # 运行dotnet命令，将proj添加到sln文件中
            WriteLine(result);  # 输出结果
        }
    }
}

void generateMissingProjs() {  # 定义generateMissingProjs函数
    foreach (var item in infos.Where(x => x.Projs.None())) {
        # 遍历 infos 列表中满足条件的元素，条件是其 Projs 属性为空
        # 我们无法使用 dotnet 命令使用内置控制台模板创建新项目，因为该模板的一部分是 Program.cs / Program.vb 文件。
        # 如果已经存在代码文件，就不需要添加新的空文件；而且如果已经存在这样的文件，它可能会尝试覆盖它。

        var projText = item.Lang switch {
            "csharp" => @"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
    <LangVersion>10</LangVersion>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
",
            "vbnet" => @$"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <RootNamespace>{item.GameName}</RootNamespace>
```
```


```python
    foreach (var item in infos.Where(x => x.Projs.None())) {
        # 遍历 infos 列表中满足条件的元素，条件是其 Projs 属性为空
        # 我们无法使用 dotnet 命令使用内置控制台模板创建新项目，因为该模板的一部分是 Program.cs / Program.vb 文件。
        # 如果已经存在代码文件，就不需要添加新的空文件；而且如果已经存在这样的文件，它可能会尝试覆盖它。

        var projText = item.Lang switch {
            "csharp" => @"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
    <LangVersion>10</LangVersion>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
",
            "vbnet" => @$"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <RootNamespace>{item.GameName}</RootNamespace>
```
    <TargetFramework>net6.0</TargetFramework>  // 设置项目的目标框架为 .NET 6.0
    <LangVersion>16.9</LangVersion>  // 设置项目的语言版本为 16.9
  </PropertyGroup>
</Project>
",
            _ => throw new InvalidOperationException()  // 如果条件不匹配，则抛出 InvalidOperationException 异常
        };
        var projFullPath = Combine(item.LangPath, $"{item.GameName}.{item.ProjExt}");  // 构建项目的完整路径
        File.WriteAllText(projFullPath, projText);  // 将项目文本内容写入到项目文件中

        if (item.Slns.Length == 1) {  // 如果解决方案文件数量为1
            var result = RunProcess("dotnet", $"sln {item.Slns[0]} add {projFullPath}");  // 运行 dotnet 命令向解决方案中添加项目
            WriteLine(result);  // 输出命令执行结果
        }
    }
}

void checkProjects() {  // 定义一个名为 checkProjects 的函数
    foreach (var info in infos) {  // 遍历 infos 列表中的每个元素
        WriteLine(info.LangPath);  // 输出 info 对象的 LangPath 属性值
        printProjectWarnings(info);  # 调用printProjectWarnings函数，传入info参数
    }
}

void printProjectWarnings(PortInfo info) {  # 定义printProjectWarnings函数，接受PortInfo类型的参数info
    foreach (var proj in info.Projs) {  # 遍历info.Projs中的每个项目
        var warnings = new List<string>();  # 创建一个字符串列表warnings
        var parent = XDocument.Load(proj).Element("Project")?.Element("PropertyGroup");  # 从proj加载XDocument，获取"Project"元素下的"PropertyGroup"元素，如果为空则返回null

        var (  # 定义一个元组，包含framework, nullable, implicitUsing, rootNamespace, langVersion, optionStrict这些变量
            framework,
            nullable,
            implicitUsing,
            rootNamespace,
            langVersion,
            optionStrict
        ) = (
            getValue(parent, "TargetFramework", "TargetFrameworks"),  # 调用getValue函数，传入parent和"TargetFramework", "TargetFrameworks"参数，将返回值赋给framework变量
            getValue(parent, "Nullable"),  # 调用getValue函数，传入parent和"Nullable"参数，将返回值赋给nullable变量
            getValue(parent, "ImplicitUsings"),  # 调用getValue函数，传入parent和"ImplicitUsings"参数，将返回值赋给implicitUsing变量
            getValue(parent, "RootNamespace"),  # 从父节点中获取名为"RootNamespace"的值
            getValue(parent, "LangVersion"),  # 从父节点中获取名为"LangVersion"的值
            getValue(parent, "OptionStrict")  # 从父节点中获取名为"OptionStrict"的值
        );

        if (framework != "net6.0") {  # 如果framework不等于"net6.0"
            warnings.Add($"Target: {framework}");  # 向警告列表中添加字符串"Target: {framework}"
        }

        if (info.Lang == "csharp") {  # 如果info.Lang等于"csharp"
            if (nullable != "enable") {  # 如果nullable不等于"enable"
                warnings.Add($"Nullable: {nullable}");  # 向警告列表中添加字符串"Nullable: {nullable}"
            }
            if (implicitUsing != "enable") {  # 如果implicitUsing不等于"enable"
                warnings.Add($"ImplicitUsings: {implicitUsing}");  # 向警告列表中添加字符串"ImplicitUsings: {implicitUsing}"
            }
            if (rootNamespace != null && rootNamespace != info.GameName) {  # 如果rootNamespace不为空且不等于info.GameName
                warnings.Add($"RootNamespace: {rootNamespace}");  # 向警告列表中添加字符串"RootNamespace: {rootNamespace}"
            }
            if (langVersion != "10") {  # 如果langVersion不等于"10"
# 如果编程语言是 "vbnet"，并且根命名空间不等于游戏名称，则添加警告
if (info.Lang == "vbnet") {
    if (rootNamespace != info.GameName) {
        warnings.Add($"RootNamespace: {rootNamespace}");
    }
    # 如果语言版本不是 "16.9"，则添加警告
    if (langVersion != "16.9") {
        warnings.Add($"LangVersion: {langVersion}");
    }
    # 如果选项严格不是 "On"，则添加警告
    if (optionStrict != "On") {
        warnings.Add($"OptionStrict: {optionStrict}");
    }
}

# 如果存在任何警告，则输出相对路径、警告内容，并换行
if (warnings.Any()) {
    WriteLine(proj.RelativePath(info.LangPath));
    WriteLine(string.Join("\n", warnings));
    WriteLine();
}
void checkExecutableProject() {
    // 遍历infos列表中的每个项目
    foreach (var item in infos) {
        // 检查项目中的所有proj，如果没有一个proj的OutputType为"Exe"，则执行下面的代码
        if (item.Projs.All(proj => getValue(proj, "OutputType") != "Exe")) {
            // 输出item的LangPath
            WriteLine($"{item.LangPath}");
        }
    }
}

void noCodeFiles() {
    // 从infos列表中筛选出CodeFiles为空的项，并按照Lang属性进行排序
    var qry = infos
        .Where(x => x.CodeFiles.None())
        .OrderBy(x => x.Lang);
    // 遍历qry列表中的每个项目
    foreach (var item in qry) {
        // 输出item的LangPath
        WriteLine(item.LangPath);
    }
}
void tryBuild() {
    // 如果有代码文件，尝试构建
    // if has code files, try to build
}

void printPortInfo() {
    // 提示输入端口号
    // prompt for port number
    Write("Enter number from 1 to 96 ");
    var index = getChoice(1, 96);

    Write("Enter 0 for C#, 1 for VB ");
    var lang = getChoice(1) switch {
        0 => "csharp",
        1 => "vbnet",
        _ => throw new InvalidOperationException()
    };

    WriteLine();

    // 根据输入的端口号和语言类型查找相应的信息
    // find the corresponding information based on the input port number and language type
    var info = infos.Single(x => x.Index == index && x.Lang == lang);
}
    WriteLine(info.LangPath); // 打印 info.LangPath 变量的值
    WriteLine(new string('-', 50)); // 打印 50 个连字符

    // 打印解决方案
    printSlns(info);

    // 解决方案名称/位置不匹配？（预期 x）
    var expectedSlnName = Combine(info.LangPath, $"{info.GameName}.sln"); // 创建预期的解决方案名称
    if (!info.Slns.Contains(expectedSlnName)) { // 如果 info.Slns 不包含预期的解决方案名称
        WriteLine($"Expected name/path: {expectedSlnName.RelativePath(info.LangPath)}"); // 打印预期的解决方案名称/路径
    }

    // 是否有可执行项目？
    if (info.Projs.All(proj => getValue(proj, "OutputType") != "Exe")) { // 如果所有项目的输出类型不是 "Exe"
        WriteLine("No executable project"); // 打印 "No executable project"
    }

    WriteLine(); // 打印空行
    // 打印项目信息
    printProjs(info);

    // 项目名称/位置不匹配？（预期为 x）
    var expectedProjName = Combine(info.LangPath, $"{info.GameName}.{info.ProjExt}");
    if (info.Projs.Length < 2 && !info.Projs.Contains(expectedProjName)) {
        WriteLine($"Expected name/path: {expectedProjName.RelativePath(info.LangPath)}");
    }

    WriteLine();

    // 验证项目属性
    printProjectWarnings(info);

    WriteLine("Code files:");

    // 列出代码文件
    foreach (var codeFile in info.CodeFiles) {
        WriteLine(codeFile.RelativePath(info.LangPath));
    }
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建一个ZIP文件对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历ZIP文件中的文件名列表，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭ZIP文件对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```