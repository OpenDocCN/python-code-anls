# `d:/src/tocomm/basic-computer-games\53_King\csharp\Reign.cs`

```
    _country = country;
    _yearNumber = year;
}

public void StartReign()
{
    // 检查是否已经达到最大任期
    if (_yearNumber >= MaxTerm)
    {
        _io.WriteLine("The reign has reached its maximum term.");
        return;
    }

    // 开始统治
    _io.WriteLine("The reign has begun.");

    // 进行统治的相关操作，这里省略具体代码

    // 增加统治年限
    _yearNumber += 1;
}

public void EndReign()
{
    // 结束统治
    _io.WriteLine("The reign has ended.");

    // 进行统治结束的相关操作，这里省略具体代码
}

public float GetYearNumber()
{
    return _yearNumber;
}
```

希望以上注释对你有所帮助。
        _country = country;  # 将传入的国家赋值给私有变量_country
        _yearNumber = year;  # 将传入的年份赋值给私有变量_yearNumber
    }

    public bool PlayYear()  # 定义一个公共方法PlayYear，返回布尔值
    {
        var year = new Year(_country, _random, _io);  # 创建一个新的Year对象，传入_country、_random和_io参数

        _io.Write(year.Status);  # 使用_io对象的Write方法输出year对象的Status属性值

        var result = year.GetPlayerActions() ?? year.EvaluateResults() ?? IsAtEndOfTerm();  # 使用年份对象的GetPlayerActions方法，如果返回值为空则使用EvaluateResults方法，如果还是为空则调用IsAtEndOfTerm方法，并将结果赋值给result
        if (result.IsGameOver)  # 如果result的IsGameOver属性为true
        {
            _io.WriteLine(result.Message);  # 使用_io对象的WriteLine方法输出result的Message属性值
            return false;  # 返回false
        }

        return true;  # 返回true
    }
private Result IsAtEndOfTerm()  # 定义一个私有方法IsAtEndOfTerm，用于判断是否到达学期末尾
    => _yearNumber == MaxTerm  # 使用箭头函数判断当前年数是否等于最大学期数
        ? Result.GameOver(EndCongratulations(MaxTerm))  # 如果是最后一个学期，则返回游戏结束的结果，并传入最大学期数作为参数
        : Result.Continue;  # 如果不是最后一个学期，则返回继续游戏的结果
}
```