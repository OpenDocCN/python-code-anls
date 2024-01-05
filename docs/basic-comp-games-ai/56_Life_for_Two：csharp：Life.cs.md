# `56_Life_for_Two\csharp\Life.cs`

```
    // 获取当前代的结果
        yield return current;
        // 计算下一代
        current = current.Next();
    }
    // 设置最终结果
    Result = current.Result;
}

IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
```

希望这可以帮助到你！
            current = current.CalculateNextGeneration();  # 计算当前状态的下一代状态
            yield return current;  # 返回当前状态

            if (current.Result is null) { current.AddPieces(_io); }  # 如果当前状态的结果为空，则添加一些数据

        }

        Result = current.Result;  # 将当前状态的结果赋值给类的属性Result

    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();  # 实现接口IEnumerable的GetEnumerator方法，返回迭代器
}
```