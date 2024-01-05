# `d:/src/tocomm/basic-computer-games\90_Tower\csharp\UI\TowerDisplay.cs`

```
            # 对每个元素进行遍历，将其添加到字符串构建器中
            builder.Append(string.Join(" ", row));
            # 在每行结束后添加换行符
            builder.AppendLine();
        }

        # 返回构建好的字符串
        return builder.ToString();
    }
}
```

在这段代码中，我们首先创建了一个名为`TowerDisplay`的类，它包含一个名为`_towers`的私有成员变量，以及一个接受`towers`参数的构造函数。然后，我们重写了`ToString`方法，该方法使用一个`StringBuilder`对象来构建一个字符串，其中包含了`_towers`中的元素。在`foreach`循环中，我们遍历了`_towers`中的每一行，并将其添加到`builder`中，最后返回构建好的字符串。
            {
                # 调用AppendTower方法，将row.Item1添加到builder中
                AppendTower(row.Item1);
                # 调用AppendTower方法，将row.Item2添加到builder中
                AppendTower(row.Item2);
                # 调用AppendTower方法，将row.Item3添加到builder中
                AppendTower(row.Item3);
                # 在builder中添加换行符
                builder.AppendLine();
            }

            # 将builder转换为字符串并返回
            return builder.ToString();

            # 定义一个名为AppendTower的方法，接受一个整数参数size
            void AppendTower(int size)
            {
                # 计算左右两侧空格的数量
                var padding = 10 - size / 2;
                # 在builder中添加左侧空格、星号和右侧空格
                builder.Append(' ', padding).Append('*', Math.Max(1, size)).Append(' ', padding);
            }
        }
    }
}
```