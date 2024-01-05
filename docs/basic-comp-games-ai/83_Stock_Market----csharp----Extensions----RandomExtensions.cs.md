# `83_Stock_Market\csharp\Extensions\RandomExtensions.cs`

```
        /// The exclusive upper bound of the range to generate.
        /// </param>
        /// <returns>
        /// An infinite sequence of random numbers within the specified range.
        /// </returns>
        public static IEnumerable<int> InfiniteRandom(this Random random, int min, int max)
        {
            while (true)
            {
                yield return random.Next(min, max);
            }
        }
    }
}
```

注释：

1. ```/// <summary>``` - 用于描述方法或类的作用
2. ```/// Generates an infinite sequence of random numbers.``` - 生成一个无限序列的随机数。
3. ```/// <param name="random">``` - 随机数生成器。
4. ```/// <param name="min">``` - 要生成的范围的包含下限。
5. ```/// <param name="max">``` - 要生成的范围的不包含上限。
6. ```/// <returns>``` - 返回值的描述
7. ```/// An infinite sequence of random numbers within the specified range.``` - 在指定范围内的无限序列的随机数。
8. ```public static IEnumerable<int> InfiniteRandom(this Random random, int min, int max)``` - 定义了一个扩展方法，返回一个无限序列的随机数。
9. ```while (true)``` - 无限循环，生成随机数。
10. ```yield return random.Next(min, max);``` - 生成一个随机数并返回，使用yield关键字可以将方法转换为迭代器。
11. ```}``` - 方法结束。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
        {
            # 使用 while 循环来生成随机数，直到条件不满足为止
            while (true)
                # 使用 random.Next 方法生成一个介于 min 和 max 之间的随机数，并返回
                yield return random.Next(min, max);
        }
    }
}
```