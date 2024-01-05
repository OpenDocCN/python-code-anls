# `d:/src/tocomm/basic-computer-games\90_Tower\csharp\Models\Needle.cs`

```
        }
        else
        {
            return false;
        }
    }

    public int Take()
    {
        return _disks.Pop();
    }

    public IEnumerator<int> GetEnumerator()
    {
        return _disks.GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
            }
            # 返回 false
            return false;
        }

        # 尝试获取栈顶的磁盘，并将其弹出
        public bool TryGetTopDisk(out int disk) => _disks.TryPop(out disk);

        # 获取枚举器，用于遍历栈中的元素
        public IEnumerator<int> GetEnumerator() =>
            Enumerable.Repeat(0, 7 - _disks.Count).Concat(_disks).GetEnumerator();

        # 获取非泛型枚举器
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
```