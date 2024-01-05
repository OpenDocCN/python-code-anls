# `d:/src/tocomm/basic-computer-games\72_Queen\csharp\Position.cs`

```
    {
        // 返回一个新的 Position 对象，其对角线和行数分别为当前对象对角线加上移动的对角线数，行数加上移动的行数
        return new Position(position.Diagonal + move.Diagonal, position.Row + move.Row);
    }

    public static bool operator ==(Position left, Position right)
    {
        // 检查两个 Position 对象的对角线和行数是否相等
        return left.Diagonal == right.Diagonal && left.Row == right.Row;
    }

    public static bool operator !=(Position left, Position right)
    {
        // 检查两个 Position 对象的对角线和行数是否不相等
        return !(left == right);
    }

    public override bool Equals(object obj)
    {
        // 检查当前 Position 对象是否与另一个对象相等
        if (obj is Position)
        {
            return this == (Position)obj;
        }
        return false;
    }

    public override int GetHashCode()
    {
        // 返回当前 Position 对象的哈希码
        return (Diagonal, Row).GetHashCode();
    }
}
    => new(Diagonal: position.Diagonal + move.Diagonal, Row: position.Row + move.Row);
```
这行代码是一个重载的加法运算符，用于计算两个Position对象的和。它创建一个新的Position对象，其对角线值是两个操作数的对角线值之和，行值是两个操作数的行值之和。

```
    public static Move operator -(Position to, Position from)
        => new(Diagonal: to.Diagonal - from.Diagonal, Row: to.Row - from.Row);
```
这行代码是一个重载的减法运算符，用于计算两个Position对象的差。它创建一个新的Move对象，其对角线值是to对象的对角线值减去from对象的对角线值，行值是to对象的行值减去from对象的行值。
```