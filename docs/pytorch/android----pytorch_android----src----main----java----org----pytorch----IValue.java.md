# `.\pytorch\android\pytorch_android\src\main\java\org\pytorch\IValue.java`

```
/**
 * Java表示TorchScript值的类，实现为标记联合，可以是支持的多种类型之一：https://pytorch.org/docs/stable/jit.html#types 。
 *
 * 调用不适当类型的 {@code toX} 方法将抛出 {@link IllegalStateException} 异常。
 *
 * {@code IValue} 对象可以使用 {@code IValue.from(value)}，{@code IValue.tupleFrom(value1, value2, ...)}，
 * {@code IValue.listFrom(value1, value2, ...)} 或其中一种 {@code dict} 方法进行构造，具体取决于键的类型。
 *
 * 从 {@code IValue} 对象中获取数据使用 {@code toX()} 方法。注意 {@code str} 类型的 IValue 必须使用 {@link #toStr()}，
 * 而不是 {@link #toString()} 提取。
 *
 * {@code IValue} 对象可能会保留对传入其构造函数的对象的引用，并且可能会从 {@code toX()} 返回其内部状态的引用。
 */
@DoNotStrip
public class IValue {
  private static final int TYPE_CODE_NULL = 1;
  private static final int TYPE_CODE_TENSOR = 2;
  private static final int TYPE_CODE_BOOL = 3;
  private static final int TYPE_CODE_LONG = 4;
  private static final int TYPE_CODE_DOUBLE = 5;
  private static final int TYPE_CODE_STRING = 6;
  private static final int TYPE_CODE_TUPLE = 7;
  private static final int TYPE_CODE_BOOL_LIST = 8;
  private static final int TYPE_CODE_LONG_LIST = 9;
  private static final int TYPE_CODE_DOUBLE_LIST = 10;
  private static final int TYPE_CODE_TENSOR_LIST = 11;
  private static final int TYPE_CODE_LIST = 12;
  private static final int TYPE_CODE_DICT_STRING_KEY = 13;
  private static final int TYPE_CODE_DICT_LONG_KEY = 14;

  // TorchScript值类型名称数组
  private String[] TYPE_NAMES = {
    "Unknown",
    "Null",
    "Tensor",
    "Bool",
    "Long",
    "Double",
    "String",
    "Tuple",
    "BoolList",
    "LongList",
    "DoubleList",
    "TensorList",
    "GenericList",
    "DictStringKey",
    "DictLongKey",
  };

  @DoNotStrip
  private final int mTypeCode; // 值的类型代码

  @DoNotStrip
  private Object mData; // 数据对象引用

  /**
   * 私有构造函数，创建指定类型代码的 {@code IValue} 对象。
   *
   * @param typeCode 值的类型代码
   */
  @DoNotStrip
  private IValue(int typeCode) {
    this.mTypeCode = typeCode;
  }

  /**
   * 检查值是否为 {@code Null} 类型。
   *
   * @return 如果值是 {@code Null} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isNull() {
    return TYPE_CODE_NULL == this.mTypeCode;
  }

  /**
   * 检查值是否为 {@code Tensor} 类型。
   *
   * @return 如果值是 {@code Tensor} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isTensor() {
    return TYPE_CODE_TENSOR == this.mTypeCode;
  }

  /**
   * 检查值是否为 {@code Bool} 类型。
   *
   * @return 如果值是 {@code Bool} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isBool() {
    return TYPE_CODE_BOOL == this.mTypeCode;
  }

  /**
   * 检查值是否为 {@code Long} 类型。
   *
   * @return 如果值是 {@code Long} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isLong() {
    return TYPE_CODE_LONG == this.mTypeCode;
  }

  /**
   * 检查值是否为 {@code Double} 类型。
   *
   * @return 如果值是 {@code Double} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isDouble() {
    return TYPE_CODE_DOUBLE == this.mTypeCode;
  }

  /**
   * 检查值是否为 {@code String} 类型。
   *
   * @return 如果值是 {@code String} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isString() {
    return TYPE_CODE_STRING == this.mTypeCode;
  }

  /**
   * 检查值是否为 {@code Tuple} 类型。
   *
   * @return 如果值是 {@code Tuple} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isTuple() {
    return TYPE_CODE_TUPLE == this.mTypeCode;
  }

  /**
   * 检查值是否为 {@code BoolList} 类型。
   *
   * @return 如果值是 {@code BoolList} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isBoolList() {

    return TYPE_CODE_BOOL_LIST == this.mTypeCode;
  }

  /**
   * 检查值是否为 {@code LongList} 类型。
   *
   * @return 如果值是 {@code LongList} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isLongList() {
    return TYPE_CODE_LONG_LIST == this.mTypeCode;
  }

  /**
   * 检查值是否为 {@code DoubleList} 类型。
   *
   * @return 如果值是 {@code DoubleList} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isDoubleList() {
    return TYPE_CODE_DOUBLE_LIST == this.mTypeCode;
  }

  /**
   * 检查值是否为 {@code TensorList} 类型。
   *
   * @return 如果值是 {@code TensorList} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isTensorList() {
    return TYPE_CODE_TENSOR_LIST == this.mTypeCode;
  }

  /**
   * 检查值是否为 {@code List} 类型。
   *
   * @return 如果值是 {@code List} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isList() {
    return TYPE_CODE_LIST == this.mTypeCode;
  }

  /**
   * 检查值是否为 {@code DictStringKey} 类型。
   *
   * @return 如果值是 {@code DictStringKey} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isDictStringKey() {
    return TYPE_CODE_DICT_STRING_KEY == this.mTypeCode;
  }

  /**
   * 检查值是否为 {@code DictLongKey} 类型。
   *
   * @return 如果值是 {@code DictLongKey} 类型则返回 {@code true}，否则返回 {@code false}
   */
  @DoNotStrip
  public boolean isDictLongKey() {
    return TYPE_CODE_DICT_LONG_KEY == this.mTypeCode;
  }

  // 返回类型名称数组
  public String[] getTypeNames() {
    return TYPE_NAMES;
  }

  /**
   * 返回值的类型代码。
   *
   * @return 值的类型代码
   */
  @DoNotStrip
  public int getTypeCode() {
    return mTypeCode;
  }

  /**
   * 返回值的数据对象引用。
   *
   * @return 数据对象引用
   */
  @DoNotStrip
  public Object getData() {
    return mData;
  }
}
  // 检查当前 IValue 是否代表布尔类型列表
  return TYPE_CODE_BOOL_LIST == this.mTypeCode;
}

@DoNotStrip
public boolean isLongList() {
  // 检查当前 IValue 是否代表长整型列表
  return TYPE_CODE_LONG_LIST == this.mTypeCode;
}

@DoNotStrip
public boolean isDoubleList() {
  // 检查当前 IValue 是否代表双精度浮点数列表
  return TYPE_CODE_DOUBLE_LIST == this.mTypeCode;
}

@DoNotStrip
public boolean isTensorList() {
  // 检查当前 IValue 是否代表张量列表
  return TYPE_CODE_TENSOR_LIST == this.mTypeCode;
}

@DoNotStrip
public boolean isList() {
  // 检查当前 IValue 是否代表通用列表
  return TYPE_CODE_LIST == this.mTypeCode;
}

@DoNotStrip
public boolean isDictStringKey() {
  // 检查当前 IValue 是否代表字符串键字典
  return TYPE_CODE_DICT_STRING_KEY == this.mTypeCode;
}

@DoNotStrip
public boolean isDictLongKey() {
  // 检查当前 IValue 是否代表长整型键字典
  return TYPE_CODE_DICT_LONG_KEY == this.mTypeCode;
}

/** 创建一个新的 {@code IValue}，其类型为 {@code Optional}，不包含值。 */
@DoNotStrip
public static IValue optionalNull() {
  // 创建一个类型为 {@code Optional} 的新 {@code IValue}，其值为 null
  return new IValue(TYPE_CODE_NULL);
}

/** 根据给定的张量创建一个新的 {@code IValue}。 */
@DoNotStrip
public static IValue from(Tensor tensor) {
  // 创建一个类型为 {@code Tensor} 的新 {@code IValue}，并将张量设置为其数据
  final IValue iv = new IValue(TYPE_CODE_TENSOR);
  iv.mData = tensor;
  return iv;
}

/** 根据给定的布尔值创建一个新的 {@code IValue}。 */
@DoNotStrip
public static IValue from(boolean value) {
  // 创建一个类型为 {@code bool} 的新 {@code IValue}，并将布尔值设置为其数据
  final IValue iv = new IValue(TYPE_CODE_BOOL);
  iv.mData = value;
  return iv;
}

/** 根据给定的长整型值创建一个新的 {@code IValue}。 */
@DoNotStrip
public static IValue from(long value) {
  // 创建一个类型为 {@code int} 的新 {@code IValue}，并将长整型值设置为其数据
  final IValue iv = new IValue(TYPE_CODE_LONG);
  iv.mData = value;
  return iv;
}

/** 根据给定的双精度浮点数值创建一个新的 {@code IValue}。 */
@DoNotStrip
public static IValue from(double value) {
  // 创建一个类型为 {@code float} 的新 {@code IValue}，并将双精度浮点数值设置为其数据
  final IValue iv = new IValue(TYPE_CODE_DOUBLE);
  iv.mData = value;
  return iv;
}

/** 根据给定的字符串值创建一个新的 {@code IValue}
    // 获取数组的长度
    final int size = array.length;
    // 如果数组长度大于0
    if (size > 0) {
      // 获取数组第一个元素的类型码
      final int typeCode0 = array[0].mTypeCode;
      // 遍历数组从第二个元素到最后一个元素
      for (int i = 1; i < size; i++) {
        // 如果当前元素的类型码与第一个元素的类型码不同
        if (typeCode0 != array[i].mTypeCode) {
          // 抛出异常，要求列表必须包含相同类型的项
          throw new IllegalArgumentException("List must contain items of the same type");
        }
      }
    }

    // 创建一个新的 IValue 对象，类型为 TYPE_CODE_LIST
    final IValue iv = new IValue(TYPE_CODE_LIST);
    // 将数组设置为 IValue 对象的数据
    iv.mData = array;
    // 返回该 IValue 对象
    return iv;
  }

/** Creates a new {@code IValue} of type {@code Tuple[T0, T1, ...]}. */
@DoNotStrip
public static IValue tupleFrom(IValue... array) {
    // 创建一个新的 IValue 对象，类型为 TYPE_CODE_TUPLE
    final IValue iv = new IValue(TYPE_CODE_TUPLE);
    // 将数组设置为 IValue 对象的数据
    iv.mData = array;
    // 返回该 IValue 对象
    return iv;
}

/** Creates a new {@code IValue} of type {@code Dict[str, V]}. */
@DoNotStrip
public static IValue dictStringKeyFrom(Map<String, IValue> map) {
    // 创建一个新的 IValue 对象，类型为 TYPE_CODE_DICT_STRING_KEY
    final IValue iv = new IValue(TYPE_CODE_DICT_STRING_KEY);
    // 将 map 设置为 IValue 对象的数据
    iv.mData = map;
    // 返回该 IValue 对象
    return iv;
}

/** Creates a new {@code IValue} of type {@code Dict[int, V]}. */
@DoNotStrip
public static IValue dictLongKeyFrom(Map<Long, IValue> map) {
    // 创建一个新的 IValue 对象，类型为 TYPE_CODE_DICT_LONG_KEY
    final IValue iv = new IValue(TYPE_CODE_DICT_LONG_KEY);
    // 将 map 设置为 IValue 对象的数据
    iv.mData = map;
    // 返回该 IValue 对象
    return iv;
}

@DoNotStrip
public Tensor toTensor() {
    // 检查当前对象的类型码是否为 TYPE_CODE_TENSOR
    preconditionType(TYPE_CODE_TENSOR, mTypeCode);
    // 返回 mData 强制转换为 Tensor 类型
    return (Tensor) mData;
}

@DoNotStrip
public boolean toBool() {
    // 检查当前对象的类型码是否为 TYPE_CODE_BOOL
    preconditionType(TYPE_CODE_BOOL, mTypeCode);
    // 返回 mData 强制转换为 boolean 类型
    return (boolean) mData;
}

@DoNotStrip
public long toLong() {
    // 检查当前对象的类型码是否为 TYPE_CODE_LONG
    preconditionType(TYPE_CODE_LONG, mTypeCode);
    // 返回 mData 强制转换为 long 类型
    return (long) mData;
}

@DoNotStrip
public double toDouble() {
    // 检查当前对象的类型码是否为 TYPE_CODE_DOUBLE
    preconditionType(TYPE_CODE_DOUBLE, mTypeCode);
    // 返回 mData 强制转换为 double 类型
    return (double) mData;
}

@DoNotStrip
public String toStr() {
    // 检查当前对象的类型码是否为 TYPE_CODE_STRING
    preconditionType(TYPE_CODE_STRING, mTypeCode);
    // 返回 mData 强制转换为 String 类型
    return (String) mData;
}

@DoNotStrip
public boolean[] toBoolList() {
    // 检查当前对象的类型码是否为 TYPE_CODE_BOOL_LIST
    preconditionType(TYPE_CODE_BOOL_LIST, mTypeCode);
    // 返回 mData 强制转换为 boolean[] 类型
    return (boolean[]) mData;
}

@DoNotStrip
public long[] toLongList() {
    // 检查当前对象的类型码是否为 TYPE_CODE_LONG_LIST
    preconditionType(TYPE_CODE_LONG_LIST, mTypeCode);
    // 返回 mData 强制转换为 long[] 类型
    return (long[]) mData;
}

@DoNotStrip
public double[] toDoubleList() {
    // 检查当前对象的类型码是否为 TYPE_CODE_DOUBLE_LIST
    preconditionType(TYPE_CODE_DOUBLE_LIST, mTypeCode);
    // 返回 mData 强制转换为 double[] 类型
    return (double[]) mData;
}

@DoNotStrip
public Tensor[] toTensorList() {
    // 检查当前对象的类型码是否为 TYPE_CODE_TENSOR_LIST
    preconditionType(TYPE_CODE_TENSOR_LIST, mTypeCode);
    // 返回 mData 强制转换为 Tensor[] 类型
    return (Tensor[]) mData;
}

@DoNotStrip
public IValue[] toList() {
    // 检查当前对象的类型码是否为 TYPE_CODE_LIST
    preconditionType(TYPE_CODE_LIST, mTypeCode);
    // 返回 mData 强制转换为 IValue[] 类型
    return (IValue[]) mData;
}

@DoNotStrip
public IValue[] toTuple() {
    // 检查当前对象的类型码是否为 TYPE_CODE_TUPLE
    preconditionType(TYPE_CODE_TUPLE, mTypeCode);
    // 返回 mData 强制转换为 IValue[] 类型
    return (IValue[]) mData;
}

@DoNotStrip
public Map<String, IValue> toDictStringKey() {
    // 检查当前对象的类型码是否为 TYPE_CODE_DICT_STRING_KEY
    preconditionType(TYPE_CODE_DICT_STRING_KEY, mTypeCode);
    // 返回 mData 强制转换为 Map<String, IValue> 类型
    return (Map<String, IValue>) mData;
}

@DoNotStrip
public Map<Long, IValue> toDictLongKey() {
    // 检查当前对象的类型码是否为 TYPE_CODE_DICT_LONG_KEY
    preconditionType(TYPE_CODE_DICT_LONG_KEY, mTypeCode);
    // 返回 mData 强制转换为 Map<Long, IValue> 类型
    return (Map<Long, IValue>) mData;
}

private void preconditionType(int typeCodeExpected, int typeCode) {
    // 检查当前对象的类型码是否与预期类型码一致，如果不一致则抛出异常
    if (typeCode != typeCodeExpected) {
      // 如果传入的 typeCode 和期望的 typeCode 不相等，则抛出异常
      throw new IllegalStateException(
          String.format(
              Locale.US,
              "Expected IValue type %s, actual type %s",
              getTypeName(typeCodeExpected),  // 获取期望的类型名称
              getTypeName(typeCode)));       // 获取实际传入的类型名称
    }
  }

  private String getTypeName(int typeCode) {
    // 根据 typeCode 获取类型名称，如果 typeCode 超出范围，则返回 "Unknown"
    return typeCode >= 0 && typeCode < TYPE_NAMES.length ? TYPE_NAMES[typeCode] : "Unknown";
  }
}



# 这是一个单独的右花括号 '}'，用于结束一个代码块或数据结构的定义。
```