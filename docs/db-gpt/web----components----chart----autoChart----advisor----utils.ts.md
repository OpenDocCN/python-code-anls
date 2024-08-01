# `.\DB-GPT-src\web\components\chart\autoChart\advisor\utils.ts`

```py
// 导入从 lodash 中导入 isNull 函数
import { isNull } from 'lodash';
// 导入 Advice 类型定义从 @antv/ava 模块
import type { Advice } from '@antv/ava';

// 默认的 Advices 过滤函数，接受一个包含 advices 属性的对象作为参数
export function defaultAdvicesFilter(props: { advices: Advice[] }) {
  // 解构 props 参数，获取 advices 数组
  const { advices } = props;
  // 返回 advices 数组本身
  return advices;
}

// 比较函数 compare，接受两个参数 f1 和 f2
export const compare = (f1: any, f2: any) => {
  // 如果 f1.distinct 或者 f2.distinct 是 null，则执行以下条件判断
  if (isNull(f1.distinct) || isNull(f2.distinct)) {
    // 如果 f1.distinct 小于 f2.distinct，则返回 1
    if (f1.distinct! < f2!.distinct!) {
      return 1;
    }
    // 如果 f1.distinct 大于 f2.distinct，则返回 -1
    if (f1.distinct! > f2.distinct!) {
      return -1;
    }
    // 否则返回 0
    return 0;
  }
  // 如果 f1.distinct 和 f2.distinct 都不是 null，则返回 0
  return 0;
};

// 判断数组 array1 是否包含数组 array2 的所有元素
export function hasSubset(array1: any[], array2: any[]): boolean {
  return array2.every((e) => array1.includes(e));
}

// 判断数组 array1 是否与数组 array2 至少有一个相同的元素
export function intersects(array1: any[], array2: any[]): boolean {
  return array2.some((e) => array1.includes(e));
}

// 根据 lom 参数返回对应的编码类型
export function LOM2EncodingType(lom: string) {
  switch (lom) {
    case 'Nominal':
      return 'nominal';
    case 'Ordinal':
      return 'ordinal';
    case 'Interval':
      return 'quantitative';
    case 'Time':
      return 'temporal';
    case 'Continuous':
      return 'quantitative';
    case 'Discrete':
      return 'nominal';
    default:
      return 'nominal';
  }
}
```