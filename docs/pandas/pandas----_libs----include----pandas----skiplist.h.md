# `D:\src\scipysrc\pandas\pandas\_libs\include\pandas\skiplist.h`

```
/*
Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.

Flexibly-sized, index-able skiplist data structure for maintaining a sorted
list of values

Port of Wes McKinney's Cython version of Raymond Hettinger's original pure
Python recipe (https://rhettinger.wordpress.com/2010/02/06/lost-knowledge/)
*/

#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义一个静态内联函数，返回一个特定的浮点数表示 NaN
static inline float __skiplist_nanf(void) {
  const union {
    int __i;
    float __f;
  } __bint = {0x7fc00000UL};
  return __bint.__f;
}
// 定义宏 PANDAS_NAN，表示 NaN 的双精度浮点数值
#define PANDAS_NAN ((double)__skiplist_nanf())

// 定义一个静态内联函数，计算以2为底的对数
static inline double Log2(double val) { return log(val) / log(2.); }

// 定义节点结构体 node_t
typedef struct node_t node_t;

struct node_t {
  node_t **next;   // 指向下一层节点的指针数组
  int *width;      // 各层节点之间的跨度数组
  double value;    // 节点存储的值
  int is_nil;      // 标志节点是否为NIL节点
  int levels;      // 节点的层数
  int ref_count;   // 节点的引用计数
};

// 定义跳表结构体 skiplist_t
typedef struct {
  node_t *head;     // 指向跳表头节点的指针
  node_t **tmp_chain;  // 临时链表节点数组
  int *tmp_steps;   // 临时步长数组
  int size;         // 跳表的大小（节点数）
  int maxlevels;    // 跳表的最大层数
} skiplist_t;

// 返回一个随机浮点数，用于生成跳表节点的层数
static inline double urand(void) {
  return ((double)rand() + 1) / ((double)RAND_MAX + 2);
}

// 返回两个整数中较小的那个
static inline int int_min(int a, int b) { return a < b ? a : b; }

// 初始化一个节点，分配内存并设置初始值
static inline node_t *node_init(double value, int levels) {
  node_t *result;
  result = (node_t *)malloc(sizeof(node_t));
  if (result) {
    result->value = value;
    result->levels = levels;
    result->is_nil = 0;
    result->ref_count = 0;
    // 分配层数对应的指针数组和宽度数组
    result->next = (node_t **)malloc(levels * sizeof(node_t *));
    result->width = (int *)malloc(levels * sizeof(int));
    // 如果分配失败且层数不为0，则释放已分配的内存并返回NULL
    if (!(result->next && result->width) && (levels != 0)) {
      free(result->next);
      free(result->width);
      free(result);
      return NULL;
    }
  }
  return result;
}

// 增加节点的引用计数
static inline void node_incref(node_t *node) { ++(node->ref_count); }

// 减少节点的引用计数
static inline void node_decref(node_t *node) { --(node->ref_count); }

// 递归销毁节点及其子节点
static void node_destroy(node_t *node) {
  int i;
  if (node) {
    // 如果节点的引用计数小于等于1，则递归销毁每一层的节点，并释放分配的内存
    if (node->ref_count <= 1) {
      for (i = 0; i < node->levels; ++i) {
        node_destroy(node->next[i]);
      }
      free(node->next);
      free(node->width);
      free(node);
    } else {
      // 否则，减少节点的引用计数
      node_decref(node);
    }
  }
}

// 销毁整个跳表结构，包括头节点及其相关的临时链表和步长数组
static inline void skiplist_destroy(skiplist_t *skp) {
  if (skp) {
    node_destroy(skp->head);
    free(skp->tmp_steps);
    free(skp->tmp_chain);
    free(skp);
  }
}

// 初始化跳表结构，分配内存并设置初始值
static inline skiplist_t *skiplist_init(int expected_size) {
  skiplist_t *result;
  node_t *NIL, *head;
  int maxlevels, i;

  // 计算跳表的最大层数
  maxlevels = 1 + Log2((double)expected_size);
  result = (skiplist_t *)malloc(sizeof(skiplist_t));
  if (!result) {
    // 如果分配失败，则返回NULL

    return NULL;
  }

  // 分配头节点
  head = node_init(PANDAS_NAN, maxlevels);
  if (!head) {
    // 如果分配失败，则释放已分配的内存并返回NULL
    free(result);
    return NULL;
  }

  // 初始化跳表结构的各项属性
  result->head = head;
  result->size = 0;
  result->maxlevels = maxlevels;
  result->tmp_chain = NULL;
  result->tmp_steps = (int *)malloc(maxlevels * sizeof(int));
  if (!result->tmp_steps) {
    // 如果分配失败，则释放已分配的内存并返回NULL
    free(head);
    free(result);
    return NULL;
  }

  // 返回初始化后的跳表结构
  return result;
}
    // 如果 result 指针为 NULL，返回 NULL
    return NULL;
  }
  // 分配 tmp_chain 数组，用于存储节点指针数组
  result->tmp_chain = (node_t **)malloc(maxlevels * sizeof(node_t *));
  // 分配 tmp_steps 数组，用于存储每层的步数
  result->tmp_steps = (int *)malloc(maxlevels * sizeof(int));
  // 设置最大层数和当前跳表大小
  result->maxlevels = maxlevels;
  result->size = 0;

  // 初始化头节点，并设置头指针
  head = result->head = node_init(PANDAS_NAN, maxlevels);
  // 创建一个 NIL 节点，表示跳表末尾
  NIL = node_init(0.0, 0);

  // 检查内存分配是否成功，若有任何一个失败，则释放资源并返回 NULL
  if (!(result->tmp_chain && result->tmp_steps && result->head && NIL)) {
    skiplist_destroy(result);  // 销毁跳表对象
    node_destroy(NIL);         // 销毁 NIL 节点
    return NULL;               // 返回 NULL 表示失败
  }

  // 增加头节点的引用计数
  node_incref(head);

  // 将 NIL 标记为末尾节点
  NIL->is_nil = 1;

  // 初始化每层头节点的指向和宽度，并增加 NIL 节点的引用计数
  for (i = 0; i < maxlevels; ++i) {
    head->next[i] = NIL;
    head->width[i] = 1;
    node_incref(NIL);
  }

  // 返回跳表对象 result
  return result;
// 关闭函数定义
}

// 比较节点值与给定值的大小关系，返回-1表示节点值大于给定值，0表示相等，1表示小于
static inline int _node_cmp(node_t *node, double value) {
  // 如果节点为空或者节点值大于给定值，则返回-1
  if (node->is_nil || node->value > value) {
    return -1;
  } else if (node->value < value) { // 如果节点值小于给定值，则返回1
    return 1;
  } else { // 否则返回0，表示相等
    return 0;
  }
}

// 获取跳表中第i个元素的值，同时返回操作是否成功的标志
static inline double skiplist_get(skiplist_t *skp, int i, int *ret) {
  node_t *node;
  int level;

  // 如果索引i超出范围，返回0并设置操作失败标志
  if (i < 0 || i >= skp->size) {
    *ret = 0;
    return 0;
  }

  // 初始化节点为头节点，并逐层查找目标节点
  node = skp->head;
  ++i;
  for (level = skp->maxlevels - 1; level >= 0; --level) {
    while (node->width[level] <= i) {
      i -= node->width[level];
      node = node->next[level];
    }
  }

  // 设置操作成功标志并返回目标节点的值
  *ret = 1;
  return node->value;
}

// 返回具有给定值的所有元素的最低排名，与skiplist_insert返回的最高排名相对应
static inline int skiplist_min_rank(skiplist_t *skp, double value) {
  node_t *node;
  int level, rank = 0;

  // 初始化节点为头节点，并逐层查找目标值的位置
  node = skp->head;
  for (level = skp->maxlevels - 1; level >= 0; --level) {
    while (_node_cmp(node->next[level], value) > 0) {
      rank += node->width[level];
      node = node->next[level];
    }
  }

  // 返回最终的排名
  return rank + 1;
}

// 插入元素并返回其排名，当有重复元素时，返回最高排名，类似于pandas.DataFrame.rank的'max'方法
static inline int skiplist_insert(skiplist_t *skp, double value) {
  node_t *node, *prevnode, *newnode, *next_at_level;
  int *steps_at_level;
  int size, steps, level, rank = 0;
  node_t **chain;

  // 获取临时链表和步数数组
  chain = skp->tmp_chain;
  steps_at_level = skp->tmp_steps;
  memset(steps_at_level, 0, skp->maxlevels * sizeof(int));

  // 初始化节点为头节点
  node = skp->head;

  // 逐层查找插入位置，并计算每层的步数和排名
  for (level = skp->maxlevels - 1; level >= 0; --level) {
    next_at_level = node->next[level];
    while (_node_cmp(next_at_level, value) >= 0) {
      steps_at_level[level] += node->width[level];
      rank += node->width[level];
      node = next_at_level;
      next_at_level = node->next[level];
    }
    chain[level] = node;
  }

  // 随机确定新节点的层数
  size = int_min(skp->maxlevels, 1 - ((int)Log2(urand())));

  // 初始化新节点
  newnode = node_init(value, size);
  if (!newnode) {
    return -1; // 分配新节点失败，返回-1表示失败
  }
  steps = 0;

  // 在每层链表中插入新节点，并更新宽度
  for (level = 0; level < size; ++level) {
    prevnode = chain[level];
    newnode->next[level] = prevnode->next[level];
    prevnode->next[level] = newnode;
    node_incref(newnode); // 增加节点的引用计数
    newnode->width[level] = prevnode->width[level] - steps;
    prevnode->width[level] = steps + 1;
    steps += steps_at_level[level];
  }

  // 更新上层链表的宽度
  for (level = size; level < skp->maxlevels; ++level) {
    chain[level]->width[level] += 1;
  }

  // 增加跳表的大小
  ++(skp->size);

  // 返回插入节点的排名
  return rank + 1;
}

// 从跳表中删除具有给定值的元素
static inline int skiplist_remove(skiplist_t *skp, double value) {
  int level, size;
  node_t *node, *prevnode, *tmpnode, *next_at_level;
  node_t **chain;

  // 获取临时链表和节点
  chain = skp->tmp_chain;
  node = skp->head;

  // 逐层查找并删除具有给定值的节点
  for (level = skp->maxlevels - 1; level >= 0; --level) {
    next_at_level = node->next[level];
    while (_node_cmp(next_at_level, value) > 0) {
      node = next_at_level;
      next_at_level = node->next[level];
    }
    chain[level] = node;
  }
    while (_node_cmp(next_at_level, value) > 0) {
      // 在跳表中找到合适的位置，使得 next_at_level 的节点值大于 value
      node = next_at_level;
      next_at_level = node->next[level];
    }
    // 将每个层级的链表节点保存在 chain 数组中，以便后续操作
    chain[level] = node;
  }

  // 验证是否找到要删除的节点
  if (value != chain[0]->next[0]->value) {
    return 0;  // 没有找到要删除的节点，返回 0
  }

  // 获取节点的层数
  size = chain[0]->next[0]->levels;

  // 逐层删除节点
  for (level = 0; level < size; ++level) {
    prevnode = chain[level];
    tmpnode = prevnode->next[level];

    // 更新宽度信息
    prevnode->width[level] += tmpnode->width[level] - 1;
    prevnode->next[level] = tmpnode->next[level];

    // 将 tmpnode 从跳表中断开
    tmpnode->next[level] = NULL;
    node_destroy(tmpnode); // 减少引用计数或释放内存
  }

  // 更新删除节点后每个层级的宽度信息
  for (level = size; level < skp->maxlevels; ++level) {
    --(chain[level]->width[level]);
  }

  // 更新跳表的节点数量
  --(skp->size);

  return 1;  // 删除操作成功，返回 1
}



# 这是一个单独的大括号，可能用于结束一个代码块或函数定义。
```