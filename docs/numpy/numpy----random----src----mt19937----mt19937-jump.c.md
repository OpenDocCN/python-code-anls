# `.\numpy\numpy\random\src\mt19937\mt19937-jump.c`

```py
/* 包含头文件 "mt19937-jump.h" 和 "mt19937.h" */
#include "mt19937-jump.h"
#include "mt19937.h"

/* 定义一个函数：获取多项式 pf 的第 i 个系数 */
unsigned long get_coef(unsigned long *pf, unsigned int deg) {
  /* 检查多项式中第 i 个系数是否为 1 */
  if ((pf[deg >> 5] & (LSB << (deg & 0x1ful))) != 0)
    return (1);
  else
    return (0);
}

/* 定义一个函数：复制状态 */
void copy_state(mt19937_state *target_state, mt19937_state *state) {
  int i;

  /* 循环复制状态的关键数组和位置 */
  for (i = 0; i < N; i++)
    target_state->key[i] = state->key[i];

  target_state->pos = state->pos;
}

/* 定义一个函数：生成下一个状态 */
void gen_next(mt19937_state *state) {
  int num;
  unsigned long y;
  static unsigned long mag02[2] = {0x0ul, MATRIX_A};

  num = state->pos;
  if (num < N - M) {
    /* 计算下一个状态的关键数组中的值 */
    y = (state->key[num] & UPPER_MASK) | (state->key[num + 1] & LOWER_MASK);
    state->key[num] = state->key[num + M] ^ (y >> 1) ^ mag02[y % 2];
    state->pos++;
  } else if (num < N - 1) {
    y = (state->key[num] & UPPER_MASK) | (state->key[num + 1] & LOWER_MASK);
    state->key[num] = state->key[num + (M - N)] ^ (y >> 1) ^ mag02[y % 2];
    state->pos++;
  } else if (num == N - 1) {
    y = (state->key[N - 1] & UPPER_MASK) | (state->key[0] & LOWER_MASK);
    state->key[N - 1] = state->key[M - 1] ^ (y >> 1) ^ mag02[y % 2];
    state->pos = 0;
  }
}

/* 定义一个函数：添加状态 */
void add_state(mt19937_state *state1, mt19937_state *state2) {
  int i, pt1 = state1->pos, pt2 = state2->pos;

  if (pt2 - pt1 >= 0) {
    /* 将状态 state2 添加到状态 state1 */
    for (i = 0; i < N - pt2; i++)
      state1->key[i + pt1] ^= state2->key[i + pt2];
    for (; i < N - pt1; i++)
      state1->key[i + pt1] ^= state2->key[i + (pt2 - N)];
    for (; i < N; i++)
      state1->key[i + (pt1 - N)] ^= state2->key[i + (pt2 - N)];
  } else {
    for (i = 0; i < N - pt1; i++)
      state1->key[i + pt1] ^= state2->key[i + pt2];
    for (; i < N - pt2; i++)
      state1->key[i + (pt1 - N)] ^= state2->key[i + pt2];
    for (; i < N; i++)
      state1->key[i + (pt1 - N)] ^= state2->key[i + (pt2 - N)];
  }
}

/* 定义一个函数：使用标准 Horner 方法计算 pf(ss) */
void horner1(unsigned long *pf, mt19937_state *state) {
  int i = MEXP - 1;
  mt19937_state *temp;

  /* 分配内存并初始化 temp 为 mt19937_state 类型的指针 */
  temp = (mt19937_state *)calloc(1, sizeof(mt19937_state));

  /* 通过 Horner 方法计算多项式的值 */
  while (get_coef(pf, i) == 0)
    i--;

  if (i > 0) {
    copy_state(temp, state);
    gen_next(temp);
    i--;
    for (; i > 0; i--) {
      if (get_coef(pf, i) != 0)
        add_state(temp, state);
      else
        ;
      gen_next(temp);
    }
    if (get_coef(pf, 0) != 0)
      add_state(temp, state);
    else
      ;
  } else if (i == 0)
    copy_state(temp, state);
  else
    ;

  /* 将 temp 的状态复制到 state，释放 temp 的内存 */
  copy_state(state, temp);
  free(temp);
}

/* 定义一个函数：MT19937 跳跃状态 */
void mt19937_jump_state(mt19937_state *state) {
  unsigned long *pf;
  int i;

  /* 分配内存并初始化 pf 数组 */
  pf = (unsigned long *)calloc(P_SIZE, sizeof(unsigned long));
  for (i = 0; i<P_SIZE; i++) {
    pf[i] = poly_coef[i];
  }

  /* 如果状态的位置超过数组长度 N，则将位置设置为 0 */
  if (state->pos >= N) {
    state->pos = 0;
  }

  /* 使用 Horner 方法计算 pf(ss)，更新状态 */
  horner1(pf, state);

  /* 释放 pf 数组的内存 */
  free(pf);
}
```