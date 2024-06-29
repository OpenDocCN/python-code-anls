# `.\numpy\numpy\random\src\sfc64\sfc64.c`

```py
#include "sfc64.h"

/* 设置 sfc64 状态的种子值
 * 将提供的种子数组设置为状态的 s 数组的值，第四个元素固定为 1
 */
extern void sfc64_set_seed(sfc64_state *state, uint64_t *seed) {
  int i;

  state->s[0] = seed[0];
  state->s[1] = seed[1];
  state->s[2] = seed[2];
  state->s[3] = 1;

  // 执行12次 sfc64_next，初始设置
  for (i=0; i<12; i++) {
    (void)sfc64_next(state->s);
  }
}

/* 获取 sfc64 状态的当前值
 * 将状态的 s 数组的当前值复制到 state_arr 中，同时复制其他状态信息
 */
extern void sfc64_get_state(sfc64_state *state, uint64_t *state_arr, int *has_uint32,
                            uint32_t *uinteger) {
  int i;

  for (i=0; i<4; i++) {
    state_arr[i] = state->s[i];
  }
  has_uint32[0] = state->has_uint32;
  uinteger[0] = state->uinteger;
}

/* 设置 sfc64 状态的当前值
 * 将提供的 state_arr 设置为状态的 s 数组的值，同时更新其他状态信息
 */
extern void sfc64_set_state(sfc64_state *state, uint64_t *state_arr, int has_uint32,
                            uint32_t uinteger) {
  int i;

  for (i=0; i<4; i++) {
    state->s[i] = state_arr[i];
  }
  state->has_uint32 = has_uint32;
  state->uinteger = uinteger;
}
```