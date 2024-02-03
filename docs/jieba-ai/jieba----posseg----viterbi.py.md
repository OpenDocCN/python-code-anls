# `jieba\jieba\posseg\viterbi.py`

```
# 导入 sys 模块
import sys
# 导入 operator 模块
import operator
# 定义最小浮点数常量
MIN_FLOAT = -3.14e100
# 定义负无穷大常量
MIN_INF = float("-inf")

# 如果 Python 版本大于 2，则将 xrange 设置为 range
if sys.version_info[0] > 2:
    xrange = range

# 获取前 K 个概率最高的状态
def get_top_states(t_state_v, K=4):
    return sorted(t_state_v, key=t_state_v.__getitem__, reverse=True)[:K]

# 维特比算法实现
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # 存储表格
    mem_path = [{}]  # 存储路径
    all_states = trans_p.keys()
    
    # 初始化第一个观测值
    for y in states.get(obs[0], all_states):
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        mem_path[0][y] = ''
    
    # 逐步计算每个观测值的状态
    for t in xrange(1, len(obs)):
        V.append({})
        mem_path.append({})
        
        # 获取前一时刻概率最高的状态
        prev_states = [
            x for x in mem_path[t - 1].keys() if len(trans_p[x]) > 0]
        
        # 获取前一时刻状态可能转移到的下一时刻状态
        prev_states_expect_next = set(
            (y for x in prev_states for y in trans_p[x].keys()))
        
        # 获取当前观测值对应的状态
        obs_states = set(
            states.get(obs[t], all_states)) & prev_states_expect_next
        
        # 如果当前观测值没有对应的状态，则使用前一时刻状态可能转移到的状态或所有状态
        if not obs_states:
            obs_states = prev_states_expect_next if prev_states_expect_next else all_states
        
        # 计算当前观测值对应的状态的概率
        for y in obs_states:
            prob, state = max((V[t - 1][y0] + trans_p[y0].get(y, MIN_INF) +
                               emit_p[y].get(obs[t], MIN_FLOAT), y0) for y0 in prev_states)
            V[t][y] = prob
            mem_path[t][y] = state
    
    # 获取最终路径
    last = [(V[-1][y], y) for y in mem_path[-1].keys()]
    prob, state = max(last)

    # 回溯路径
    route = [None] * len(obs)
    i = len(obs) - 1
    while i >= 0:
        route[i] = state
        state = mem_path[i][state]
        i -= 1
    return (prob, route)
```