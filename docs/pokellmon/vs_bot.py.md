# `.\PokeLLMon\vs_bot.py`

```
# 导入必要的库
import asyncio
import time
from tqdm import tqdm
import numpy as np
from poke_env import AccountConfiguration, ShowdownServerConfiguration
import os
import pickle as pkl
import argparse

# 从 poke_env.player 模块中导入 LLMPlayer 和 SimpleHeuristicsPlayer 类
from poke_env.player import LLMPlayer, SimpleHeuristicsPlayer

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
# 添加命令行参数
parser.add_argument("--backend", type=str, default="gpt-4-0125-preview", choices=["gpt-3.5-turbo-0125", "gpt-4-1106-preview", "gpt-4-0125-preview"])
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--prompt_algo", default="io", choices=["io", "sc", "cot", "tot"])
parser.add_argument("--log_dir", type=str, default="./battle_log/pokellmon_vs_bot")
# 解析命令行参数
args = parser.parse_args()

# 异步函数，主要逻辑在其中实现
async def main():

    # 创建 SimpleHeuristicsPlayer 对象
    heuristic_player = SimpleHeuristicsPlayer(battle_format="gen8randombattle")

    # 确保日志目录存在
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建 LLMPlayer 对象
    llm_player = LLMPlayer(battle_format="gen8randombattle",
                           api_key="Your_openai_api_key",
                           backend=args.backend,
                           temperature=args.temperature,
                           prompt_algo=args.prompt_algo,
                           log_dir=args.log_dir,
                           account_configuration=AccountConfiguration("Your_account", "Your_password"),
                           save_replays=args.log_dir
                           )

    # 禁用动态最大化功能
    heuristic_player._dynamax_disable = True
    llm_player._dynamax_disable = True

    # 进行五场对战
    for i in tqdm(range(5)):
        # 随机选择对手
        x = np.random.randint(0, 100)
        if x > 50:
            # 与 LLMPlayer 对战
            await heuristic_player.battle_against(llm_player, n_battles=1)
        else:
            # 与 SimpleHeuristicsPlayer 对战
            await llm_player.battle_against(heuristic_player, n_battles=1)
        # 保存对战记录
        for battle_id, battle in llm_player.battles.items():
            with open(f"{args.log_dir}/{battle_id}.pkl", "wb") as f:
                pkl.dump(battle, f)


if __name__ == "__main__":
    # 获取当前事件循环并运行直到完成 main() 函数的执行
    asyncio.get_event_loop().run_until_complete(main())
```