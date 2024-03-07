# `.\PokeLLMon\vs_ladder_player.py`

```
# 导入必要的库
import asyncio
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import LLMPlayer
import pickle as pkl
from tqdm import tqdm
import argparse
import os

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, default="gpt-4-0125-preview", choices=["gpt-3.5-turbo-0125", "gpt-4-1106-preview", "gpt-4-0125-preview"])
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--prompt_algo", default="io", choices=["io", "sc", "cot", "tot"])
parser.add_argument("--log_dir", type=str, default="./battle_log/pokellmon_vs_ladder_player")
args = parser.parse_args()

# 异步函数，用于执行主要逻辑
async def main():

    # 确保日志目录存在
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建 LLMPlayer 实例
    llm_player = LLMPlayer(battle_format="gen8randombattle",
                           api_key="Your_openai_api_key",
                           backend=args.backend,
                           temperature=args.temperature,
                           prompt_algo=args.prompt_algo,
                           log_dir=args.log_dir,
                           account_configuration=AccountConfiguration("Your_account", "Your_password"),
                           server_configuration=ShowdownServerConfiguration,
                           save_replays=args.log_dir
                           )

    # 在 ladder 上进行 5 场比赛
    for i in tqdm(range(1)):
        try:
            # 在 ladder 上进行比赛
            await llm_player.ladder(1)
            # 保存每场比赛的数据
            for battle_id, battle in llm_player.battles.items():
                with open(f"{args.log_dir}/{battle_id}.pkl", "wb") as f:
                    pkl.dump(battle, f)
        except:
            continue

# 如果作为独立脚本运行，则执行主函数
if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
```