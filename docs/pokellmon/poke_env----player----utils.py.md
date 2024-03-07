# `.\PokeLLMon\poke_env\player\utils.py`

```py
"""This module contains utility functions and objects related to Player classes.
"""

# 导入必要的模块和对象
import asyncio
import math
from concurrent.futures import Future
from typing import Dict, List, Optional, Tuple

from poke_env.concurrency import POKE_LOOP
from poke_env.data import to_id_str
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer

# 定义不同玩家类型的评估等级
_EVALUATION_RATINGS = {
    RandomPlayer: 1,
    MaxBasePowerPlayer: 7.665994,
    SimpleHeuristicsPlayer: 128.757145,
}

# 后台交叉评估函数，返回一个 Future 对象
def background_cross_evaluate(
    players: List[Player], n_challenges: int
) -> "Future[Dict[str, Dict[str, Optional[float]]]]":
    return asyncio.run_coroutine_threadsafe(
        cross_evaluate(players, n_challenges), POKE_LOOP
    )

# 异步交叉评估函数
async def cross_evaluate(
    players: List[Player], n_challenges: int
) -> Dict[str, Dict[str, Optional[float]]]:
    # 初始化结果字典
    results: Dict[str, Dict[str, Optional[float]]] = {
        p_1.username: {p_2.username: None for p_2 in players} for p_1 in players
    }
    # 遍历玩家列表，进行交叉评估
    for i, p_1 in enumerate(players):
        for j, p_2 in enumerate(players):
            if j <= i:
                continue
            # 并发发送挑战和接受挑战
            await asyncio.gather(
                p_1.send_challenges(
                    opponent=to_id_str(p_2.username),
                    n_challenges=n_challenges,
                    to_wait=p_2.ps_client.logged_in,
                ),
                p_2.accept_challenges(
                    opponent=to_id_str(p_1.username),
                    n_challenges=n_challenges,
                    packed_team=p_2.next_team,
                ),
            )
            # 更新结果字典中的胜率信息
            results[p_1.username][p_2.username] = p_1.win_rate
            results[p_2.username][p_1.username] = p_2.win_rate

            # 重置战斗状态
            p_1.reset_battles()
            p_2.reset_battles()
    return results

# 从结果中估计实力函数
def _estimate_strength_from_results(
    number_of_games: int, number_of_wins: int, opponent_rating: float
# 估计玩家实力基于游戏结果和对手评分
def evaluate_player(
    number_of_games: int,  # 游戏评估的数量
    number_of_wins: int,  # 赢得的评估游戏数量
    opponent_rating: float,  # 对手的评分
) -> Tuple[float, Tuple[float, float]]:  # 返回估计的玩家实力和95%置信区间的元组

    n, p = number_of_games, number_of_wins / number_of_games  # 计算游戏数量和胜率
    q = 1 - p  # 计算失败率

    if n * p * q < 9:  # 如果无法应用二项分布的正态近似
        raise ValueError(
            "The results obtained in evaluate_player are too extreme to obtain an "
            "accurate player evaluation. You can try to solve this issue by increasing"
            " the total number of battles. Obtained results: %d victories out of %d"
            " games." % (p * n, n)
        )

    estimate = opponent_rating * p / q  # 估计玩家实力
    error = (
        math.sqrt(n * p * q) / n * 1.96
    )  # 95%置信区间的正态分布

    lower_bound = max(0, p - error)  # 下限
    lower_bound = opponent_rating * lower_bound / (1 - lower_bound)  # 下限的评估

    higher_bound = min(1, p + error)  # 上限

    if higher_bound == 1:  # 如果上限为1
        higher_bound = math.inf  # 上限为正无穷大
    else:
        higher_bound = opponent_rating * higher_bound / (1 - higher_bound)  # 上限的评估

    return estimate, (lower_bound, higher_bound)  # 返回估计值和置信区间的元组


# 后台评估玩家
def background_evaluate_player(
    player: Player,  # 玩家对象
    n_battles: int = 1000,  # 战斗数量默认为1000
    n_placement_battles: int = 30,  # 放置战斗数量默认为30
) -> "Future[Tuple[float, Tuple[float, float]]]":  # 返回Future对象，包含估计值和置信区间的元组

    return asyncio.run_coroutine_threadsafe(
        evaluate_player(player, n_battles, n_placement_battles), POKE_LOOP
    )  # 在POKE_LOOP中异步运行评估玩家函数


# 异步评估玩家
async def evaluate_player(
    player: Player,  # 玩家对象
    n_battles: int = 1000,  # 战斗数量默认为1000
    n_placement_battles: int = 30,  # 放置战斗数量默认为30
# 估算玩家实力的函数
def estimate_player_strength(player: Player, n_battles: int, n_placement_battles: int) -> Tuple[float, Tuple[float, float]]:
    """Estimate player strength.

    This functions calculates an estimate of a player's strength, measured as its
    expected performance against a random opponent in a gen 8 random battle. The
    returned number can be interpreted as follows: a strength of k means that the
    probability of the player winning a gen 8 random battle against a random player is k
    times higher than the probability of the random player winning.

    The function returns a tuple containing the best guess based on the played games
    as well as a tuple describing a 95% confidence interval for that estimated strength.

    The actual evaluation can be performed against any baseline player for which an
    accurate strength estimate is available. This baseline is determined at the start of
    the process, by playing a limited number of placement battles and choosing the
    opponent closest to the player in terms of performance.

    :param player: The player to evaluate.
    :type player: Player
    :param n_battles: The total number of battle to perform, including placement
        battles.
    :type n_battles: int
    :param n_placement_battles: Number of placement battles to perform per baseline
        player.
    :type n_placement_battles: int
    :raises: ValueError if the results are too extreme to be interpreted.
    :raises: AssertionError if the player is not configured to play gen8battles or the
        selected number of games to play it too small.
    :return: A tuple containing the estimated player strength and a 95% confidence
        interval
    :rtype: tuple of float and tuple of floats
    """
    # 检查输入
    assert player.format == "gen8randombattle", (
        "Player %s can not be evaluated as its current format (%s) is not "
        "gen8randombattle." % (player, player.format)
    )
    # 如果放置战斗的数量乘以评估等级的数量大于总战斗数量的一半，则进行警告
    if n_placement_battles * len(_EVALUATION_RATINGS) > n_battles // 2:
        player.logger.warning(
            "Number of placement battles reduced from %d to %d due to limited number of"
            " battles (%d). A more accurate evaluation can be performed by increasing "
            "the total number of players.",
            n_placement_battles,
            n_battles // len(_EVALUATION_RATINGS) // 2,
            n_battles,
        )
        # 将放置战斗数量减少到总战斗数量的一半除以评估等级的数量
        n_placement_battles = n_battles // len(_EVALUATION_RATINGS) // 2

    # 断言放置战斗数量大于0，否则抛出异常
    assert (
        n_placement_battles > 0
    ), "Not enough battles to perform placement battles. Please increase the number of "
    "battles to perform to evaluate the player."

    # 初始化放置战斗
    baselines = [p(max_concurrent_battles=n_battles) for p in _EVALUATION_RATINGS]

    # 对每个基准玩家进行放置战斗
    for p in baselines:
        await p.battle_against(player, n_placement_battles)

    # 选择最佳对手进行评估
    best_opp = min(
        baselines, key=lambda p: (abs(p.win_rate - 0.5), -_EVALUATION_RATINGS[type(p)])
    )

    # 执行主要评估
    remaining_battles = n_battles - len(_EVALUATION_RATINGS) * n_placement_battles
    await best_opp.battle_against(player, remaining_battles)

    # 从结果中估计玩家的实力
    return _estimate_strength_from_results(
        best_opp.n_finished_battles,
        best_opp.n_lost_battles,
        _EVALUATION_RATINGS[type(best_opp)],
    )
```