# `.\PokeLLMon\poke_env\player\openai_api.py`

```py
"""This module defines a player class with the OpenAI API on the main thread.
For a black-box implementation consider using the module env_player.
"""
# 导入必要的模块
from __future__ import annotations

import asyncio
import copy
import random
import time
from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Tuple, Union

# 导入自定义模块
from gymnasium.core import ActType, Env, ObsType
from gymnasium.spaces import Discrete, Space

# 导入自定义模块
from poke_env.concurrency import POKE_LOOP, create_in_poke_loop
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
from poke_env.player.player import Player
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder.teambuilder import Teambuilder

# 定义一个异步队列类
class _AsyncQueue:
    def __init__(self, queue: asyncio.Queue[Any]):
        self.queue = queue

    # 异步获取队列中的元素
    async def async_get(self):
        return await self.queue.get()

    # 获取队列中的元素
    def get(self):
        res = asyncio.run_coroutine_threadsafe(self.queue.get(), POKE_LOOP)
        return res.result()

    # 异步向队列中放入元素
    async def async_put(self, item: Any):
        await self.queue.put(item)

    # 向队列中放入元素
    def put(self, item: Any):
        task = asyncio.run_coroutine_threadsafe(self.queue.put(item), POKE_LOOP)
        task.result()

    # 判断队列是否为空
    def empty(self):
        return self.queue.empty()

    # 阻塞直到队列中的所有元素都被处理
    def join(self):
        task = asyncio.run_coroutine_threadsafe(self.queue.join(), POKE_LOOP)
        task.result()

    # 异步等待队列中的所有元素都被处理
    async def async_join(self):
        await self.queue.join()

# 定义一个异步玩家类
class _AsyncPlayer(Generic[ObsType, ActType], Player):
    actions: _AsyncQueue
    observations: _AsyncQueue

    def __init__(
        self,
        user_funcs: OpenAIGymEnv[ObsType, ActType],
        username: str,
        **kwargs: Any,
    # 定义一个类，继承自AsyncPlayer类
    ):
        # 设置类名为username
        self.__class__.__name__ = username
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置类名为"_AsyncPlayer"
        self.__class__.__name__ = "_AsyncPlayer"
        # 初始化observations为一个异步队列
        self.observations = _AsyncQueue(create_in_poke_loop(asyncio.Queue, 1))
        # 初始化actions为一个异步队列
        self.actions = _AsyncQueue(create_in_poke_loop(asyncio.Queue, 1))
        # 初始化current_battle为None
        self.current_battle: Optional[AbstractBattle] = None
        # 初始化_user_funcs为user_funcs

    # 定义一个方法，用于选择移动
    def choose_move(self, battle: AbstractBattle) -> Awaitable[BattleOrder]:
        # 返回_env_move方法的结果
        return self._env_move(battle)

    # 定义一个异步方法，用于处理环境移动
    async def _env_move(self, battle: AbstractBattle) -> BattleOrder:
        # 如果当前战斗为空或已结束，则将当前战斗设置为传入的战斗
        if not self.current_battle or self.current_battle.finished:
            self.current_battle = battle
        # 如果当前战斗不等于传入的战斗，则抛出异常
        if not self.current_battle == battle:
            raise RuntimeError("Using different battles for queues")
        # 将战斗嵌入到用户函数中，并异步放入observations队列中
        battle_to_send = self._user_funcs.embed_battle(battle)
        await self.observations.async_put(battle_to_send)
        # 从actions队列中异步获取动作
        action = await self.actions.async_get()
        # 如果动作为-1，则返回放弃战斗的指令
        if action == -1:
            return ForfeitBattleOrder()
        # 将动作转换为移动指令并返回
        return self._user_funcs.action_to_move(action, battle)

    # 定义一个回调方法，用于处理战斗结束时的操作
    def _battle_finished_callback(self, battle: AbstractBattle):
        # 将战斗嵌入到用户函数中，并异步放入observations队列中
        to_put = self._user_funcs.embed_battle(battle)
        # 在POKE_LOOP中安全地运行异步放入操作
        asyncio.run_coroutine_threadsafe(self.observations.async_put(to_put), POKE_LOOP)
# 定义一个元类，继承自 ABC 类型
class _ABCMetaclass(type(ABC)):
    pass

# 定义一个元类，继承自 Env 类型
class _EnvMetaclass(type(Env)):
    pass

# 定义一个元类，继承自 _EnvMetaclass 和 _ABCMetaclass
class _OpenAIGymEnvMetaclass(_EnvMetaclass, _ABCMetaclass):
    pass

# 定义一个类 OpenAIGymEnv，继承自 Env[ObsType, ActType] 和 ABC 类型，使用 _OpenAIGymEnvMetaclass 元类
class OpenAIGymEnv(
    Env[ObsType, ActType],
    ABC,
    metaclass=_OpenAIGymEnvMetaclass,
):
    """
    Base class implementing the OpenAI Gym API on the main thread.
    """

    # 初始化重试次数
    _INIT_RETRIES = 100
    # 重试之间的时间间隔
    _TIME_BETWEEN_RETRIES = 0.5
    # 切换挑战任务的重试次数
    _SWITCH_CHALLENGE_TASK_RETRIES = 30
    # 切换重试之间的时间间隔
    _TIME_BETWEEN_SWITCH_RETIRES = 1

    # 初始化方法
    def __init__(
        self,
        account_configuration: Optional[AccountConfiguration] = None,
        *,
        avatar: Optional[int] = None,
        battle_format: str = "gen8randombattle",
        log_level: Optional[int] = None,
        save_replays: Union[bool, str] = False,
        server_configuration: Optional[
            ServerConfiguration
        ] = LocalhostServerConfiguration,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        team: Optional[Union[str, Teambuilder]] = None,
        start_challenging: bool = False,
    # 抽象方法，计算奖励
    @abstractmethod
    def calc_reward(
        self, last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:
        """
        Returns the reward for the current battle state. The battle state in the previous
        turn is given as well and can be used for comparisons.

        :param last_battle: The battle state in the previous turn.
        :type last_battle: AbstractBattle
        :param current_battle: The current battle state.
        :type current_battle: AbstractBattle

        :return: The reward for current_battle.
        :rtype: float
        """
        pass

    # 抽象方法
    @abstractmethod
    # 根据给定的动作和当前战斗状态返回相应的战斗指令
    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        """
        Returns the BattleOrder relative to the given action.

        :param action: The action to take.
        :type action: int
        :param battle: The current battle state
        :type battle: AbstractBattle

        :return: The battle order for the given action in context of the current battle.
        :rtype: BattleOrder
        """
        pass

    # 返回当前战斗状态的嵌入，格式与OpenAI gym API兼容
    @abstractmethod
    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        """
        Returns the embedding of the current battle state in a format compatible with
        the OpenAI gym API.

        :param battle: The current battle state.
        :type battle: AbstractBattle

        :return: The embedding of the current battle state.
        """
        pass

    # 返回嵌入的描述，必须返回一个指定了下限和上限的Space
    @abstractmethod
    def describe_embedding(self) -> Space[ObsType]:
        """
        Returns the description of the embedding. It must return a Space specifying
        low bounds and high bounds.

        :return: The description of the embedding.
        :rtype: Space
        """
        pass

    # 返回动作空间的大小，如果大小为x，则动作空间从0到x-1
    @abstractmethod
    def action_space_size(self) -> int:
        """
        Returns the size of the action space. Given size x, the action space goes
        from 0 to x - 1.

        :return: The action space size.
        :rtype: int
        """
        pass

    # 返回将在挑战循环的下一次迭代中挑战的对手（或对手列表）
    # 如果返回一个列表，则在挑战循环期间将随机选择一个元素
    @abstractmethod
    def get_opponent(
        self,
    ) -> Union[Player, str, List[Player], List[str]]:
        """
        Returns the opponent (or list of opponents) that will be challenged
        on the next iteration of the challenge loop. If a list is returned,
        a random element will be chosen at random during the challenge loop.

        :return: The opponent (or list of opponents).
        :rtype: Player or str or list(Player) or list(str)
        """
        pass
    # 获取对手玩家或字符串
    def _get_opponent(self) -> Union[Player, str]:
        # 获取对手
        opponent = self.get_opponent()
        # 如果对手是列表，则随机选择一个对手，否则直接返回对手
        random_opponent = (
            random.choice(opponent) if isinstance(opponent, list) else opponent
        )
        return random_opponent

    # 重置环境
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        # 如果有种子值，则使用种子值重置环境
        if seed is not None:
            super().reset(seed=seed)  # type: ignore
            self._seed_initialized = True
        # 如果种子值未初始化，则使用当前时间戳作为种子值
        elif not self._seed_initialized:
            super().reset(seed=int(time.time()))  # type: ignore
            self._seed_initialized = True
        # 如果当前没有对战，则等待对战开始
        if not self.agent.current_battle:
            count = self._INIT_RETRIES
            while not self.agent.current_battle:
                if count == 0:
                    raise RuntimeError("Agent is not challenging")
                count -= 1
                time.sleep(self._TIME_BETWEEN_RETRIES)
        # 如果当前对战未结束，则等待对战结束
        if self.current_battle and not self.current_battle.finished:
            if self.current_battle == self.agent.current_battle:
                self._actions.put(-1)
                self._observations.get()
            else:
                raise RuntimeError(
                    "Environment and agent aren't synchronized. Try to restart"
                )
        # 等待当前对战与对手对战不同
        while self.current_battle == self.agent.current_battle:
            time.sleep(0.01)
        # 更新当前对战为对手对战
        self.current_battle = self.agent.current_battle
        battle = copy.copy(self.current_battle)
        battle.logger = None
        self.last_battle = copy.deepcopy(battle)
        return self._observations.get(), self.get_additional_info()

    # 获取额外信息
    def get_additional_info(self) -> Dict[str, Any]:
        """
        Returns additional info for the reset method.
        Override only if you really need it.

        :return: Additional information as a Dict
        :rtype: Dict
        """
        return {}
    def step(
        self, action: ActType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """
        Execute the specified action in the environment.

        :param ActType action: The action to be executed.
        :return: A tuple containing the new observation, reward, termination flag, truncation flag, and info dictionary.
        :rtype: Tuple[ObsType, float, bool, bool, Dict[str, Any]]
        """
        # 如果当前战斗为空，则重置环境并返回初始观察和信息
        if not self.current_battle:
            obs, info = self.reset()
            return obs, 0.0, False, False, info
        # 如果当前战斗已经结束，则抛出异常
        if self.current_battle.finished:
            raise RuntimeError("Battle is already finished, call reset")
        # 复制当前战斗对象，以便进行操作
        battle = copy.copy(self.current_battle)
        battle.logger = None
        # 深度复制当前战斗对象，用于记录上一次的战斗状态
        self.last_battle = copy.deepcopy(battle)
        # 将动作放入动作队列
        self._actions.put(action)
        # 从观察队列中获取观察结果
        observation = self._observations.get()
        # 计算奖励
        reward = self.calc_reward(self.last_battle, self.current_battle)
        terminated = False
        truncated = False
        # 如果当前战斗已经结束
        if self.current_battle.finished:
            size = self.current_battle.team_size
            # 计算剩余队伍中未被击倒的精灵数量
            remaining_mons = size - len(
                [mon for mon in self.current_battle.team.values() if mon.fainted]
            )
            remaining_opponent_mons = size - len(
                [
                    mon
                    for mon in self.current_battle.opponent_team.values()
                    if mon.fainted
                ]
            )
            # 如果一方队伍的精灵全部被击倒，则游戏结束
            if (remaining_mons == 0) != (remaining_opponent_mons == 0):
                terminated = True
            else:
                truncated = True
        # 返回观察结果、奖励、游戏是否结束、游戏是否截断以及额外信息
        return observation, reward, terminated, truncated, self.get_additional_info()
    # 渲染当前战斗状态，显示当前回合信息和双方精灵状态
    def render(self, mode: str = "human"):
        # 如果当前存在战斗
        if self.current_battle is not None:
            # 打印当前回合信息和双方精灵状态
            print(
                "  Turn %4d. | [%s][%3d/%3dhp] %10.10s - %10.10s [%3d%%hp][%s]"
                % (
                    self.current_battle.turn,
                    "".join(
                        [
                            "⦻" if mon.fainted else "●"
                            for mon in self.current_battle.team.values()
                        ]
                    ),
                    self.current_battle.active_pokemon.current_hp or 0,
                    self.current_battle.active_pokemon.max_hp or 0,
                    self.current_battle.active_pokemon.species,
                    self.current_battle.opponent_active_pokemon.species,
                    self.current_battle.opponent_active_pokemon.current_hp or 0,
                    "".join(
                        [
                            "⦻" if mon.fainted else "●"
                            for mon in self.current_battle.opponent_team.values()
                        ]
                    ),
                ),
                end="\n" if self.current_battle.finished else "\r",
            )

    # 关闭当前战斗，清理资源
    def close(self, purge: bool = True):
        # 如果当前没有战斗或者当前战斗已结束
        if self.current_battle is None or self.current_battle.finished:
            # 等待1秒
            time.sleep(1)
            # 如果当前战斗不是代理的当前战斗
            if self.current_battle != self.agent.current_battle:
                self.current_battle = self.agent.current_battle
        # 创建一个异步任务来停止挑战循环
        closing_task = asyncio.run_coroutine_threadsafe(
            self._stop_challenge_loop(purge=purge), POKE_LOOP
        )
        # 获取异步任务的结果
        closing_task.result()
    def background_send_challenge(self, username: str):
        """
        Sends a single challenge to a specified player asynchronously. The function immediately returns
        to allow use of the OpenAI gym API.

        :param username: The username of the player to challenge.
        :type username: str
        """
        # 检查是否已经有挑战任务在进行，如果有则抛出异常
        if self._challenge_task and not self._challenge_task.done():
            raise RuntimeError(
                "Agent is already challenging opponents with the challenging loop. "
                "Try to specify 'start_challenging=True' during instantiation or call "
                "'await agent.stop_challenge_loop()' to clear the task."
            )
        # 在另一个线程中异步运行发送挑战的方法
        self._challenge_task = asyncio.run_coroutine_threadsafe(
            self.agent.send_challenges(username, 1), POKE_LOOP
        )

    def background_accept_challenge(self, username: str):
        """
        Accepts a single challenge from a specified player asynchronously. The function immediately returns
        to allow use of the OpenAI gym API.

        :param username: The username of the player to challenge.
        :type username: str
        """
        # 检查是否已经有挑战任务在进行，如果有则抛出异常
        if self._challenge_task and not self._challenge_task.done():
            raise RuntimeError(
                "Agent is already challenging opponents with the challenging loop. "
                "Try to specify 'start_challenging=True' during instantiation or call "
                "'await agent.stop_challenge_loop()' to clear the task."
            )
        # 在另一个线程中异步运行接受挑战的方法
        self._challenge_task = asyncio.run_coroutine_threadsafe(
            self.agent.accept_challenges(username, 1, self.agent.next_team), POKE_LOOP
        )

    async def _challenge_loop(
        self,
        n_challenges: Optional[int] = None,
        callback: Optional[Callable[[AbstractBattle], None]] = None,
    # 如果没有指定挑战次数，则持续挑战直到 self._keep_challenging 为 False
    ):
        # 如果没有挑战次数且 self._keep_challenging 为 True
        if not n_challenges:
            # 持续挑战直到 self._keep_challenging 为 False
            while self._keep_challenging:
                # 获取对手
                opponent = self._get_opponent()
                # 如果对手是 Player 类型
                if isinstance(opponent, Player):
                    # 进行一场对战
                    await self.agent.battle_against(opponent, 1)
                else:
                    # 发送挑战请求
                    await self.agent.send_challenges(opponent, 1)
                # 如果有回调函数且当前对战不为 None
                if callback and self.current_battle is not None:
                    # 复制当前对战并调用回调函数
                    callback(copy.deepcopy(self.current_battle))
        # 如果指定了挑战次数且挑战次数大于 0
        elif n_challenges > 0:
            # 循环指定次数
            for _ in range(n_challenges):
                # 获取对手
                opponent = self._get_opponent()
                # 如果对手是 Player 类型
                if isinstance(opponent, Player):
                    # 进行一场对战
                    await self.agent.battle_against(opponent, 1)
                else:
                    # 发送挑战请求
                    await self.agent.send_challenges(opponent, 1)
                # 如果有回调函数且当前对战不为 None
                if callback and self.current_battle is not None:
                    # 复制当前对战并调用回调函数
                    callback(copy.deepcopy(self.current_battle))
        # 如果挑战次数小于等于 0
        else:
            # 抛出数值错误异常
            raise ValueError(f"Number of challenges must be > 0. Got {n_challenges}")

    # 开始挑战
    def start_challenging(
        # 指定挑战次数，默认为 None
        self,
        n_challenges: Optional[int] = None,
        # 回调函数，接受 AbstractBattle 类型参数并返回 None
        callback: Optional[Callable[[AbstractBattle], None]] = None,
    ):
        """
        Starts the challenge loop.

        :param n_challenges: The number of challenges to send. If empty it will run until
            stopped.
        :type n_challenges: int, optional
        :param callback: The function to callback after each challenge with a copy of
            the final battle state.
        :type callback: Callable[[AbstractBattle], None], optional
        """
        # 检查是否存在正在进行的挑战任务，如果有则等待直到完成
        if self._challenge_task and not self._challenge_task.done():
            count = self._SWITCH_CHALLENGE_TASK_RETRIES
            while not self._challenge_task.done():
                if count == 0:
                    raise RuntimeError("Agent is already challenging")
                count -= 1
                time.sleep(self._TIME_BETWEEN_SWITCH_RETIRES)
        # 如果没有指定挑战次数，则设置为持续挑战
        if not n_challenges:
            self._keep_challenging = True
        # 启动挑战循环任务
        self._challenge_task = asyncio.run_coroutine_threadsafe(
            self._challenge_loop(n_challenges, callback), POKE_LOOP
        )

    async def _ladder_loop(
        self,
        n_challenges: Optional[int] = None,
        callback: Optional[Callable[[AbstractBattle], None]] = None,
    ):
        # 如果指定了挑战次数，则进行相应次数的挑战
        if n_challenges:
            if n_challenges <= 0:
                raise ValueError(
                    f"Number of challenges must be > 0. Got {n_challenges}"
                )
            for _ in range(n_challenges):
                await self.agent.ladder(1)
                # 如果有回调函数且当前战斗状态不为空，则执行回调函数
                if callback and self.current_battle is not None:
                    callback(copy.deepcopy(self.current_battle))
        # 如果未指定挑战次数，则持续挑战直到停止
        else:
            while self._keep_challenging:
                await self.agent.ladder(1)
                # 如果有回调函数且当前战斗状态不为空，则执行回调函数
                if callback and self.current_battle is not None:
                    callback(copy.deepcopy(self.current_battle))

    # 启动 ladder 循环挑战
    def start_laddering(
        self,
        n_challenges: Optional[int] = None,
        callback: Optional[Callable[[AbstractBattle], None]] = None,
    ):
        """
        Starts the laddering loop.

        :param n_challenges: The number of ladder games to play. If empty it
            will run until stopped.
        :type n_challenges: int, optional
        :param callback: The function to callback after each challenge with a
            copy of the final battle state.
        :type callback: Callable[[AbstractBattle], None], optional
        """
        # 检查是否存在正在进行的挑战任务，如果有则等待直到完成
        if self._challenge_task and not self._challenge_task.done():
            count = self._SWITCH_CHALLENGE_TASK_RETRIES
            while not self._challenge_task.done():
                if count == 0:
                    raise RuntimeError("Agent is already challenging")
                count -= 1
                time.sleep(self._TIME_BETWEEN_SWITCH_RETIRES)
        # 如果没有指定挑战次数，则设置为持续挑战
        if not n_challenges:
            self._keep_challenging = True
        # 使用 asyncio 在另一个线程中运行 _ladder_loop 方法，传入挑战次数和回调函数
        self._challenge_task = asyncio.run_coroutine_threadsafe(
            self._ladder_loop(n_challenges, callback), POKE_LOOP
        )

    async def _stop_challenge_loop(
        self, force: bool = True, wait: bool = True, purge: bool = False
    ):  # 定义一个方法，接受多个参数
        self._keep_challenging = False  # 将属性_keep_challenging设置为False

        if force:  # 如果force为真
            if self.current_battle and not self.current_battle.finished:  # 如果存在当前战斗且未结束
                if not self._actions.empty():  # 如果_actions队列不为空
                    await asyncio.sleep(2)  # 异步等待2秒
                    if not self._actions.empty():  # 如果_actions队列仍不为空
                        raise RuntimeError(  # 抛出运行时错误
                            "The agent is still sending actions. "
                            "Use this method only when training or "
                            "evaluation are over."
                        )
                if not self._observations.empty():  # 如果_observations队列不为空
                    await self._observations.async_get()  # 异步获取_observations队列中的数据
                await self._actions.async_put(-1)  # 异步将-1放入_actions队列中

        if wait and self._challenge_task:  # 如果wait为真且_challenge_task存在
            while not self._challenge_task.done():  # 当_challenge_task未完成时
                await asyncio.sleep(1)  # 异步等待1秒
            self._challenge_task.result()  # 获取_challenge_task的结果

        self._challenge_task = None  # 将_challenge_task设置为None
        self.current_battle = None  # 将current_battle设置为None
        self.agent.current_battle = None  # 将agent的current_battle设置为None
        while not self._actions.empty():  # 当_actions队列不为空时
            await self._actions.async_get()  # 异步获取_actions队列中的数据
        while not self._observations.empty():  # 当_observations队列不为空时
            await self._observations.async_get()  # 异步获取_observations队列中的数据

        if purge:  # 如果purge为真
            self.agent.reset_battles()  # 调用agent的reset_battles方法

    def reset_battles(self):  # 定义一个方法reset_battles
        """Resets the player's inner battle tracker."""  # 重置玩家的内部战斗追踪器
        self.agent.reset_battles()  # 调用agent的reset_battles方法
    # 检查任务是否完成，可设置超时时间
    def done(self, timeout: Optional[int] = None) -> bool:
        """
        Returns True if the task is done or is done after the timeout, false otherwise.

        :param timeout: The amount of time to wait for if the task is not already done.
            If empty it will wait until the task is done.
        :type timeout: int, optional

        :return: True if the task is done or if the task gets completed after the
            timeout.
        :rtype: bool
        """
        # 如果挑战任务为空，则返回True
        if self._challenge_task is None:
            return True
        # 如果超时时间为空，则等待任务完成
        if timeout is None:
            self._challenge_task.result()
            return True
        # 如果挑战任务已完成，则返回True
        if self._challenge_task.done():
            return True
        # 等待一段时间后再次检查任务是否完成
        time.sleep(timeout)
        return self._challenge_task.done()

    # 暴露Player类的属性

    @property
    def battles(self) -> Dict[str, AbstractBattle]:
        return self.agent.battles

    @property
    def format(self) -> str:
        return self.agent.format

    @property
    def format_is_doubles(self) -> bool:
        return self.agent.format_is_doubles

    @property
    def n_finished_battles(self) -> int:
        return self.agent.n_finished_battles

    @property
    def n_lost_battles(self) -> int:
        return self.agent.n_lost_battles

    @property
    def n_tied_battles(self) -> int:
        return self.agent.n_tied_battles

    @property
    def n_won_battles(self) -> int:
        return self.agent.n_won_battles

    @property
    def win_rate(self) -> float:
        return self.agent.win_rate

    # 暴露Player Network Interface Class的属性

    @property
    def logged_in(self) -> asyncio.Event:
        """Event object associated with user login.

        :return: The logged-in event
        :rtype: Event
        """
        return self.agent.ps_client.logged_in

    @property
    # 返回与玩家相关联的日志记录器
    def logger(self) -> Logger:
        """Logger associated with the player.

        :return: The logger.
        :rtype: Logger
        """
        return self.agent.logger

    # 返回玩家的用户名
    @property
    def username(self) -> str:
        """The player's username.

        :return: The player's username.
        :rtype: str
        """
        return self.agent.username

    # 返回 WebSocket 的 URL
    @property
    def websocket_url(self) -> str:
        """The websocket url.

        It is derived from the server url.

        :return: The websocket url.
        :rtype: str
        """
        return self.agent.ps_client.websocket_url

    # 获取属性的值
    def __getattr__(self, item: str):
        return getattr(self.agent, item)
```