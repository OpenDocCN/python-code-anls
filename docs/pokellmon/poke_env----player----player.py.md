# `.\PokeLLMon\poke_env\player\player.py`

```
"""This module defines a base class for players.
"""

import asyncio
import random
from abc import ABC, abstractmethod
from asyncio import Condition, Event, Queue, Semaphore
from logging import Logger
from time import perf_counter
from typing import Any, Awaitable, Dict, List, Optional, Union

import orjson

from poke_env.concurrency import create_in_poke_loop, handle_threaded_coroutines
from poke_env.data import GenData, to_id_str
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.exceptions import ShowdownException
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
)
from poke_env.ps_client import PSClient
from poke_env.ps_client.account_configuration import (
    CONFIGURATION_FROM_PLAYER_COUNTER,
    AccountConfiguration,
)
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from poke_env.teambuilder.teambuilder import Teambuilder

class Player(ABC):
    """
    Base class for players.
    """

    # Set of messages to ignore during battle
    MESSAGES_TO_IGNORE = {"", "t:", "expire", "uhtmlchange"}

    # Chance of using default order after an error due to invalid choice
    DEFAULT_CHOICE_CHANCE = 1 / 1000
    # 初始化函数，设置各种参数的默认值
    def __init__(
        self,
        account_configuration: Optional[AccountConfiguration] = None,  # 账户配置，默认为None
        *,
        avatar: Optional[int] = None,  # 头像，默认为None
        battle_format: str = "gen9randombattle",  # 战斗格式，默认为"gen9randombattle"
        log_level: Optional[int] = None,  # 日志级别，默认为None
        max_concurrent_battles: int = 1,  # 最大并发战斗数，默认为1
        save_replays: Union[bool, str] = False,  # 是否保存战斗重放，默认为False
        server_configuration: Optional[ServerConfiguration] = None,  # 服务器配置，默认为None
        start_timer_on_battle_start: bool = False,  # 战斗开始时是否启动计时器，默认为False
        start_listening: bool = True,  # 是否开始监听，默认为True
        ping_interval: Optional[float] = 20.0,  # ping间隔时间，默认为20.0
        ping_timeout: Optional[float] = 20.0,  # ping超时时间，默认为20.0
        team: Optional[Union[str, Teambuilder]] = None,  # 队伍信息，默认为None
    # 辅助奖励计算函数
    def reward_computing_helper(
        self,
        battle: AbstractBattle,  # 战斗对象
        *,
        fainted_value: float = 0.0,  # 损失生命值的价值，默认为0.0
        hp_value: float = 0.0,  # 生命值的价值，默认为0.0
        number_of_pokemons: int = 6,  # 参与战斗的宝可梦数量，默认为6
        starting_value: float = 0.0,  # 初始价值，默认为0.0
        status_value: float = 0.0,  # 状态价值，默认为0.0
        victory_value: float = 1.0,  # 胜利价值，默认为1.0
    # 定义一个方法，计算战斗的奖励值
    ) -> float:

        # 如果战斗不在奖励缓冲区中，则将其初始值添加到缓冲区中
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        # 遍历我方队伍中的每只精灵
        for mon in battle.team.values():
            # 根据当前生命值比例计算当前值
            current_value += mon.current_hp_fraction * hp_value
            # 如果精灵已经倒下，则减去倒下的值
            if mon.fainted:
                current_value -= fainted_value
            # 如果精灵有异常状态，则减去异常状态的值
            elif mon.status is not None:
                current_value -= status_value

        # 根据我方队伍中精灵数量与总精灵数量的差值计算当前值
        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        # 遍历对手队伍中的每只精灵
        for mon in battle.opponent_team.values():
            # 根据当前生命值比例计算当前值
            current_value -= mon.current_hp_fraction * hp_value
            # 如果精灵已经倒下，则加上倒下的值
            if mon.fainted:
                current_value += fainted_value
            # 如果精灵有异常状态，则加上异常状态的值
            elif mon.status is not None:
                current_value += status_value

        # 根据对手队伍中精灵数量与总精灵数量的差值计算当前值
        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        # 如果战斗胜利，则加上胜利的值；如果战斗失败，则减去胜利的值
        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        # 计算最终奖励值并更新奖励缓冲区
        to_return = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value
        return to_return


    # 定义一个方法，创建账户配置
    def _create_account_configuration(self) -> AccountConfiguration:
        key = type(self).__name__
        # 更新配置计数器
        CONFIGURATION_FROM_PLAYER_COUNTER.update([key])
        username = "%s %d" % (key, CONFIGURATION_FROM_PLAYER_COUNTER[key])
        # 如果用户名长度超过18个字符，则截取前18个字符
        if len(username) > 18:
            username = "%s %d" % (
                key[: 18 - len(username)],
                CONFIGURATION_FROM_PLAYER_COUNTER[key],
            )
        return AccountConfiguration(username, None)

    # 定义一个方法，处理战斗结束的回调
    def _battle_finished_callback(self, battle: AbstractBattle):
        pass
    def update_team(self, team: Union[Teambuilder, str]):
        """Updates the team used by the player.

        :param team: The new team to use.
        :type team: str or Teambuilder
        """
        # 如果传入的 team 是 Teambuilder 类型，则直接赋值给 self._team
        if isinstance(team, Teambuilder):
            self._team = team
        else:
            # 如果传入的 team 不是 Teambuilder 类型，则使用 ConstantTeambuilder 类创建一个新的 team
            self._team = ConstantTeambuilder(team)

    async def _get_battle(self, battle_tag: str) -> AbstractBattle:
        # 去掉 battle_tag 的第一个字符
        battle_tag = battle_tag[1:]
        while True:
            # 如果 battle_tag 在 self._battles 中，则返回对应的 AbstractBattle 对象
            if battle_tag in self._battles:
                return self._battles[battle_tag]
            # 否则等待 _battle_start_condition 条件满足
            async with self._battle_start_condition:
                await self._battle_start_condition.wait()

    async def _handle_battle_request(
        self,
        battle: AbstractBattle,
        from_teampreview_request: bool = False,
        maybe_default_order: bool = False,
    ):
        # 如果 maybe_default_order 为 True 且随机数小于 DEFAULT_CHOICE_CHANCE，则选择默认的移动
        if maybe_default_order and random.random() < self.DEFAULT_CHOICE_CHANCE:
            message = self.choose_default_move().message
        elif battle.teampreview:
            # 如果在队伍预览阶段，且不是来自队伍预览请求，则返回
            if not from_teampreview_request:
                return
            message = self.teampreview(battle)
        else:
            # 选择移动或者等待选择移动的结果
            message = self.choose_move(battle)
            if isinstance(message, Awaitable):
                message = await message
            message = message.message

        # 发送消息到 ps_client，指定 battle.battle_tag
        await self.ps_client.send_message(message, battle.battle_tag)

    async def _handle_challenge_request(self, split_message: List[str]):
        """Handles an individual challenge."""
        # 获取挑战者的用户名
        challenging_player = split_message[2].strip()

        # 如果挑战者不是当前用户，并且消息长度大于等于 6，且消息中指定的格式与 self._format 相同，则将挑战者加入到 _challenge_queue 中
        if challenging_player != self.username:
            if len(split_message) >= 6:
                if split_message[5] == self._format:
                    await self._challenge_queue.put(challenging_player)
    async def _update_challenges(self, split_message: List[str]):
        """Update internal challenge state.

        Add corresponding challenges to internal queue of challenges, where they will be
        processed if relevant.

        :param split_message: Recevied message, split.
        :type split_message: List[str]
        """
        # 记录调试信息，更新挑战信息
        self.logger.debug("Updating challenges with %s", split_message)
        # 从消息中解析挑战信息
        challenges = orjson.loads(split_message[2]).get("challengesFrom", {})
        # 遍历挑战信息，将符合条件的挑战加入挑战队列
        for user, format_ in challenges.items():
            if format_ == self._format:
                await self._challenge_queue.put(user)

    async def accept_challenges(
        self,
        opponent: Optional[Union[str, List[str]]],
        n_challenges: int,
        packed_team: Optional[str] = None,
    ):
        """Let the player wait for challenges from opponent, and accept them.

        If opponent is None, every challenge will be accepted. If opponent if a string,
        all challenges from player with that name will be accepted. If opponent is a
        list all challenges originating from players whose name is in the list will be
        accepted.

        Up to n_challenges challenges will be accepted, after what the function will
        wait for these battles to finish, and then return.

        :param opponent: Players from which challenges will be accepted.
        :type opponent: None, str or list of str
        :param n_challenges: Number of challenges that will be accepted
        :type n_challenges: int
        :packed_team: Team to use. Defaults to generating a team with the agent's teambuilder.
        :type packed_team: string, optional.
        """
        # 如果没有指定队伍，则使用下一个队伍
        if packed_team is None:
            packed_team = self.next_team

        # 导入日志模块，记录警告信息
        import logging
        logging.warning("AAAHHH in accept_challenges")
        # 处理多线程协程，接受挑战
        await handle_threaded_coroutines(
            self._accept_challenges(opponent, n_challenges, packed_team)
        )
    # 异步方法，用于接受挑战
    async def _accept_challenges(
        self,
        opponent: Optional[Union[str, List[str]]],  # 对手的用户名，可以是字符串或字符串列表
        n_challenges: int,  # 挑战的次数
        packed_team: Optional[str],  # 打包的队伍信息
    ):
        import logging  # 导入日志模块
        logging.warning("AAAHHH in _accept_challenges")  # 记录警告日志
        if opponent:  # 如果存在对手
            if isinstance(opponent, list):  # 如果对手是列表
                opponent = [to_id_str(o) for o in opponent]  # 将对手列表中的每个元素转换为字符串
            else:
                opponent = to_id_str(opponent)  # 将对手转换为字符串
        await self.ps_client.logged_in.wait()  # 等待客户端登录完成
        self.logger.debug("Event logged in received in accept_challenge")  # 记录调试日志

        for _ in range(n_challenges):  # 循环处理挑战次数
            while True:  # 无限循环
                username = to_id_str(await self._challenge_queue.get())  # 从挑战队列中获取用户名并转换为字符串
                self.logger.debug(
                    "Consumed %s from challenge queue in accept_challenge", username
                )  # 记录调试日志
                if (
                    (opponent is None)  # 如果对手为空
                    or (opponent == username)  # 或对手等于用户名
                    or (isinstance(opponent, list) and (username in opponent))  # 或对手是列表且用户名在对手列表中
                ):
                    await self.ps_client.accept_challenge(username, packed_team)  # 接受挑战
                    await self._battle_semaphore.acquire()  # 获取战斗信号量
                    break  # 跳出循环
        await self._battle_count_queue.join()  # 等待所有战斗完成

    @abstractmethod
    def choose_move(
        self, battle: AbstractBattle
    ) -> Union[BattleOrder, Awaitable[BattleOrder]]:
        """Abstract method to choose a move in a battle.

        :param battle: The battle.
        :type battle: AbstractBattle
        :return: The move order.
        :rtype: str
        """
        pass

    # 选择默认的战斗指令
    def choose_default_move(self) -> DefaultBattleOrder:
        """Returns showdown's default move order.

        This order will result in the first legal order - according to showdown's
        ordering - being chosen.
        """
        return DefaultBattleOrder()
    # 选择一个随机的单个行动，返回一个 BattleOrder 对象
    def choose_random_singles_move(self, battle: Battle) -> BattleOrder:
        # 创建一个包含可用行动的 BattleOrder 对象列表
        available_orders = [BattleOrder(move) for move in battle.available_moves]
        # 将可用交换的 BattleOrder 对象也添加到列表中
        available_orders.extend(
            [BattleOrder(switch) for switch in battle.available_switches]
        )

        # 如果可以超级进化，将带有 mega 标记的 BattleOrder 对象也添加到列表中
        if battle.can_mega_evolve:
            available_orders.extend(
                [BattleOrder(move, mega=True) for move in battle.available_moves]
            )

        # 如果可以极巨化，将带有 dynamax 标记的 BattleOrder 对象也添加到列表中
        if battle.can_dynamax:
            available_orders.extend(
                [BattleOrder(move, dynamax=True) for move in battle.available_moves]
            )

        # 如果可以使用 tera，将带有 terastallize 标记的 BattleOrder 对象也添加到列表中
        if battle.can_tera:
            available_orders.extend(
                [
                    BattleOrder(move, terastallize=True)
                    for move in battle.available_moves
                ]
            )

        # 如果可以使用 Z 招式且有活跃的精灵，将带有 z_move 标记的 BattleOrder 对象也添加到列表中
        if battle.can_z_move and battle.active_pokemon:
            available_z_moves = set(battle.active_pokemon.available_z_moves)
            available_orders.extend(
                [
                    BattleOrder(move, z_move=True)
                    for move in battle.available_moves
                    if move in available_z_moves
                ]
            )

        # 如果有可用的行动，则随机选择一个返回
        if available_orders:
            return available_orders[int(random.random() * len(available_orders))]
        # 如果没有可用的行动，则选择默认行动
        else:
            return self.choose_default_move()
    # 从给定的战斗中选择一个随机的合法移动
    def choose_random_move(self, battle: AbstractBattle) -> BattleOrder:
        """Returns a random legal move from battle.

        :param battle: The battle in which to move.
        :type battle: AbstractBattle
        :return: Move order
        :rtype: str
        """
        # 如果战斗是单打模式
        if isinstance(battle, Battle):
            # 调用选择单打模式下的随机移动方法
            return self.choose_random_singles_move(battle)
        # 如果战斗是双打模式
        elif isinstance(battle, DoubleBattle):
            # 调用选择双打模式下的随机移动方法
            return self.choose_random_doubles_move(battle)
        else:
            # 抛出数值错误，提示战斗应该是 Battle 或 DoubleBattle 类型
            raise ValueError(
                "battle should be Battle or DoubleBattle. Received %d" % (type(battle))
            )

    # 在天梯上进行游戏
    async def ladder(self, n_games: int):
        """Make the player play games on the ladder.

        n_games defines how many battles will be played.

        :param n_games: Number of battles that will be played
        :type n_games: int
        """
        # 等待玩家登录完成
        await handle_threaded_coroutines(self._ladder(n_games))

    # 在天梯上进行游戏的内部方法
    async def _ladder(self, n_games: int):
        # 等待玩家登录完成
        await self.ps_client.logged_in.wait()
        # 记录开始时间
        start_time = perf_counter()

        # 循环进行指定次数的游戏
        for _ in range(n_games):
            # 异步等待战斗开始条件
            async with self._battle_start_condition:
                # 搜索天梯游戏并等待战斗开始
                await self.ps_client.search_ladder_game(self._format, self.next_team)
                await self._battle_start_condition.wait()
                # 当战斗计数队列已满时
                while self._battle_count_queue.full():
                    # 异步等待战斗结束条件
                    async with self._battle_end_condition:
                        await self._battle_end_condition.wait()
                # 获取战斗信号量
                await self._battle_semaphore.acquire()
        # 等待战斗计数队列完成
        await self._battle_count_queue.join()
        # 记录日志，显示天梯游戏完成所花费的时间
        self.logger.info(
            "Laddering (%d battles) finished in %fs",
            n_games,
            perf_counter() - start_time,
        )
    # 异步方法，使玩家与对手进行指定次数的对战
    async def battle_against(self, opponent: "Player", n_battles: int = 1):
        """Make the player play n_battles against opponent.

        This function is a wrapper around send_challenges and accept challenges.

        :param opponent: The opponent to play against.
        :type opponent: Player
        :param n_battles: The number of games to play. Defaults to 1.
        :type n_battles: int
        """
        # 调用异步方法处理线程化的协程
        await handle_threaded_coroutines(self._battle_against(opponent, n_battles))

    # 异步方法，实际进行玩家与对手的对战
    async def _battle_against(self, opponent: "Player", n_battles: int):
        # 并发执行发送挑战和接受挑战的操作
        await asyncio.gather(
            self.send_challenges(
                to_id_str(opponent.username),
                n_battles,
                to_wait=opponent.ps_client.logged_in,
            ),
            opponent.accept_challenges(
                to_id_str(self.username), n_battles, opponent.next_team
            ),
        )

    # 异步方法，使玩家发送挑战给对手
    async def send_challenges(
        self, opponent: str, n_challenges: int, to_wait: Optional[Event] = None
    ):
        """Make the player send challenges to opponent.

        opponent must be a string, corresponding to the name of the player to challenge.

        n_challenges defines how many challenges will be sent.

        to_wait is an optional event that can be set, in which case it will be waited
        before launching challenges.

        :param opponent: Player username to challenge.
        :type opponent: str
        :param n_challenges: Number of battles that will be started
        :type n_challenges: int
        :param to_wait: Optional event to wait before launching challenges.
        :type to_wait: Event, optional.
        """
        # 调用异步方法处理线程化的协程
        await handle_threaded_coroutines(
            self._send_challenges(opponent, n_challenges, to_wait)
        )

    # 异步方法，实际发送挑战给对手
    async def _send_challenges(
        self, opponent: str, n_challenges: int, to_wait: Optional[Event] = None
    ): 
        # 等待玩家登录完成
        await self.ps_client.logged_in.wait()
        # 在发送挑战时记录事件已登录
        self.logger.info("Event logged in received in send challenge")

        # 如果有需要等待的事件，则等待
        if to_wait is not None:
            await to_wait.wait()

        # 记录开始时间
        start_time = perf_counter()

        # 发起指定数量的挑战
        for _ in range(n_challenges):
            await self.ps_client.challenge(opponent, self._format, self.next_team)
            await self._battle_semaphore.acquire()
        # 等待所有挑战结束
        await self._battle_count_queue.join()
        # 记录挑战完成所用时间
        self.logger.info(
            "Challenges (%d battles) finished in %fs",
            n_challenges,
            perf_counter() - start_time,
        )

    # 生成随机的队伍预览顺序
    def random_teampreview(self, battle: AbstractBattle) -> str:
        """Returns a random valid teampreview order for the given battle.

        :param battle: The battle.
        :type battle: AbstractBattle
        :return: The random teampreview order.
        :rtype: str
        """
        # 生成队伍成员列表
        members = list(range(1, len(battle.team) + 1))
        # 随机打乱成员顺序
        random.shuffle(members)
        return "/team " + "".join([str(c) for c in members])

    # 重置玩家的内部战斗追踪器
    def reset_battles(self):
        """Resets the player's inner battle tracker."""
        # 遍历所有战斗，如果有未结束的战斗则抛出异常
        for battle in list(self._battles.values()):
            if not battle.finished:
                raise EnvironmentError(
                    "Can not reset player's battles while they are still running"
                )
        # 重置战斗追踪器
        self._battles = {}
    # 定义一个方法，用于生成给定战斗的队伍预览顺序
    def teampreview(self, battle: AbstractBattle) -> str:
        """Returns a teampreview order for the given battle.

        This order must be of the form /team TEAM, where TEAM is a string defining the
        team chosen by the player. Multiple formats are supported, among which '3461'
        and '3, 4, 6, 1' are correct and indicate leading with pokemon 3, with pokemons
        4, 6 and 1 in the back in single battles or leading with pokemons 3 and 4 with
        pokemons 6 and 1 in the back in double battles.

        Please refer to Pokemon Showdown's protocol documentation for more information.

        :param battle: The battle.
        :type battle: AbstractBattle
        :return: The teampreview order.
        :rtype: str
        """
        # 返回一个随机的队伍预览顺序
        return self.random_teampreview(battle)

    @staticmethod
    def create_order(
        order: Union[Move, Pokemon],
        mega: bool = False,
        z_move: bool = False,
        dynamax: bool = False,
        terastallize: bool = False,
        move_target: int = DoubleBattle.EMPTY_TARGET_POSITION,
    ) -> BattleOrder:
        """Formats an move order corresponding to the provided pokemon or move.

        :param order: Move to make or Pokemon to switch to.
        :type order: Move or Pokemon
        :param mega: Whether to mega evolve the pokemon, if a move is chosen.
        :type mega: bool
        :param z_move: Whether to make a zmove, if a move is chosen.
        :type z_move: bool
        :param dynamax: Whether to dynamax, if a move is chosen.
        :type dynamax: bool
        :param terastallize: Whether to terastallize, if a move is chosen.
        :type terastallize: bool
        :param move_target: Target Pokemon slot of a given move
        :type move_target: int
        :return: Formatted move order
        :rtype: str
        """
        # 格式化一个移动顺序，对应于提供的精灵或招式
        return BattleOrder(
            order,
            mega=mega,
            move_target=move_target,
            z_move=z_move,
            dynamax=dynamax,
            terastallize=terastallize,
        )

    @property
    def battles(self) -> Dict[str, AbstractBattle]:
        # 返回私有属性 _battles
        return self._battles

    @property
    def format(self) -> str:
        # 返回私有属性 _format
        return self._format

    @property
    def format_is_doubles(self) -> bool:
        # 检查比赛格式是否为双打
        format_lowercase = self._format.lower()
        return (
            "vgc" in format_lowercase
            or "double" in format_lowercase
            or "metronome" in format_lowercase
        )

    @property
    def n_finished_battles(self) -> int:
        # 返回已完成的比赛数量
        return len([None for b in self._battles.values() if b.finished])

    @property
    def n_lost_battles(self) -> int:
        # 返回已输掉的比赛数量
        return len([None for b in self._battles.values() if b.lost])

    @property
    def n_tied_battles(self) -> int:
        # 返回打平的比赛数量
        return self.n_finished_battles - self.n_lost_battles - self.n_won_battles

    @property
    def n_won_battles(self) -> int:
        # 返回已赢得的比赛数量
        return len([None for b in self._battles.values() if b.won])

    @property
    # 计算胜率，即已赢得的战斗数除以已完成的战斗数
    def win_rate(self) -> float:
        return self.n_won_battles / self.n_finished_battles

    # 返回日志记录器对象
    @property
    def logger(self) -> Logger:
        return self.ps_client.logger

    # 返回用户名
    @property
    def username(self) -> str:
        return self.ps_client.username

    # 返回下一个队伍的名称，如果存在的话
    @property
    def next_team(self) -> Optional[str]:
        # 如果存在队伍对象，则返回其生成的队伍名称
        if self._team:
            return self._team.yield_team()
        # 否则返回空值
        return None
```