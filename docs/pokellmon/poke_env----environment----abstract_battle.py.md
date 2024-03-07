# `.\PokeLLMon\poke_env\environment\abstract_battle.py`

```
# 导入必要的模块
import os
from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# 导入自定义模块
from poke_env.data import GenData, to_id_str
from poke_env.data.replay_template import REPLAY_TEMPLATE
from poke_env.environment.field import Field
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import STACKABLE_CONDITIONS, SideCondition
from poke_env.environment.weather import Weather

# 定义一个抽象类 AbstractBattle
class AbstractBattle(ABC):
    # 定义一个常量集合，包含需要忽略的消息
    MESSAGES_TO_IGNORE = {
        "-anim",
        "-burst",
        "-block",
        "-center",
        "-crit",
        "-combine",
        "-fail",
        "-fieldactivate",
        "-hint",
        "-hitcount",
        "-ohko",
        "-miss",
        "-notarget",
        "-nothing",
        "-resisted",
        "-singlemove",
        "-singleturn",
        "-supereffective",
        "-waiting",
        "-zbroken",
        "askreg",
        "debug",
        "chat",
        "c",
        "crit",
        "deinit",
        "gametype",
        "gen",
        "html",
        "init",
        "immune",
        "join",
        "j",
        "J",
        "leave",
        "l",
        "L",
        "name",
        "n",
        "rated",
        "resisted",
        "split",
        "supereffective",
        "teampreview",
        "tier",
        "upkeep",
        "zbroken",
    }
    # 定义类的属性，使用 __slots__ 来限制实例的属性，提高内存利用效率
    __slots__ = (
        "_anybody_inactive",
        "_available_moves",
        "_available_switches",
        "_battle_tag",
        "_can_dynamax",
        "_can_mega_evolve",
        "_can_tera",
        "_can_z_move",
        "_data",
        "_dynamax_turn",
        "_fields",
        "_finished",
        "_force_switch",
        "_format",
        "in_team_preview",
        "_max_team_size",
        "_maybe_trapped",
        "_move_on_next_request",
        "_opponent_can_dynamax",
        "_opponent_can_mega_evolve",
        "_opponent_can_terrastallize",
        "_opponent_can_z_move",
        "_opponent_dynamax_turn",
        "_opponent_rating",
        "_opponent_side_conditions",
        "_opponent_team",
        "_opponent_username",
        "_player_role",
        "_player_username",
        "_players",
        "_rating",
        "_reconnected",
        "_replay_data",
        "_rqid",
        "rules",
        "_reviving",
        "_save_replays",
        "_side_conditions",
        "_team_size",
        "_team",
        "_teampreview_opponent_team",
        "_teampreview",
        "_trapped",
        "_turn",
        "_wait",
        "_weather",
        "_won",
        "logger",
    )

    # 初始化方法，用于创建类的实例
    def __init__(
        self,
        battle_tag: str,  # 战斗标签
        username: str,  # 用户名
        logger: Logger,  # 日志记录器
        save_replays: Union[str, bool],  # 保存重播记录的路径或布尔值
        gen: int,  # 世代
        # 加载数据
        self._data = GenData.from_gen(gen)

        # 工具属性
        self._battle_tag: str = battle_tag
        self._format: Optional[str] = None
        self._max_team_size: Optional[int] = None
        self._opponent_username: Optional[str] = None
        self._player_role: Optional[str] = None
        self._player_username: str = username
        self._players: List[Dict[str, str]] = []
        self._replay_data: List[List[str]] = []
        self._save_replays: Union[str, bool] = save_replays
        self._team_size: Dict[str, int] = {}
        self._teampreview: bool = False
        self._teampreview_opponent_team: Set[Pokemon] = set()
        self._anybody_inactive: bool = False
        self._reconnected: bool = True
        self.logger: Optional[Logger] = logger

        # 回合选择属性
        self.in_team_preview: bool = False
        self._move_on_next_request: bool = False
        self._wait: Optional[bool] = None

        # 战斗状态属性
        self._dynamax_turn: Optional[int] = None
        self._finished: bool = False
        self._rqid = 0
        self.rules: List[str] = []
        self._turn: int = 0
        self._opponent_can_terrastallize: bool = True
        self._opponent_dynamax_turn: Optional[int] = None
        self._opponent_rating: Optional[int] = None
        self._rating: Optional[int] = None
        self._won: Optional[bool] = None

        # 游戏中的战斗状态属性
        self._weather: Dict[Weather, int] = {}
        self._fields: Dict[Field, int] = {}  # set()
        self._opponent_side_conditions: Dict[SideCondition, int] = {}  # set()
        self._side_conditions: Dict[SideCondition, int] = {}  # set()
        self._reviving: bool = False

        # Pokemon 属性
        self._team: Dict[str, Pokemon] = {}
        self._opponent_team: Dict[str, Pokemon] = {}
    # 定义一个方法用于获取精灵信息
    def get_pokemon(
        self,
        identifier: str,
        force_self_team: bool = False,
        details: str = "",
        request: Optional[Dict[str, Any]] = None,
    
    # 定义一个抽象方法用于清除所有精灵的增益效果
    @abstractmethod
    def clear_all_boosts(self):
        pass

    # 检查伤害信息中是否包含关于道具的信息
    def _check_damage_message_for_item(self, split_message: List[str]):
        # 捕获对方精灵受到道具伤害的情况
        # 道具属于未受伤害的一方
        if (
            len(split_message) == 6
            and split_message[4].startswith("[from] item:")
            and split_message[5].startswith("[of]")
        ):
            item = split_message[4].split("item:")[-1]
            pkmn = split_message[5].split("[of]")[-1].strip()
            self.get_pokemon(pkmn).item = to_id_str(item)

        # 捕获自身精灵受到道具伤害的情况
        # 道具属于受伤害的一方
        elif len(split_message) == 5 and split_message[4].startswith("[from] item:"):
            item = split_message[4].split("item:")[-1]
            pkmn = split_message[2]
            self.get_pokemon(pkmn).item = to_id_str(item)
    def _check_damage_message_for_ability(self, split_message: List[str]):
        # 检查是否有对手的能力造成伤害的消息
        # 物品来自未受伤害的一方
        # 例如:
        #   |-damage|p2a: Archeops|88/100|[from] ability: Iron Barbs|[of] p1a: Ferrothorn
        if (
            len(split_message) == 6
            and split_message[4].startswith("[from] ability:")
            and split_message[5].startswith("[of]")
        ):
            # 从消息中提取能力信息
            ability = split_message[4].split("ability:")[-1]
            # 从消息中提取宝可梦信息
            pkmn = split_message[5].split("[of]")[-1].strip()
            # 设置宝可梦的能力
            self.get_pokemon(pkmn).ability = to_id_str(ability)

    def _check_heal_message_for_item(self, split_message: List[str]):
        # 检查是否有宝可梦从自己的物品中恢复
        # 检查物品不为 None 是必要的，因为 PS 模拟器会在消耗掉一颗树果后才显示恢复消息
        # 例子:
        #  |-heal|p2a: Quagsire|100/100|[from] item: Leftovers
        #  |-heal|p2a: Quagsire|100/100|[from] item: Sitrus Berry
        if len(split_message) == 5 and split_message[4].startswith("[from] item:"):
            # 从消息中提取宝可梦信息
            pkmn = split_message[2]
            # 从消息中提取物品信息
            item = split_message[4].split("item:")[-1]
            # 获取宝可梦对象
            pkmn_object = self.get_pokemon(pkmn)
            # 如果宝可梦已经有物品，则设置物品
            if pkmn_object.item is not None:
                pkmn_object.item = to_id_str(item)
    # 检查治疗消息中是否包含能力相关信息
    def _check_heal_message_for_ability(self, split_message: List[str]):
        # 捕获当一方通过自身能力进行治疗的情况
        # PS 服务器发送的 "of" 组件有点误导性
        #   它暗示能力来自对立方
        # 示例:
        #   |-heal|p2a: Quagsire|100/100|[from] ability: Water Absorb|[of] p1a: Genesect
        if len(split_message) == 6 and split_message[4].startswith("[from] ability:"):
            # 提取能力信息
            ability = split_message[4].split("ability:")[-1]
            # 提取宝可梦名称
            pkmn = split_message[2]
            # 设置宝可梦的能力
            self.get_pokemon(pkmn).ability = to_id_str(ability)

    @abstractmethod
    # 结束幻象状态的抽象方法
    def end_illusion(self, pokemon_name: str, details: str):
        pass

    # 结束幻象状态的具体实现
    def _end_illusion_on(
        self, illusionist: Optional[str], illusioned: Optional[Pokemon], details: str
    ):
        # 如果没有幻象者，则抛出异常
        if illusionist is None:
            raise ValueError("Cannot end illusion without an active pokemon.")
        # 如果没有被幻象的宝可梦，则抛出异常
        if illusioned is None:
            raise ValueError("Cannot end illusion without an illusioned pokemon.")
        # 获取幻象者的宝可梦对象
        illusionist_mon = self.get_pokemon(illusionist, details=details)

        # 如果幻象者和被幻象的宝可梦是同一个，则直接返回幻象者
        if illusionist_mon is illusioned:
            return illusionist_mon

        # 将幻象者切换到战斗状态
        illusionist_mon.switch_in(details=details)
        # 设置幻象者的状态
        illusionist_mon.status = (
            illusioned.status.name if illusioned.status is not None else None
        )
        # 设置幻象者的生命值
        illusionist_mon.set_hp(f"{illusioned.current_hp}/{illusioned.max_hp}")

        # 标记被幻象的宝可梦已经解除幻象状态
        illusioned.was_illusioned()

        return illusionist_mon

    # 处理场地结束状态的方法
    def _field_end(self, field_str: str):
        # 从 Showdown 消息中创建场地对象
        field = Field.from_showdown_message(field_str)
        # 如果场地不是未知状态，则移除该场地
        if field is not Field.UNKNOWN:
            self._fields.pop(field)
    # 定义一个方法，用于处理战场开始的字段信息
    def field_start(self, field_str: str):
        # 将传入的字段信息转换为Field对象
        field = Field.from_showdown_message(field_str)

        # 如果字段是地形字段
        if field.is_terrain:
            # 更新战场上的字段信息，移除之前的地形字段
            self._fields = {
                field: turn
                for field, turn in self._fields.items()
                if not field.is_terrain
            }

        # 将当前字段信息添加到战场上
        self._fields[field] = self.turn

    # 完成战斗
    def _finish_battle(self):
        # 如果需要保存战斗回放
        if self._save_replays:
            # 根据保存回放的设置确定保存的文件夹
            if self._save_replays is True:
                folder = "replays"
            else:
                folder = str(self._save_replays)

            # 如果文件夹不存在，则创建文件夹
            if not os.path.exists(folder):
                os.mkdir(folder)

            # 打开文件，写入格式化后的回放数据
            with open(
                os.path.join(
                    folder, f"{self._player_username} - {self.battle_tag}.html"
                ),
                "w+",
                encoding="utf-8",
            ) as f:
                formatted_replay = REPLAY_TEMPLATE

                # 替换模板中的占位符为实际数据
                formatted_replay = formatted_replay.replace(
                    "{BATTLE_TAG}", f"{self.battle_tag}"
                )
                formatted_replay = formatted_replay.replace(
                    "{PLAYER_USERNAME}", f"{self._player_username}"
                )
                formatted_replay = formatted_replay.replace(
                    "{OPPONENT_USERNAME}", f"{self._opponent_username}"
                )
                replay_log = f">{self.battle_tag}" + "\n".join(
                    ["|".join(split_message) for split_message in self._replay_data]
                )
                formatted_replay = formatted_replay.replace("{REPLAY_LOG}", replay_log)

                f.write(formatted_replay)

        # 标记战斗结束
        self._finished = True

    # 抽象方法，用于解析请求
    @abstractmethod
    def parse_request(self, request: Dict[str, Any]):
        pass

    # 注册对手的队伍信息
    def _register_teampreview_pokemon(self, player: str, details: str):
        # 如果玩家不是当前玩家角色
        if player != self._player_role:
            # 创建Pokemon对象，并添加到对手的队伍信息中
            mon = Pokemon(details=details, gen=self._data.gen)
            self._teampreview_opponent_team.add(mon)
    # 根据给定的边（side）和条件字符串（condition_str）来结束边的状态
    def side_end(self, side: str, condition_str: str):
        # 如果边的前两个字符与玩家角色相同，则使用边的条件
        if side[:2] == self._player_role:
            conditions = self.side_conditions
        else:
            conditions = self.opponent_side_conditions
        # 从 Showdown 消息中创建边的条件对象
        condition = SideCondition.from_showdown_message(condition_str)
        # 如果条件不是未知状态，则从条件中移除
        if condition is not SideCondition.UNKNOWN:
            conditions.pop(condition)

    # 根据给定的边（side）和条件字符串（condition_str）来开始边的状态
    def _side_start(self, side: str, condition_str: str):
        # 如果边的前两个字符与玩家角色相同，则使用边的条件
        if side[:2] == self._player_role:
            conditions = self.side_conditions
        else:
            conditions = self.opponent_side_conditions
        # 从 Showdown 消息中创建边的条件对象
        condition = SideCondition.from_showdown_message(condition_str)
        # 如果条件可以叠加，则将条件添加到边的条件中
        if condition in STACKABLE_CONDITIONS:
            conditions[condition] = conditions.get(condition, 0) + 1
        # 如果条件不在边的条件中，则将条件添加到边的条件中，并记录回合数
        elif condition not in conditions:
            conditions[condition] = self.turn

    # 交换精灵，暂未实现
    def _swap(self, pokemon_str: str, slot: str):
        if self.logger is not None:
            self.logger.warning("swap method in Battle is not implemented")

    # 切换精灵的抽象方法
    @abstractmethod
    def switch(self, pokemon_str: str, details: str, hp_status: str):
        pass

    # 平局结束战斗
    def tied(self):
        self._finish_battle()

    # 从请求中更新队伍信息
    def _update_team_from_request(self, side: Dict[str, Any]):
        for pokemon in side["pokemon"]:
            # 如果精灵在队伍中，则更新精灵信息，否则创建新的精灵
            if pokemon["ident"] in self._team:
                self._team[pokemon["ident"]].update_from_request(pokemon)
            else:
                self.get_pokemon(
                    pokemon["ident"], force_self_team=True, request=pokemon
                )

    # 根据获胜玩家名字结束战斗
    def won_by(self, player_name: str):
        # 如果获胜玩家名字与玩家用户名相同，则设置胜利标志为 True，否则为 False
        if player_name == self._player_username:
            self._won = True
        else:
            self._won = False
        # 结束战斗
        self._finish_battle()

    # 结束回合
    def end_turn(self, turn: int):
        # 更新当前回合数
        self.turn = turn

        # 对所有活跃的精灵执行结束回合操作
        for mon in self.all_active_pokemons:
            if mon:
                mon.end_turn()

    # 获取当前活跃的精灵的抽象属性
    @property
    @abstractmethod
    def active_pokemon(self) -> Any:
        pass
    @property
    @abstractmethod
    def all_active_pokemons(self) -> List[Optional[Pokemon]]:
        pass



    @property
    @abstractmethod
    def available_moves(self) -> Any:
        pass



    @property
    @abstractmethod
    def available_switches(self) -> Any:
        pass



    @property
    def battle_tag(self) -> str:
        """
        :return: The battle identifier.
        :rtype: str
        """
        return self._battle_tag



    @property
    @abstractmethod
    def can_dynamax(self) -> Any:
        pass



    @property
    @abstractmethod
    def can_mega_evolve(self) -> Any:
        pass



    @property
    @abstractmethod
    def can_z_move(self) -> Any:
        pass



    @property
    @abstractmethod
    def can_tera(self) -> Any:
        pass



    @property
    def dynamax_turns_left(self) -> Optional[int]:
        """
        :return: How many turns of dynamax are left. None if dynamax is not active
        :rtype: int, optional
        """
        if self._dynamax_turn is not None and any(
            map(lambda pokemon: pokemon.is_dynamaxed, self._team.values())
        ):
            return max(3 - (self.turn - self._dynamax_turn), 0)



    @property
    def fields(self) -> Dict[Field, int]:
        """
        :return: A Dict mapping fields to the turn they have been activated.
        :rtype: Dict[Field, int]
        """
        return self._fields



    @property
    def finished(self) -> bool:
        """
        :return: A boolean indicating whether the battle is finished.
        :rtype: Optional[bool]
        """
        return self._finished



    @property
    @abstractmethod
    def force_switch(self) -> Any:
        pass



    @property
    def lost(self) -> Optional[bool]:
        """
        :return: If the battle is finished, a boolean indicating whether the battle is
            lost. Otherwise None.
        :rtype: Optional[bool]
        """
        return None if self._won is None else not self._won



    @property
    # 返回团队预览中可接受的最大团队大小，如果适用的话
    def max_team_size(self) -> Optional[int]:
        return self._max_team_size

    # 抽象方法，可能被困住的情况
    @property
    @abstractmethod
    def maybe_trapped(self) -> Any:
        pass

    # 抽象方法，对手的当前激活精灵
    @property
    @abstractmethod
    def opponent_active_pokemon(self) -> Any:
        pass

    # 抽象方法，对手是否可以激活极巨化
    @property
    @abstractmethod
    def opponent_can_dynamax(self) -> Any:
        pass

    # 设置对手是否可以激活极巨化
    @opponent_can_dynamax.setter
    @abstractmethod
    def opponent_can_dynamax(self, value: bool) -> Any:
        pass

    # 返回对手的精灵剩余的极巨化回合数
    @property
    def opponent_dynamax_turns_left(self) -> Optional[int]:
        if self._opponent_dynamax_turn is not None and any(
            map(lambda pokemon: pokemon.is_dynamaxed, self._opponent_team.values())
        ):
            return max(3 - (self.turn - self._opponent_dynamax_turn), 0)

    # 返回对手的角色在给定的战斗中，p1 或 p2
    @property
    def opponent_role(self) -> Optional[str]:
        if self.player_role == "p1":
            return "p2"
        if self.player_role == "p2":
            return "p1"

    # 返回对手的场地状态
    @property
    def opponent_side_conditions(self) -> Dict[SideCondition, int]:
        return self._opponent_side_conditions
    def opponent_team(self) -> Dict[str, Pokemon]:
        """
        During teampreview, keys are not definitive: please rely on values.

        :return: The opponent's team. Keys are identifiers, values are pokemon objects.
        :rtype: Dict[str, Pokemon]
        """
        # 如果已经存在对手队伍信息，则直接返回
        if self._opponent_team:
            return self._opponent_team
        else:
            # 否则根据对手队伍预览信息生成对手队伍字典并返回
            return {mon.species: mon for mon in self._teampreview_opponent_team}

    @property
    def opponent_username(self) -> Optional[str]:
        """
        :return: The opponent's username, or None if unknown.
        :rtype: str, optional.
        """
        # 返回对手的用户名，如果未知则返回 None
        return self._opponent_username

    @opponent_username.setter
    def opponent_username(self, value: str):
        # 设置对手的用户名
        self._opponent_username = value

    @property
    def player_role(self) -> Optional[str]:
        """
        :return: Player's role in given battle. p1/p2
        :rtype: str, optional
        """
        # 返回玩家在战斗中的角色，可能是 p1 或 p2
        return self._player_role

    @player_role.setter
    def player_role(self, value: Optional[str]):
        # 设置玩家在战斗中的角色
        self._player_role = value

    @property
    def player_username(self) -> str:
        """
        :return: The player's username.
        :rtype: str
        """
        # 返回玩家的用户名
        return self._player_username

    @player_username.setter
    def player_username(self, value: str):
        # 设置玩家的用户名
        self._player_username = value

    @property
    def players(self) -> Tuple[str, str]:
        """
        :return: The pair of players' usernames.
        :rtype: Tuple[str, str]
        """
        # 返回玩家对的用户名组成的元组
        return self._players[0]["username"], self._players[1]["username"]

    @players.setter
    def players(self, players: Tuple[str, str]):
        """Sets the battle player's name:

        :param player_1: First player's username.
        :type player_1: str
        :param player_1: Second player's username.
        :type player_2: str
        """
        # 解包玩家名称元组
        player_1, player_2 = players
        # 根据当前玩家用户名设置对手用户名
        if player_1 != self._player_username:
            self._opponent_username = player_1
        else:
            self._opponent_username = player_2

    @property
    def rating(self) -> Optional[int]:
        """
        Player's rating after the end of the battle, if it was received.

        :return: The player's rating after the end of the battle.
        :rtype: int, optional
        """
        # 返回玩家战斗结束后的评分
        return self._rating

    @property
    def opponent_rating(self) -> Optional[int]:
        """
        Opponent's rating after the end of the battle, if it was received.

        :return: The opponent's rating after the end of the battle.
        :rtype: int, optional
        """
        # 返回对手战斗结束后的评分
        return self._opponent_rating

    @property
    def rqid(self) -> int:
        """
        Should not be used.

        :return: The last request's rqid.
        :rtype: Tuple[str, str]
        """
        # 不应该使用，返回最后一个请求的 rqid
        return self._rqid

    @property
    def side_conditions(self) -> Dict[SideCondition, int]:
        """
        :return: The player's side conditions. Keys are SideCondition objects, values
            are:

            - the number of layers of the side condition if the side condition is
                stackable
            - the turn where the SideCondition was setup otherwise
        :rtype: Dict[SideCondition, int]
        """
        # 返回玩家的边界条件，键为 SideCondition 对象，值为边界条件的层数或设置边界条件的回合数
        return self._side_conditions

    @property
    def team(self) -> Dict[str, Pokemon]:
        """
        :return: The player's team. Keys are identifiers, values are pokemon objects.
        :rtype: Dict[str, Pokemon]
        """
        # 返回玩家的队伍，键为标识符，值为 Pokemon 对象
        return self._team

    @team.setter
    def team(self, value: Dict[str, Pokemon]):
        # 设置玩家的队伍
        self._team = value
    @property
    def team_size(self) -> int:
        """
        :return: The number of Pokemon in the player's team.
        :rtype: int
        """
        # 返回玩家队伍中的 Pokemon 数量
        if self._player_role is not None:
            return self._team_size[self._player_role]
        # 如果没有分配玩家角色，则引发 ValueError
        raise ValueError(
            "Team size cannot be inferred without an assigned player role."
        )

    @property
    def teampreview(self) -> bool:
        """
        :return: Whether the battle is awaiting a teampreview order.
        :rtype: bool
        """
        # 返回战斗是否等待 teampreview 命令
        return self._teampreview

    @property
    @abstractmethod
    def trapped(self) -> Any:
        pass

    @trapped.setter
    @abstractmethod
    def trapped(self, value: Any):
        pass

    @property
    def turn(self) -> int:
        """
        :return: The current battle turn.
        :rtype: int
        """
        # 返回当前战斗回合数
        return self._turn

    @turn.setter
    def turn(self, turn: int):
        """Sets the current turn counter to given value.

        :param turn: Current turn value.
        :type turn: int
        """
        # 将当前回合计数器设置为给定值
        self._turn = turn

    @property
    def weather(self) -> Dict[Weather, int]:
        """
        :return: A Dict mapping the battle's weather (if any) to its starting turn
        :rtype: Dict[Weather, int]
        """
        # 返回将战斗天气（如果有）映射到其起始回合的字典
        return self._weather

    @property
    def won(self) -> Optional[bool]:
        """
        :return: If the battle is finished, a boolean indicating whether the battle is
            won. Otherwise None.
        :rtype: Optional[bool]
        """
        # 如果战斗结束，返回一个布尔值指示战斗是否获胜，否则返回 None
        return self._won

    @property
    def move_on_next_request(self) -> bool:
        """
        :return: Whether the next received request should yield a move order directly.
            This can happen when a switch is forced, or an error is encountered.
        :rtype: bool
        """
        # 返回下一个接收到的请求是否应直接产生移动顺序
        # 当强制切换或遇到错误时会发生这种情况
        return self._move_on_next_request

    @move_on_next_request.setter
    # 设置是否继续处理下一个请求的标志位
    def move_on_next_request(self, value: bool):
        # 将传入的布尔值赋给私有属性 _move_on_next_request
        self._move_on_next_request = value

    # 获取是否正在恢复的属性
    @property
    def reviving(self) -> bool:
        # 返回私有属性 _reviving 的布尔值
        return self._reviving
```