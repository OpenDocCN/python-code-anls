# `.\PokeLLMon\poke_env\environment\battle.py`

```py
# 导入所需模块
from logging import Logger
from typing import Any, Dict, List, Optional, Union

# 导入自定义模块
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType

# 定义 Battle 类，继承自 AbstractBattle 类
class Battle(AbstractBattle):
    # 初始化方法
    def __init__(
        self,
        battle_tag: str,
        username: str,
        logger: Logger,
        gen: int,
        save_replays: Union[str, bool] = False,
    ):
        # 调用父类的初始化方法
        super(Battle, self).__init__(battle_tag, username, logger, save_replays, gen)

        # 初始化回合选择属性
        self._available_moves: List[Move] = []
        self._available_switches: List[Pokemon] = []
        self._can_dynamax: bool = False
        self._can_mega_evolve: bool = False
        self._can_tera: Optional[PokemonType] = None
        self._can_z_move: bool = False
        self._opponent_can_dynamax = True
        self._opponent_can_mega_evolve = True
        self._opponent_can_z_move = True
        self._opponent_can_tera: bool = False
        self._force_switch: bool = False
        self._maybe_trapped: bool = False
        self._trapped: bool = False

        # 初始化属性
        self.battle_msg_history = ""
        self.pokemon_hp_log_dict = {}
        self.speed_list = []

    # 清除所有属性提升
    def clear_all_boosts(self):
        if self.active_pokemon is not None:
            self.active_pokemon.clear_boosts()
        if self.opponent_active_pokemon is not None:
            self.opponent_active_pokemon.clear_boosts()

    # 结束幻象状态
    def end_illusion(self, pokemon_name: str, details: str):
        # 根据角色名判断幻象状态的 Pokemon
        if pokemon_name[:2] == self._player_role:
            active = self.active_pokemon
        else:
            active = self.opponent_active_pokemon

        # 如果没有活跃的 Pokemon，则抛出异常
        if active is None:
            raise ValueError("Cannot end illusion without an active pokemon.")

        # 结束幻象状态
        self._end_illusion_on(
            illusioned=active, illusionist=pokemon_name, details=details
        )
    def switch(self, pokemon_str: str, details: str, hp_status: str):
        # 从传入的字符串中提取精灵标识符
        identifier = pokemon_str.split(":")[0][:2]

        # 如果标识符与玩家角色相同
        if identifier == self._player_role:
            # 如果存在活跃的精灵，让其退出战斗
            if self.active_pokemon:
                self.active_pokemon.switch_out()
        else:
            # 如果对手存在活跃的精灵，让其退出战斗
            if self.opponent_active_pokemon:
                self.opponent_active_pokemon.switch_out()

        # 获取指定的精灵对象
        pokemon = self.get_pokemon(pokemon_str, details=details)

        # 让指定的精灵进入战斗，并设置其血量状态
        pokemon.switch_in(details=details)
        pokemon.set_hp_status(hp_status)

    @property
    def active_pokemon(self) -> Optional[Pokemon]:
        """
        :return: 活跃的精灵
        :rtype: Optional[Pokemon]
        """
        # 返回队伍中活跃的精灵
        for pokemon in self.team.values():
            if pokemon.active:
                return pokemon

    @property
    def all_active_pokemons(self) -> List[Optional[Pokemon]]:
        """
        :return: 包含所有活跃精灵和/或 None 的列表
        :rtype: List[Optional[Pokemon]
        """
        # 返回包含玩家和对手活跃精灵的列表
        return [self.active_pokemon, self.opponent_active_pokemon]

    @property
    def available_moves(self) -> List[Move]:
        """
        :return: 玩家可以在当前回合使用的招式列表
        :rtype: List[Move]
        """
        # 返回玩家可以使用的招式列表
        return self._available_moves

    @property
    def available_switches(self) -> List[Pokemon]:
        """
        :return: 玩家可以在当前回合进行的替换列表
        :rtype: List[Pokemon]
        """
        # 返回玩家可以进行的替换列表
        return self._available_switches

    @property
    def can_dynamax(self) -> bool:
        """
        :return: 当前活跃精灵是否可以极巨化
        :rtype: bool
        """
        # 返回当前活跃精灵是否可以进行极巨化
        return self._can_dynamax

    @property
    def can_mega_evolve(self) -> bool:
        """
        :return: 当前活跃精灵是否可以超级进化
        :rtype: bool
        """
        # 返回当前活跃精灵是否可以进行超级进化
        return self._can_mega_evolve

    @property
    def can_tera(self) -> Optional[PokemonType]:
        """
        :return: None, or the type the active pokemon can terastallize into.
        :rtype: PokemonType, optional
        """
        # 返回当前活跃宝可梦可以转变成的类型，如果不能则返回 None
        return self._can_tera

    @property
    def can_z_move(self) -> bool:
        """
        :return: Whether or not the current active pokemon can z-move.
        :rtype: bool
        """
        # 返回当前活跃宝可梦是否可以使用 Z 招式
        return self._can_z_move

    @property
    def force_switch(self) -> bool:
        """
        :return: A boolean indicating whether the active pokemon is forced to switch
            out.
        :rtype: Optional[bool]
        """
        # 返回一个布尔值，指示当前活跃宝可梦是否被迫交换出场
        return self._force_switch

    @property
    def maybe_trapped(self) -> bool:
        """
        :return: A boolean indicating whether the active pokemon is maybe trapped by the
            opponent.
        :rtype: bool
        """
        # 返回一个布尔值，指示当前活跃宝可梦是否可能被对手困住
        return self._maybe_trapped

    @property
    def opponent_active_pokemon(self) -> Optional[Pokemon]:
        """
        :return: The opponent active pokemon
        :rtype: Pokemon
        """
        # 返回对手当前活跃的宝可梦
        for pokemon in self.opponent_team.values():
            if pokemon.active:
                return pokemon
        return None

    @property
    def opponent_can_dynamax(self) -> bool:
        """
        :return: Whether or not opponent's current active pokemon can dynamax
        :rtype: bool
        """
        # 返回对手当前活跃的宝可梦是否可以极巨化
        return self._opponent_can_dynamax

    @opponent_can_dynamax.setter
    def opponent_can_dynamax(self, value: bool):
        self._opponent_can_dynamax = value

    @property
    def opponent_can_mega_evolve(self) -> bool:
        """
        :return: Whether or not opponent's current active pokemon can mega-evolve
        :rtype: bool
        """
        # 返回对手当前活跃的宝可梦是否可以超级进化
        return self._opponent_can_mega_evolve

    @opponent_can_mega_evolve.setter
    def opponent_can_mega_evolve(self, value: bool):
        self._opponent_can_mega_evolve = value
    def opponent_can_tera(self) -> bool:
        """
        :return: Whether or not opponent's current active pokemon can terastallize
        :rtype: bool
        """
        # 返回对手当前激活的宝可梦是否可以使用 terastallize
        return self._opponent_can_tera

    @property
    def opponent_can_z_move(self) -> bool:
        """
        :return: Whether or not opponent's current active pokemon can z-move
        :rtype: bool
        """
        # 返回对手当前激活的宝可梦是否可以使用 z-move
        return self._opponent_can_z_move

    @opponent_can_z_move.setter
    def opponent_can_z_move(self, value: bool):
        # 设置对手当前激活的宝可梦是否可以使用 z-move
        self._opponent_can_z_move = value

    @property
    def trapped(self) -> bool:
        """
        :return: A boolean indicating whether the active pokemon is trapped, either by
            the opponent or as a side effect of one your moves.
        :rtype: bool
        """
        # 返回一个布尔值，指示激活的宝可梦是否被困住，无论是被对手困住还是作为你的招式的副作用
        return self._trapped

    @trapped.setter
    def trapped(self, value: bool):
        # 设置激活的宝可梦是否被困住
        self._trapped = value
```