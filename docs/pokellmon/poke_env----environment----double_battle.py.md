# `.\PokeLLMon\poke_env\environment\double_battle.py`

```
# 从 logging 模块中导入 Logger 类
from logging import Logger
# 从 typing 模块中导入 Any, Dict, List, Optional, Union 类型
from typing import Any, Dict, List, Optional, Union

# 从 poke_env.environment.abstract_battle 模块中导入 AbstractBattle 类
from poke_env.environment.abstract_battle import AbstractBattle
# 从 poke_env.environment.move 模块中导入 SPECIAL_MOVES, Move 类
from poke_env.environment.move import SPECIAL_MOVES, Move
# 从 poke_env.environment.move_category 模块中导入 MoveCategory 类
from poke_env.environment.move_category import MoveCategory
# 从 poke_env.environment.pokemon 模块中导入 Pokemon 类
from poke_env.environment.pokemon import Pokemon
# 从 poke_env.environment.pokemon_type 模块中导入 PokemonType 类
from poke_env.environment.pokemon_type import PokemonType

# 定义 DoubleBattle 类，继承自 AbstractBattle 类
class DoubleBattle(AbstractBattle):
    # 定义常量 POKEMON_1_POSITION 为 -1
    POKEMON_1_POSITION = -1
    # 定义常量 POKEMON_2_POSITION 为 -2
    POKEMON_2_POSITION = -2
    # 定义常量 OPPONENT_1_POSITION 为 1
    OPPONENT_1_POSITION = 1
    # 定义常量 OPPONENT_2_POSITION 为 2
    OPPONENT_2_POSITION = 2
    # 定义常量 EMPTY_TARGET_POSITION 为 0，仅为符号，不被 showdown 使用

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
        super(DoubleBattle, self).__init__(
            battle_tag, username, logger, save_replays, gen=gen
        )

        # 初始化回合选择属性
        self._available_moves: List[List[Move]] = [[], []]
        self._available_switches: List[List[Pokemon]] = [[], []]
        self._can_mega_evolve: List[bool] = [False, False]
        self._can_z_move: List[bool] = [False, False]
        self._can_dynamax: List[bool] = [False, False]
        self._can_tera: List[Union[bool, PokemonType]] = [False, False]
        self._opponent_can_dynamax: List[bool] = [True, True]
        self._opponent_can_mega_evolve: List[bool] = [True, True]
        self._opponent_can_z_move: List[bool] = [True, True]
        self._force_switch: List[bool] = [False, False]
        self._maybe_trapped: List[bool] = [False, False]
        self._trapped: List[bool] = [False, False]

        # 初始化战斗状态属性
        self._active_pokemon: Dict[str, Pokemon] = {}
        self._opponent_active_pokemon: Dict[str, Pokemon] = {}

        # 其他属性
        self._move_to_pokemon_id: Dict[Move, str] = {}
    # 清除所有精灵的增益效果
    def clear_all_boosts(self):
        # 遍历己方和对手的所有精灵
        for active_pokemon_group in (self.active_pokemon, self.opponent_active_pokemon):
            for active_pokemon in active_pokemon_group:
                # 如果精灵存在，则清除其增益效果
                if active_pokemon is not None:
                    active_pokemon.clear_boosts()

    # 结束幻象状态
    def end_illusion(self, pokemon_name: str, details: str):
        # 获取玩家标识符和精灵标识符
        player_identifier = pokemon_name[:2]
        pokemon_identifier = pokemon_name[:3]
        # 根据玩家标识符确定要操作的精灵字典
        if player_identifier == self._player_role:
            active_dict = self._active_pokemon
        else:
            active_dict = self._opponent_active_pokemon
        # 获取要结束幻象状态的精灵
        active = active_dict.get(pokemon_identifier)
        
        # 在指定精灵上结束幻象状态
        active_dict[pokemon_identifier] = self._end_illusion_on(
            illusioned=active, illusionist=pokemon_name, details=details
        )

    # 获取活跃的精灵
    @staticmethod
    def _get_active_pokemon(
        active_pokemon: Dict[str, Pokemon], role: str
    ) -> List[Optional[Pokemon]]:
        # 获取指定角色的两只精灵
        pokemon_1 = active_pokemon.get(f"{role}a")
        pokemon_2 = active_pokemon.get(f"{role}b")
        # 如果第一只精灵不存在或不活跃或已经倒下，则置为None
        if pokemon_1 is None or not pokemon_1.active or pokemon_1.fainted:
            pokemon_1 = None
        # 如果第二只精灵不存在或不活跃或已经倒下，则置为None
        if pokemon_2 is None or not pokemon_2.active or pokemon_2.fainted:
            pokemon_2 = None
        # 返回两只精灵的列表
        return [pokemon_1, pokemon_2]

    # 精灵交换
    def switch(self, pokemon_str: str, details: str, hp_status: str):
        # 获取精灵标识符和玩家标识符
        pokemon_identifier = pokemon_str.split(":")[0][:3]
        player_identifier = pokemon_identifier[:2]
        # 根据玩家标识符确定要操作的精灵字典
        team = (
            self._active_pokemon
            if player_identifier == self._player_role
            else self._opponent_active_pokemon
        )
        # 弹出要交换出去的精灵
        pokemon_out = team.pop(pokemon_identifier, None)
        # 如果精灵存在，则执行交换出去的操作
        if pokemon_out is not None:
            pokemon_out.switch_out()
        # 获取要交换进来的精灵
        pokemon_in = self.get_pokemon(pokemon_str, details=details)
        # 执行交换进来的操作
        pokemon_in.switch_in()
        # 设置精灵的血量状态
        pokemon_in.set_hp_status(hp_status)
        # 将新的精灵放入对应的精灵字典中
        team[pokemon_identifier] = pokemon_in
    # 交换指定精灵和指定槽位的精灵
    def _swap(self, pokemon_str: str, slot: str):
        # 获取玩家标识符
        player_identifier = pokemon_str.split(":")[0][:2]
        # 根据玩家标识符确定当前活跃的精灵
        active = (
            self._active_pokemon
            if player_identifier == self.player_role
            else self._opponent_active_pokemon
        )
        # 确定玩家槽位标识符
        slot_a = f"{player_identifier}a"
        slot_b = f"{player_identifier}b"

        # 如果槽位A或槽位B的精灵已经倒下，则不执行交换操作
        if active[slot_a].fainted or active[slot_b].fainted:
            return

        # 获取槽位A和槽位B的精灵
        slot_a_mon = active[slot_a]
        slot_b_mon = active[slot_b]

        # 获取指定的精灵
        pokemon = self.get_pokemon(pokemon_str)

        # 如果槽位为0且精灵为槽位A的精灵，或者槽位为1且精灵为槽位B的精灵，则不执行交换操作
        if (slot == "0" and pokemon == slot_a_mon) or (
            slot == "1" and pokemon == slot_b_mon
        ):
            pass
        else:
            # 交换槽位A和槽位B的精灵
            active[slot_a], active[slot_b] = active[slot_b], active[slot_a]

    # 获取可能的对战目标
    def get_possible_showdown_targets(
        self, move: Move, pokemon: Pokemon, dynamax: bool = False
    @property
    # 返回当前活跃的精灵列表，至少有一个不为None
    def active_pokemon(self) -> List[Optional[Pokemon]]:
        """
        :return: The active pokemon, always at least one is not None
        :rtype: List[Optional[Pokemon]]
        """
        if self.player_role is None:
            raise ValueError("Unable to get active_pokemon, player_role is None")
        return self._get_active_pokemon(self._active_pokemon, self.player_role)

    @property
    # 返回所有活跃的精灵列表，包括玩家和对手的
    def all_active_pokemons(self) -> List[Optional[Pokemon]]:
        """
        :return: A list containing all active pokemons and/or Nones.
        :rtype: List[Optional[Pokemon]]
        """
        return [*self.active_pokemon, *self.opponent_active_pokemon]

    @property
    # 返回可用的招式列表
    def available_moves(self) -> List[List[Move]]:
        """
        :return: A list of two lists of moves the player can use during the current
            move request for each Pokemon.
        :rtype: List[List[Move]]
        """
        return self._available_moves

    @property
    def available_switches(self) -> List[List[Pokemon]]:
        """
        :return: The list of two lists of switches the player can do during the
            current move request for each active pokemon
        :rtype: List[List[Pokemon]]
        """
        # 返回玩家在当前移动请求期间每个活跃精灵可以执行的两个交换列表
        return self._available_switches

    @property
    def can_mega_evolve(self) -> List[bool]:
        """
        :return: Whether or not either current active pokemon can mega evolve.
        :rtype: List[bool]
        """
        # 返回当前活跃精灵是否可以超级进化的布尔值列表
        return self._can_mega_evolve

    @property
    def can_z_move(self) -> List[bool]:
        """
        :return: Whether or not the current active pokemon can z-move.
        :rtype: List[bool]
        """
        # 返回当前活跃精灵是否可以使用 Z 招式的布尔值列表
        return self._can_z_move

    @property
    def can_dynamax(self) -> List[bool]:
        """
        :return: Whether or not the current active pokemon can dynamax
        :rtype: List[bool]
        """
        # 返回当前活跃精灵是否可以极巨化的布尔值列表
        return self._can_dynamax

    @property
    def can_tera(self) -> List[Union[bool, PokemonType]]:
        """
        :return: Whether or not the current active pokemon can terastallize. If yes, will be a PokemonType.
        :rtype: List[Union[bool, PokemonType]]
        """
        # 返回当前活跃精灵是否可以进行特拉斯化的布尔值列表，如果可以，将是一个 PokemonType
        return self._can_tera

    @property
    def force_switch(self) -> List[bool]:
        """
        :return: A boolean indicating whether the active pokemon is forced
            to switch out.
        :rtype: List[bool]
        """
        # 返回一个布尔值，指示活跃精灵是否被迫交换出场
        return self._force_switch

    @property
    def maybe_trapped(self) -> List[bool]:
        """
        :return: A boolean indicating whether either active pokemon is maybe trapped
            by the opponent.
        :rtype: List[bool]
        """
        # 返回一个布尔值，指示任一活跃精灵是否可能被对手困住
        return self._maybe_trapped

    @property
    def opponent_active_pokemon(self) -> List[Optional[Pokemon]]:
        """
        :return: The opponent active pokemon, always at least one is not None
        :rtype: List[Optional[Pokemon]]
        """
        # 如果对手角色为空，则抛出数值错误异常
        if self.opponent_role is None:
            raise ValueError(
                "Unable to get opponent_active_pokemon, opponent_role is None"
            )
        # 返回对手当前活跃的宝可梦列表
        return self._get_active_pokemon(
            self._opponent_active_pokemon, self.opponent_role
        )

    @property
    def opponent_can_dynamax(self) -> List[bool]:
        """
        :return: Whether or not opponent's current active pokemons can dynamax
        :rtype: List[bool]
        """
        # 返回对手当前活跃的宝可梦是否可以极巨化的列表
        return self._opponent_can_dynamax

    @opponent_can_dynamax.setter
    def opponent_can_dynamax(self, value: Union[bool, List[bool]]):
        # 如果值是布尔类型，则设置对手当前活跃的宝可梦是否可以极巨化的列表为相同的值
        if isinstance(value, bool):
            self._opponent_can_dynamax = [value, value]
        else:
            self._opponent_can_dynamax = value

    @property
    def opponent_can_mega_evolve(self) -> List[bool]:
        """
        :return: Whether or not opponent's current active pokemons can mega evolve
        :rtype: List[bool]
        """
        # 返回对手当前活跃的宝可梦是否可以超级进化的列表
        return self._opponent_can_mega_evolve

    @opponent_can_mega_evolve.setter
    def opponent_can_mega_evolve(self, value: Union[bool, List[bool]]):
        # 如果值是布尔类型，则设置对手当前活跃的宝可梦是否可以超级进化的列表为相同的值
        if isinstance(value, bool):
            self._opponent_can_mega_evolve = [value, value]
        else:
            self._opponent_can_mega_evolve = value

    @property
    def opponent_can_z_move(self) -> List[bool]:
        """
        :return: Whether or not opponent's current active pokemons can z-move
        :rtype: List[bool]
        """
        # 返回对手当前活跃的宝可梦是否可以Z招式的列表
        return self._opponent_can_z_move

    @opponent_can_z_move.setter
    def opponent_can_z_move(self, value: Union[bool, List[bool]]):
        # 如果值是布尔类型，则设置对手当前活跃的宝可梦是否可以Z招式的列表为相同的值
        if isinstance(value, bool):
            self._opponent_can_z_move = [value, value]
        else:
            self._opponent_can_z_move = value

    @property
    # 返回一个布尔列表，指示当前双方是否有精灵被对手困住
    def trapped(self) -> List[bool]:
        """
        :return: A boolean indicating whether either active pokemon is trapped by the
            opponent.
        :rtype: List[bool]
        """
        # 返回私有属性_trapped
        return self._trapped

    # 设置trapped属性的setter方法
    @trapped.setter
    def trapped(self, value: List[bool]):
        # 将传入的值赋给私有属性_trapped
        self._trapped = value
```