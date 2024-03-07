# `.\PokeLLMon\poke_env\environment\move.py`

```
# 导入 copy 模块，用于深拷贝对象
import copy
# 导入 lru_cache 装饰器，用于缓存函数的结果
from functools import lru_cache
# 导入 Any、Dict、List、Optional、Set、Tuple、Union 类型提示
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# 导入 GenData、to_id_str 函数
from poke_env.data import GenData, to_id_str
# 导入 Field 类
from poke_env.environment.field import Field
# 导入 MoveCategory 枚举
from poke_env.environment.move_category import MoveCategory
# 导入 PokemonType 枚举
from poke_env.environment.pokemon_type import PokemonType
# 导入 Status 枚举
from poke_env.environment.status import Status
# 导入 Weather 枚举
from poke_env.environment.weather import Weather

# 特殊招式集合，包含 "struggle" 和 "recharge"
SPECIAL_MOVES: Set[str] = {"struggle", "recharge"}

# 保护招式集合
_PROTECT_MOVES = {
    "protect",
    "detect",
    "endure",
    "spikyshield",
    "kingsshield",
    "banefulbunker",
    "obstruct",
    "maxguard",
}
# 辅助保护招式集合
_SIDE_PROTECT_MOVES = {"wideguard", "quickguard", "matblock"}
# 总保护招式集合，包含保护招式和辅助保护招式
_PROTECT_COUNTER_MOVES = _PROTECT_MOVES | _SIDE_PROTECT_MOVES

# 招式类
class Move:
    # 杂项标志列表
    _MISC_FLAGS = [
        "onModifyMove",
        "onEffectiveness",
        "onHitField",
        "onAfterMoveSecondarySelf",
        "onHit",
        "onTry",
        "beforeTurnCallback",
        "onAfterMove",
        "onTryHit",
        "onTryMove",
        "hasCustomRecoil",
        "onMoveFail",
        "onPrepareHit",
        "onAfterHit",
        "onBasePower",
        "basePowerCallback",
        "damageCallback",
        "onTryHitSide",
        "beforeMoveCallback",
    ]
    # 每种宝可梦类型对应的招式类别预分配字典
    _MOVE_CATEGORY_PER_TYPE_PRE_SPLIT = {
        PokemonType.BUG: MoveCategory.PHYSICAL,
        PokemonType.DARK: MoveCategory.SPECIAL,
        PokemonType.DRAGON: MoveCategory.SPECIAL,
        PokemonType.ELECTRIC: MoveCategory.SPECIAL,
        PokemonType.FIGHTING: MoveCategory.PHYSICAL,
        PokemonType.FIRE: MoveCategory.SPECIAL,
        PokemonType.FLYING: MoveCategory.PHYSICAL,
        PokemonType.GHOST: MoveCategory.PHYSICAL,
        PokemonType.GRASS: MoveCategory.SPECIAL,
        PokemonType.GROUND: MoveCategory.PHYSICAL,
        PokemonType.ICE: MoveCategory.SPECIAL,
        PokemonType.NORMAL: MoveCategory.PHYSICAL,
        PokemonType.POISON: MoveCategory.PHYSICAL,
        PokemonType.PSYCHIC: MoveCategory.SPECIAL,
        PokemonType.ROCK: MoveCategory.PHYSICAL,
        PokemonType.STEEL: MoveCategory.PHYSICAL,
        PokemonType.WATER: MoveCategory.SPECIAL,
    }

    # 招式对象的属性列表
    __slots__ = (
        "_id",
        "_base_power_override",
        "_current_pp",
        "_dynamaxed_move",
        "_gen",
        "_is_empty",
        "_moves_dict",
        "_request_target",
    )

    # 初始化招式对象
    def __init__(self, move_id: str, gen: int, raw_id: Optional[str] = None):
        self._id = move_id
        self._base_power_override = None
        self._gen = gen
        self._moves_dict = GenData.from_gen(gen).moves

        # 处理隐藏力招式
        if move_id.startswith("hiddenpower") and raw_id is not None:
            base_power = "".join([c for c in raw_id if c.isdigit()])
            self._id = "".join([c for c in to_id_str(raw_id) if not c.isdigit()])

            if base_power:
                try:
                    base_power = int(base_power)
                    self._base_power_override = base_power
                except ValueError:
                    pass

        self._current_pp = self.max_pp
        self._is_empty: bool = False

        self._dynamaxed_move = None
        self._request_target = None

    # 返回招式对象的字符串表示
    def __repr__(self) -> str:
        return f"{self._id} (Move object)"
    # 减少当前 PP 值
    def use(self):
        self._current_pp -= 1

    # 判断给定的招式 ID 是否为 Z 招式
    @staticmethod
    def is_id_z(id_: str, gen: int) -> bool:
        if id_.startswith("z") and id_[1:] in GenData.from_gen(gen).moves:
            return True
        return "isZ" in GenData.from_gen(gen).moves[id_]

    # 判断给定的招式 ID 是否为 Max 招式
    @staticmethod
    def is_max_move(id_: str, gen: int) -> bool:
        if id_.startswith("max"):
            return True
        elif (
            GenData.from_gen(gen).moves[id_].get("isNonstandard", None) == "Gigantamax"
        ):
            return True
        elif GenData.from_gen(gen).moves[id_].get("isMax", None) is not None:
            return True
        return False

    # 判断给定的招式是否应该被存储
    @staticmethod
    @lru_cache(4096)
    def should_be_stored(move_id: str, gen: int) -> bool:
        if move_id in SPECIAL_MOVES:
            return False
        if move_id not in GenData.from_gen(gen).moves:
            return False
        if Move.is_id_z(move_id, gen):
            return False
        if Move.is_max_move(move_id, gen):
            return False
        return True

    # 获取招式的命中率（0 到 1 的范围）
    @property
    def accuracy(self) -> float:
        """
        :return: The move's accuracy (0 to 1 scale).
        :rtype: float
        """
        accuracy = self.entry["accuracy"]
        if accuracy is True:
            return 1.0
        return accuracy / 100

    # 获取招式的基础威力
    @property
    def base_power(self) -> int:
        """
        :return: The move's base power.
        :rtype: int
        """
        if self._base_power_override is not None:
            return self._base_power_override
        return self.entry.get("basePower", 0)

    # 获取招式对目标的增益效果
    @property
    def boosts(self) -> Optional[Dict[str, int]]:
        """
        :return: Boosts conferred to the target by using the move.
        :rtype: Dict[str, float] | None
        """
        return self.entry.get("boosts", None)
    # 返回移动是否打破类似保护的防御
    def breaks_protect(self) -> bool:
        """
        :return: Whether the move breaks proect-like defenses.
        :rtype: bool
        """
        return self.entry.get("breaksProtect", False)

    # 返回是否存在此移动的 Z-移动版本
    @property
    def can_z_move(self) -> bool:
        """
        :return: Wheter there exist a z-move version of this move.
        :rtype: bool
        """
        return self.id not in SPECIAL_MOVES

    # 返回移动的类别
    @property
    def category(self) -> MoveCategory:
        """
        :return: The move category.
        :rtype: MoveCategory
        """
        # 如果条目中没有类别信息，则打印出当前移动和条目
        if "category" not in self.entry:
            print(self, self.entry)

        # 如果世代小于等于3且类别为"PHYSICAL"或"SPECIAL"，则返回对应的移动类别
        if self._gen <= 3 and self.entry["category"].upper() in {
            "PHYSICAL",
            "SPECIAL",
        }:
            return self._MOVE_CATEGORY_PER_TYPE_PRE_SPLIT[self.type]
        return MoveCategory[self.entry["category"].upper()]

    # 返回移动的暴击率。如果移动保证暴击，则返回6
    @property
    def crit_ratio(self) -> int:
        """
        :return: The move's crit ratio. If the move is guaranteed to crit, returns 6.
        :rtype:
        """
        if "critRatio" in self.entry:
            return int(self.entry["critRatio"])
        elif "willCrit" in self.entry:
            return 6
        return 0

    # 返回当前剩余的PP
    @property
    def current_pp(self) -> int:
        """
        :return: Remaining PP.
        :rtype: int
        """
        return self._current_pp

    # 返回移动的固定伤害。可以是整数或'level'，例如Seismic Toss
    @property
    def damage(self) -> Union[int, str]:
        """
        :return: The move's fix damages. Can be an int or 'level' for moves such as
            Seismic Toss.
        :rtype: Union[int, str]
        """
        return self.entry.get("damage", 0)
    def deduced_target(self) -> Optional[str]:
        """
        :return: Move deduced target, based on Move.target and showdown's request
            messages.
        :rtype: str, optional
        """
        # 如果移动在特殊移动列表中，则返回移动的目标
        if self.id in SPECIAL_MOVES:
            return self.target
        # 如果有请求目标，则返回请求目标
        elif self.request_target:
            return self.request_target
        # 如果目标是"randomNormal"，则返回请求目标
        elif self.target == "randomNormal":
            return self.request_target
        # 否则返回移动的目标
        return self.target

    @property
    def defensive_category(self) -> MoveCategory:
        """
        :return: Move's defender category.
        :rtype: MoveCategory
        """
        # 如果存在"overrideDefensiveStat"在条目中
        if "overrideDefensiveStat" in self.entry:
            # 如果"overrideDefensiveStat"为"def"，则返回物理类别
            if self.entry["overrideDefensiveStat"] == "def":
                return MoveCategory["PHYSICAL"]
            # 如果"overrideDefensiveStat"为"spd"，则返回特殊类别
            elif self.entry["overrideDefensiveStat"] == "spd":
                return MoveCategory["SPECIAL"]
            # 否则抛出值错误异常
            else:
                raise ValueError(
                    f"Unsupported value for overrideDefensiveStat: {self.entry['overrideDefensiveStat']}"
                )
        # 否则返回移动的类别
        return self.category

    @property
    def drain(self) -> float:
        """
        :return: Ratio of HP of inflicted damages, between 0 and 1.
        :rtype: float
        """
        # 如果存在"drain"在条目中，则返回伤害的HP比例
        if "drain" in self.entry:
            return self.entry["drain"][0] / self.entry["drain"][1]
        # 否则返回0.0
        return 0.0

    @property
    def dynamaxed(self):
        """
        :return: The dynamaxed version of the move.
        :rtype: DynamaxMove
        """
        # 如果存在动态移动，则返回动态移动
        if self._dynamaxed_move:
            return self._dynamaxed_move

        # 否则创建动态移动并返回
        self._dynamaxed_move = DynamaxMove(self)
        return self._dynamaxed_move

    @property
    def entry(self) -> Dict[str, Any]:
        """
        Should not be used directly.

        :return: The data entry corresponding to the move
        :rtype: Dict
        """
        # 检查是否存在与当前移动ID对应的数据条目，如果存在则返回该数据条目
        if self._id in self._moves_dict:
            return self._moves_dict[self._id]
        # 如果移动ID以"z"开头且去掉"z"后的部分在数据字典中存在，则返回对应的数据条目
        elif self._id.startswith("z") and self._id[1:] in self._moves_dict:
            return self._moves_dict[self._id[1:]]
        # 如果移动ID为"recharge"，则返回一个预定义的数据条目
        elif self._id == "recharge":
            return {"pp": 1, "type": "normal", "category": "Special", "accuracy": 1}
        else:
            # 抛出数值错误，表示未知的移动ID
            raise ValueError("Unknown move: %s" % self._id)

    @property
    def expected_hits(self) -> float:
        """
        :return: Expected number of hits, between 1 and 5. Equal to n_hits if n_hits is
            constant.
        :rtype: float
        """
        # 对于特定的移动ID，返回预期的击中次数
        if self._id == "triplekick" or self._id == "tripleaxel":
            # 对于Triple Kick和Triple Axel，每次击中都有准确性检查，并且每次击中的威力逐渐增加
            return 1 + 2 * 0.9 + 3 * 0.81
        min_hits, max_hits = self.n_hit
        if min_hits == max_hits:
            return min_hits
        else:
            # 如果击中次数不固定，则返回一个范围内的预期值
            assert (
                min_hits == 2 and max_hits == 5
            ), f"Move {self._id} expected to hit 2-5 times. Got {min_hits}-{max_hits}"
            return (2 + 3) / 3 + (4 + 5) / 6

    @property
    def flags(self) -> Set[str]:
        """
        This property is not well defined, and may be missing some information.
        If you need more information on some flag, please open an issue in the project.

        :return: Flags associated with this move. These can come from the data or be
            custom.
        :rtype: Set[str]
        """
        # 获取与该移动相关的标志，可能来自数据或自定义
        flags = set(self.entry["flags"])
        flags.update(set(self.entry.keys()).intersection(self._MISC_FLAGS))
        return flags

    @property
    def force_switch(self) -> bool:
        """
        :return: Whether this move forces switches.
        :rtype: bool
        """
        # 返回该移动是否强制交换精灵的布尔值
        return self.entry.get("forceSwitch", False)

    @property
    def heal(self) -> float:
        """
        :return: Proportion of the user's HP recovered.
        :rtype: float
        """
        # 如果"heal"在self.entry中
        if "heal" in self.entry:
            # 返回用户恢复的HP比例
            return self.entry["heal"][0] / self.entry["heal"][1]
        return 0.0

    @property
    def id(self) -> str:
        """
        :return: Move id.
        :rtype: str
        """
        # 返回移动的id
        return self._id

    @property
    def ignore_ability(self) -> bool:
        """
        :return: Whether the move ignore its target's ability.
        :rtype: bool
        """
        # 返回该移动是否忽略目标精灵的能力
        return self.entry.get("ignoreAbility", False)

    @property
    def ignore_defensive(self) -> bool:
        """
        :return: Whether the opponent's stat boosts are ignored.
        :rtype: bool
        """
        # 返回对手的属性提升是否被忽略
        return self.entry.get("ignoreDefensive", False)

    @property
    def ignore_evasion(self) -> bool:
        """
        :return: Wheter the opponent's evasion is ignored.
        :rtype: bool
        """
        # 返回对手的闪避是否被忽略
        return self.entry.get("ignoreEvasion", False)

    @property
    def ignore_immunity(self) -> Union[bool, Set[PokemonType]]:
        """
        :return: Whether the opponent's immunity is ignored, or a list of ignored
            immunities.
        :rtype: bool or set of Types
        """
        # 如果"ignoreImmunity"在self.entry中
        if "ignoreImmunity" in self.entry:
            # 如果self.entry["ignoreImmunity"]是布尔值
            if isinstance(self.entry["ignoreImmunity"], bool):
                return self.entry["ignoreImmunity"]
            else:
                # 返回被忽略的免疫类型的集合
                return {
                    PokemonType[t.upper().replace("'", "")]
                    for t in self.entry["ignoreImmunity"].keys()
                }
        return False

    @property
    def is_empty(self) -> bool:
        """
        :return: Whether the move is an empty move.
        :rtype: bool
        """
        # 返回移动是否为空的布尔值
        return self._is_empty

    @property
    def is_protect_counter(self) -> bool:
        """
        :return: Wheter this move increments a mon's protect counter.
        :rtype: int
        """
        # 返回移动是否增加宝可梦的保护计数器的布尔值
        return self._id in _PROTECT_COUNTER_MOVES

    @property
    def is_protect_move(self) -> bool:
        """
        :return: Wheter this move is a protect-like move.
        :rtype: int
        """
        # 返回移动是否类似于保护的布尔值
        return self._id in _PROTECT_MOVES

    @property
    def is_side_protect_move(self) -> bool:
        """
        :return: Wheter this move is a side-protect move.
        :rtype: int
        """
        # 返回移动是否是侧面保护的布尔值
        return self._id in _SIDE_PROTECT_MOVES

    @property
    def is_z(self) -> bool:
        """
        :return: Whether the move is a z move.
        :rtype: bool
        """
        # 返回移动是否是 Z 移动的布尔值
        return Move.is_id_z(self.id, gen=self._gen)

    @property
    def max_pp(self) -> int:
        """
        :return: The move's max pp.
        :rtype: int
        """
        # 返回移动的最大 PP 值
        return self.entry["pp"] * 8 // 5

    @property
    def n_hit(self) -> Tuple[int, int]:
        """
        :return: How many hits this move lands. Tuple of the form (min, max).
        :rtype: Tuple
        """
        # 返回移动命中的次数。返回形式为 (最小值, 最大值) 的元组
        if "multihit" in self.entry:
            if isinstance(self.entry["multihit"], list):
                assert len(self.entry["multihit"]) == 2
                min_hits, max_hits = self.entry["multihit"]
                return min_hits, max_hits
            else:
                return self.entry["multihit"], self.entry["multihit"]
        return 1, 1

    @property
    def no_pp_boosts(self) -> bool:
        """
        :return: Whether the move uses PPs.
        :rtype: bool
        """
        # 返回移动是否使用 PPs 的布尔值
        return "noPPBoosts" in self.entry
    def non_ghost_target(self) -> bool:
        """
        :return: True if the move targets non-ghost Pokemon.
        :rtype: bool
        """
        # Check if the key "nonGhostTarget" exists in the entry dictionary
        return "nonGhostTarget" in self.entry

    @property
    def priority(self) -> int:
        """
        :return: Priority of the move.
        :rtype: int
        """
        # Return the value associated with the key "priority" in the entry dictionary
        return self.entry["priority"]

    @property
    def pseudo_weather(self) -> str:
        """
        :return: Pseudo-weather triggered by this move.
        :rtype: str
        """
        # Return the value associated with the key "pseudoWeather" in the entry dictionary, or None if not found
        return self.entry.get("pseudoWeather", None)

    @property
    def recoil(self) -> float:
        """
        :return: Percentage of damage inflicted by the move as recoil.
        :rtype: float
        """
        # Calculate and return the recoil proportion based on the values in the entry dictionary
        if "recoil" in self.entry:
            return self.entry["recoil"][0] / self.entry["recoil"][1]
        elif "struggleRecoil" in self.entry:
            return 0.25
        return 0.0

    @property
    def request_target(self) -> Optional[str]:
        """
        :return: Target information provided by Showdown in a request message, if any.
        :rtype: str, optional
        """
        # Return the stored request target information
        return self._request_target

    @request_target.setter
    def request_target(self, request_target: Optional[str]):
        """
        :param request_target: Target information received from Showdown in a request message.
        :type request_target: str, optional
        """
        # Set the request target information received from Showdown
        self._request_target = request_target

    @staticmethod
    @lru_cache(maxsize=4096)
    def retrieve_id(move_name: str) -> str:
        """Retrieve the id of a move based on its full name.

        :param move_name: The string to convert into a move id.
        :type move_name: str
        :return: The corresponding move id.
        :rtype: str
        """
        # 将移动名称转换为标识符字符串
        move_name = to_id_str(move_name)
        # 如果移动名称以"return"开头，则返回"return"
        if move_name.startswith("return"):
            return "return"
        # 如果移动名称以"frustration"开头，则返回"frustration"
        if move_name.startswith("frustration"):
            return "frustration"
        # 如果移动名称以"hiddenpower"开头，则返回"hiddenpower"
        if move_name.startswith("hiddenpower"):
            return "hiddenpower"
        # 否则返回移动名称
        return move_name

    @property
    def secondary(self) -> List[Dict[str, Any]]:
        """
        :return: Secondary effects. At this point, the precise content of this property
            is not too clear.
        :rtype: Optional[Dict]
        """
        # 如果属性中包含"secondary"且不为空，则返回包含"secondary"的列表
        if "secondary" in self.entry and self.entry["secondary"]:
            return [self.entry["secondary"]]
        # 如果属性中包含"secondaries"，则返回"secondaries"
        elif "secondaries" in self.entry:
            return self.entry["secondaries"]
        # 否则返回空列表
        return []

    @property
    def self_boost(self) -> Optional[Dict[str, int]]:
        """
        :return: Boosts applied to the move's user.
        :rtype: Dict[str, int]
        """
        # 如果属性中包含"selfBoost"，则返回"selfBoost"中的"boosts"，否则返回None
        if "selfBoost" in self.entry:
            return self.entry["selfBoost"].get("boosts", None)
        # 如果属性中包含"self"且"boosts"在"self"中，则返回"self"中的"boosts"
        elif "self" in self.entry and "boosts" in self.entry["self"]:
            return self.entry["self"]["boosts"]
        # 否则返回None
        return None

    @property
    def self_destruct(self) -> Optional[str]:
        """
        :return: Move's self destruct consequences.
        :rtype: str | None
        """
        # 返回属性中的"selfdestruct"，如果不存在则返回None
        return self.entry.get("selfdestruct", None)

    @property
    def self_switch(self) -> Union[str, bool]:
        """
        :return: What kind of self swtich this move implies for the user.
        :rtype: str | None
        """
        # 返回属性中的"selfSwitch"，如果不存在则返回False
        return self.entry.get("selfSwitch", False)

    @property
    def side_condition(self) -> Optional[str]:
        """
        :return: Side condition inflicted by the move.
        :rtype: str | None
        """
        # 返回移动造成的副作用条件
        return self.entry.get("sideCondition", None)

    @property
    def sleep_usable(self) -> bool:
        """
        :return: Whether the move can be user by a sleeping pokemon.
        :rtype: bool
        """
        # 返回移动是否可以被睡眠的宝可梦使用
        return self.entry.get("sleepUsable", False)

    @property
    def slot_condition(self) -> Optional[str]:
        """
        :return: Which slot condition is started by this move.
        :rtype: str | None
        """
        # 返回由此招式引发的槽位条件
        return self.entry.get("slotCondition", None)

    @property
    def stalling_move(self) -> bool:
        """
        :return: Showdown classification of the move as a stalling move.
        :rtype: bool
        """
        # 返回招式作为拖延招式的Showdown分类
        return self.entry.get("stallingMove", False)

    @property
    def status(self) -> Optional[Status]:
        """
        :return: The status inflicted by the move.
        :rtype: Optional[Status]
        """
        # 如果字典中包含"status"键，则返回对应的Status枚举值，否则返回None
        if "status" in self.entry:
            return Status[self.entry["status"].upper()]
        return None

    @property
    def steals_boosts(self) -> bool:
        """
        :return: Whether the move steals its target's boosts.
        :rtype: bool
        """
        # 返回招式是否偷取目标的增益效果
        return self.entry.get("stealsBoosts", False)
    # 返回移动的目标类型
    def target(self) -> str:
        """
        :return: Move target. Possible targets (copied from PS codebase):

            * adjacentAlly - Only relevant to Doubles or Triples, the move only
              targets an ally of the user.
            * adjacentAllyOrSelf - The move can target the user or its ally.
            * adjacentFoe - The move can target a foe, but not (in Triples)
              a distant foe.
            * all - The move targets the field or all Pokémon at once.
            * allAdjacent - The move is a spread move that also hits the user's ally.
            * allAdjacentFoes - The move is a spread move.
            * allies - The move affects all active Pokémon on the user's team.
            * allySide - The move adds a side condition on the user's side.
            * allyTeam - The move affects all unfainted Pokémon on the user's team.
            * any - The move can hit any other active Pokémon, not just those adjacent.
            * foeSide - The move adds a side condition on the foe's side.
            * normal - The move can hit one adjacent Pokémon of your choice.
            * randomNormal - The move targets an adjacent foe at random.
            * scripted - The move targets the foe that damaged the user.
            * self - The move affects the user of the move.
        :rtype: str
        """
        return self.entry["target"]

    # 返回由移动引发的地形
    @property
    def terrain(self) -> Optional[Field]:
        """
        :return: Terrain started by the move.
        :rtype: Optional[Field]
        """
        # 获取移动引发的地形，如果存在则转换成 Field 对象
        terrain = self.entry.get("terrain", None)
        if terrain is not None:
            terrain = Field.from_showdown_message(terrain)
        return terrain

    # 返回移动是否解冻目标
    @property
    def thaws_target(self) -> bool:
        """
        :return: Whether the move thaws its target.
        :rtype: bool
        """
        return self.entry.get("thawsTarget", False)

    @property
    def type(self) -> PokemonType:
        """
        :return: Move type.
        :rtype: PokemonType
        """
        # 返回移动的类型，通过从名称获取 PokemonType 对象
        return PokemonType.from_name(self.entry["type"])

    @property
    def use_target_offensive(self) -> bool:
        """
        :return: Whether the move uses the target's offensive statistics.
        :rtype: bool
        """
        # 返回移动是否使用目标的进攻统计信息
        return self.entry.get("overrideOffensivePokemon", False) == "target"

    @property
    def volatile_status(self) -> Optional[str]:
        """
        :return: Volatile status inflicted by the move.
        :rtype: str | None
        """
        # 返回移动造成的短暂状态
        return self.entry.get("volatileStatus", None)

    @property
    def weather(self) -> Optional[Weather]:
        """
        :return: Weather started by the move.
        :rtype: Optional[Weather]
        """
        # 返回由移动引起的天气
        if "weather" in self.entry:
            return Weather[self.entry["weather"].upper()]
        return None

    @property
    def z_move_boost(self) -> Optional[Dict[str, int]]:
        """
        :return: Boosts associated with the z-move version of this move.
        :rtype: Dict[str, int]
        """
        # 返回与 z-move 版本的移动相关联的增益
        if "zMove" in self.entry and "boost" in self.entry["zMove"]:
            return self.entry["zMove"]["boost"]
        return None

    @property
    def z_move_effect(self) -> Optional[str]:
        """
        :return: Effects associated with the z-move version of this move.
        :rtype: str | None
        """
        # 返回与 z-move 版本的移动相关联的效果
        if "zMove" in self.entry and "effect" in self.entry["zMove"]:
            return self.entry["zMove"]["effect"]
        return None

    @property
    # 返回该招式的 Z 招式版本的基础威力
    def z_move_power(self) -> int:
        """
        :return: Base power of the z-move version of this move.
        :rtype: int
        """
        # 如果该招式有 Z 招式属性，并且包含基础威力信息，则返回基础威力
        if "zMove" in self.entry and "basePower" in self.entry["zMove"]:
            return self.entry["zMove"]["basePower"]
        # 如果该招式为状态招式，则基础威力为 0
        elif self.category == MoveCategory.STATUS:
            return 0
        # 获取基础威力值
        base_power = self.base_power
        # 如果招式命中次数不是 (1, 1)，则基础威力乘以 3
        if self.n_hit != (1, 1):
            base_power *= 3
        # 根据基础威力值返回对应的 Z 招式威力值
        elif base_power <= 55:
            return 100
        elif base_power <= 65:
            return 120
        elif base_power <= 75:
            return 140
        elif base_power <= 85:
            return 160
        elif base_power <= 95:
            return 175
        elif base_power <= 100:
            return 180
        elif base_power <= 110:
            return 185
        elif base_power <= 125:
            return 190
        elif base_power <= 130:
            return 195
        # 基础威力大于 130，则返回 200
        return 200
class EmptyMove(Move):
    # 定义一个空招式类，继承自Move类
    def __init__(self, move_id: str):
        # 初始化方法，接受一个招式ID参数
        self._id = move_id
        self._is_empty: bool = True
        # 设置_is_empty属性为True

    def __getattribute__(self, name: str):
        # 重写__getattribute__方法
        try:
            return super(Move, self).__getattribute__(name)
        except (AttributeError, TypeError, ValueError):
            return 0
        # 尝试获取属性，如果出现异常则返回0

    def __deepcopy__(self, memodict: Optional[Dict[int, Any]] = {}):
        # 重写__deepcopy__方法，实现深拷贝
        return EmptyMove(copy.deepcopy(self._id, memodict))
        # 返回一个深拷贝的EmptyMove对象


class DynamaxMove(Move):
    # 定义一个DynamaxMove类，继承自Move类
    BOOSTS_MAP = {
        # 定义BOOSTS_MAP字典，存储属性提升信息
        PokemonType.BUG: {"spa": -1},
        PokemonType.DARK: {"spd": -1},
        PokemonType.DRAGON: {"atk": -1},
        PokemonType.GHOST: {"def": -1},
        PokemonType.NORMAL: {"spe": -1},
    }
    SELF_BOOSTS_MAP = {
        # 定义SELF_BOOSTS_MAP字典，存储自身属性提升信息
        PokemonType.FIGHTING: {"atk": +1},
        PokemonType.FLYING: {"spe": +1},
        PokemonType.GROUND: {"spd": +1},
        PokemonType.POISON: {"spa": +1},
        PokemonType.STEEL: {"def": +1},
    }
    TERRAIN_MAP = {
        # 定义TERRAIN_MAP字典，存储地形信息
        PokemonType.ELECTRIC: Field.ELECTRIC_TERRAIN,
        PokemonType.FAIRY: Field.MISTY_TERRAIN,
        PokemonType.GRASS: Field.GRASSY_TERRAIN,
        PokemonType.PSYCHIC: Field.PSYCHIC_TERRAIN,
    }
    WEATHER_MAP = {
        # 定义WEATHER_MAP字典，存储天气信息
        PokemonType.FIRE: Weather.SUNNYDAY,
        PokemonType.ICE: Weather.HAIL,
        PokemonType.ROCK: Weather.SANDSTORM,
        PokemonType.WATER: Weather.RAINDANCE,
    }

    def __init__(self, parent: Move):
        # 初始化方法，接受一个Move对象作为参数
        self._parent: Move = parent

    def __getattr__(self, name: str):
        # 重写__getattr__方法
        if name[:2] == "__":
            raise AttributeError(name)
        return getattr(self._parent, name)
        # 获取父类Move对象的属性

    @property
    def accuracy(self):
        # 定义accuracy属性
        return 1
        # 返回值为1

    @property
    # 定义一个属性
    # 返回技能的基础威力，如果技能类型不是状态技能
    def base_power(self) -> int:
        # 如果技能类型不是状态技能，则获取父类的基础威力
        base_power = self._parent.base_power
        # 如果技能属性是毒或格斗
        if self.type in {PokemonType.POISON, PokemonType.FIGHTING}:
            # 根据基础威力的不同范围返回不同的威力值
            if base_power < 40:
                return 70
            if base_power < 50:
                return 75
            if base_power < 60:
                return 80
            if base_power < 70:
                return 85
            if base_power < 100:
                return 90
            if base_power < 140:
                return 95
            return 100
        else:
            # 根据基础威力的不同范围返回不同的威力值
            if base_power < 40:
                return 90
            if base_power < 50:
                return 100
            if base_power < 60:
                return 110
            if base_power < 70:
                return 120
            if base_power < 100:
                return 130
            if base_power < 140:
                return 140
            return 150
        # 如果技能类型是状态技能，则返回0
        return 0

    # 返回技能的增益效果，如果技能类型不是状态技能
    @property
    def boosts(self) -> Optional[Dict[str, int]]:
        if self.category != MoveCategory.STATUS:
            return self.BOOSTS_MAP.get(self.type, None)
        return None

    # 返回技能是否能破坏保护
    @property
    def breaks_protect(self):
        return False

    # 返回技能的暴击率
    @property
    def crit_ratio(self):
        return 0

    # 返回技能造成的伤害值
    @property
    def damage(self):
        return 0

    # 返回技能的防御类别
    @property
    def defensive_category(self):
        return self.category

    # 返回技能预期的命中次数
    @property
    def expected_hits(self):
        return 1

    # 返回是否强制交换精灵
    @property
    def force_switch(self):
        return False

    # 返回技能的治疗效果
    @property
    def heal(self):
        return 0

    # 返回技能是否是保护技能的反击
    @property
    def is_protect_counter(self):
        return self.category == MoveCategory.STATUS

    # 返回技能是否是保护技能
    @property
    def is_protect_move(self):
        return self.category == MoveCategory.STATUS

    # 返回技能的命中次数范围
    @property
    def n_hit(self):
        return 1, 1
    # 返回技能的优先级，这里默认为0
    def priority(self):
        return 0

    # 返回技能的反冲值，这里默认为0
    @property
    def recoil(self):
        return 0

    # 返回技能对自身的增益效果，如果技能不是状态技能，则返回对应类型的增益效果字典，否则返回None
    @property
    def self_boost(self) -> Optional[Dict[str, int]]:
        if self.category != MoveCategory.STATUS:
            return self.SELF_BOOSTS_MAP.get(self.type, None)
        return None

    # 返回技能的状态效果，这里默认为None
    @property
    def status(self):
        return None

    # 返回技能对场地的影响，如果技能不是状态技能，则返回对应类型的场地效果，否则返回None
    @property
    def terrain(self) -> Optional[Field]:
        if self.category != MoveCategory.STATUS:
            return self.TERRAIN_MAP.get(self.type, None)
        return None

    # 返回技能对天气的影响，如果技能不是状态技能，则返回对应类型的天气效果，否则返回None
    @property
    def weather(self) -> Optional[Weather]:
        if self.category != MoveCategory.STATUS:
            return self.WEATHER_MAP.get(self.type, None)
        return None
```