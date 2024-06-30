# `D:\src\scipysrc\sympy\sympy\physics\biomechanics\musculotendon.py`

```
"""
Implementations of musculotendon models.

Musculotendon models are a critical component of biomechanical models, one that
differentiates them from pure multibody systems. Musculotendon models produce a
force dependent on their level of activation, their length, and their
extension velocity. Length- and extension velocity-dependent force production
are governed by force-length and force-velocity characteristics.
These are normalized functions that are dependent on the musculotendon's state
and are specific to a given musculotendon model.
"""

# 导入必要的库
from abc import abstractmethod
from enum import IntEnum, unique

# 导入 SymPy 相关模块
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.dense import MutableDenseMatrix as Matrix, diag, eye, zeros
from sympy.physics.biomechanics.activation import ActivationBase
from sympy.physics.biomechanics.curve import (
    CharacteristicCurveCollection,
    FiberForceLengthActiveDeGroote2016,
    FiberForceLengthPassiveDeGroote2016,
    FiberForceLengthPassiveInverseDeGroote2016,
    FiberForceVelocityDeGroote2016,
    FiberForceVelocityInverseDeGroote2016,
    TendonForceLengthDeGroote2016,
    TendonForceLengthInverseDeGroote2016,
)
from sympy.physics.biomechanics._mixin import _NamedMixin
from sympy.physics.mechanics.actuator import ForceActuator
from sympy.physics.vector.functions import dynamicsymbols

# 定义 __all__ 列表，指定导出的模块名
__all__ = [
    'MusculotendonBase',
    'MusculotendonDeGroote2016',
    'MusculotendonFormulation',
]

@unique
class MusculotendonFormulation(IntEnum):
    """
    Enumeration of types of musculotendon dynamics formulations.

    Explanation
    ===========

    An (integer) enumeration is used as it allows for clearer selection of the
    different formulations of musculotendon dynamics.

    Members
    =======

    RIGID_TENDON : 0
        A rigid tendon model.
    FIBER_LENGTH_EXPLICIT : 1
        An explicit elastic tendon model with the muscle fiber length (l_M) as
        the state variable.
    TENDON_FORCE_EXPLICIT : 2
        An explicit elastic tendon model with the tendon force (F_T) as the
        state variable.
    FIBER_LENGTH_IMPLICIT : 3
        An implicit elastic tendon model with the muscle fiber length (l_M) as
        the state variable and the muscle fiber velocity as an additional input
        variable.
    TENDON_FORCE_IMPLICIT : 4
        An implicit elastic tendon model with the tendon force (F_T) as the
        state variable as the muscle fiber velocity as an additional input
        variable.
    """

    RIGID_TENDON = 0
    FIBER_LENGTH_EXPLICIT = 1
    TENDON_FORCE_EXPLICIT = 2
    FIBER_LENGTH_IMPLICIT = 3
    TENDON_FORCE_IMPLICIT = 4
    def __str__(self):
        """
        Returns a string representation of the enumeration value.

        Notes
        =====

        This method overrides the default behavior to ensure compatibility
        with different Python versions. In Python 3.10, there is a specific
        handling for `IntEnum` due to an issue between Python 3.10 and 3.11,
        which affects the behavior of `__str__`. This override ensures that
        the `__str__` method returns the string representation of the
        enumeration value correctly, regardless of Python version.

        In Python 3.11 and later versions, `IntEnum` uses `int.__str__` for
        `__str__`, while in earlier versions it used `Enum.__str__`. This
        method ensures consistent behavior until Python 3.11 becomes the
        minimum supported version in SymPy, at which point this override can
        be removed.

        Returns
        -------
        str
            String representation of the enumeration value.
        """
        return str(self.value)
# 默认的肌腱-肌腱形态配方为刚性肌腱模型
_DEFAULT_MUSCULOTENDON_FORMULATION = MusculotendonFormulation.RIGID_TENDON


class MusculotendonBase(ForceActuator, _NamedMixin):
    r"""Abstract base class for all musculotendon classes to inherit from.

    Explanation
    ===========

    A musculotendon generates a contractile force based on its activation,
    length, and shortening velocity. This abstract base class is to be inherited
    by all musculotendon subclasses that implement different characteristic
    musculotendon curves. Characteristic musculotendon curves are required for
    the tendon force-length, passive fiber force-length, active fiber force-
    length, and fiber force-velocity relationships.

    Parameters
    ==========

    name : str
        The name identifier associated with the musculotendon. This name is used
        as a suffix when automatically generated symbols are instantiated. It
        must be a string of nonzero length.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    activation_dynamics : ActivationBase
        The activation dynamics that will be modeled within the musculotendon.
        This must be an instance of a concrete subclass of ``ActivationBase``,
        e.g. ``FirstOrderActivationDeGroote2016``.
    musculotendon_dynamics : MusculotendonFormulation | int
        The formulation of musculotendon dynamics that should be used
        internally, i.e. rigid or elastic tendon model, the choice of
        musculotendon state etc. This must be a member of the integer
        enumeration ``MusculotendonFormulation`` or an integer that can be cast
        to a member. To use a rigid tendon formulation, set this to
        ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value ``0``,
        which will be cast to the enumeration member). There are four possible
        formulations for an elastic tendon model. To use an explicit formulation
        with the fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer value
        ``1``). To use an explicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``
        (or the integer value ``2``). To use an implicit formulation with the
        fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer value
        ``3``). To use an implicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``
        (or the integer value ``4``). The default is
        ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a rigid
        tendon formulation.
    tendon_slack_length : Expr | None
        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.
    peak_isometric_force : Expr | None
        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalize tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.
    optimal_fiber_length : Expr | None
        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.
    maximal_fiber_velocity : Expr | None
        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.
    optimal_pennation_angle : Expr | None
        The pennation angle when muscle fiber length equals the optimal fiber
        length.
    fiber_damping_coefficient : Expr | None
        The coefficient of damping to be used in the damping element in the
        muscle fiber model.
    with_defaults : bool
        Whether ``with_defaults`` alternate constructors should be used when
        automatically constructing child classes. Default is ``False``.
"""

def __init__(
    self,
    name,
    pathway,
    activation_dynamics,
    *,
    musculotendon_dynamics=_DEFAULT_MUSCULOTENDON_FORMULATION,
    tendon_slack_length=None,
    peak_isometric_force=None,
    optimal_fiber_length=None,
    maximal_fiber_velocity=None,
    optimal_pennation_angle=None,
    fiber_damping_coefficient=None,
    with_defaults=False,
):
    """
    Constructor method for initializing an instance of the class.
    
    Parameters:
    - name: Name of the instance.
    - pathway: Pathway information related to the instance.
    - activation_dynamics: Dynamics related to muscle activation.
    - musculotendon_dynamics: Dynamics formulation for musculotendon interactions.
    - tendon_slack_length: Length of the tendon in its unloaded state.
    - peak_isometric_force: Maximum force the muscle fiber can produce isometrically.
    - optimal_fiber_length: Length of the muscle fiber at maximal active force.
    - maximal_fiber_velocity: Maximum velocity of muscle fiber shortening.
    - optimal_pennation_angle: Pennation angle at optimal fiber length.
    - fiber_damping_coefficient: Coefficient of damping in the muscle fiber model.
    - with_defaults: Flag indicating if alternate constructors should be used.
    """
    pass

@classmethod
def with_defaults(
    cls,
    name,
    pathway,
    activation_dynamics,
    *,
    musculotendon_dynamics=_DEFAULT_MUSCULOTENDON_FORMULATION,
    tendon_slack_length=None,
    peak_isometric_force=None,
    optimal_fiber_length=None,
    maximal_fiber_velocity=Float('10.0'),
    optimal_pennation_angle=Float('0.0'),
    fiber_damping_coefficient=Float('0.1'),
):
    """
    Alternate constructor method that initializes an instance of the class with
    default values for certain parameters.

    Parameters:
    - name: Name of the instance.
    - pathway: Pathway information related to the instance.
    - activation_dynamics: Dynamics related to muscle activation.
    - musculotendon_dynamics: Dynamics formulation for musculotendon interactions.
    - tendon_slack_length: Length of the tendon in its unloaded state.
    - peak_isometric_force: Maximum force the muscle fiber can produce isometrically.
    - optimal_fiber_length: Length of the muscle fiber at maximal active force.
    - maximal_fiber_velocity: Maximum velocity of muscle fiber shortening (default: 10.0).
    - optimal_pennation_angle: Pennation angle at optimal fiber length (default: 0.0).
    - fiber_damping_coefficient: Coefficient of damping in the muscle fiber model (default: 0.1).
    """
    pass

@abstractmethod
def curves(cls):
    """
    Abstract method that should be implemented in subclasses to return a
    ``CharacteristicCurveCollection`` of the curves related to the specific model.
    """
    pass
    def tendon_slack_length(self):
        r"""Symbol or value corresponding to the tendon slack length constant.

        Explanation
        ===========

        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.

        The alias ``l_T_slack`` can also be used to access the same attribute.

        """
        # 返回属性 self._l_T_slack，表示肌腱松弛长度常量
        return self._l_T_slack

    @property
    def l_T_slack(self):
        r"""Symbol or value corresponding to the tendon slack length constant.

        Explanation
        ===========

        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.

        The alias ``tendon_slack_length`` can also be used to access the same
        attribute.

        """
        # 返回属性 self._l_T_slack，表示肌腱松弛长度常量（别名）
        return self._l_T_slack

    @property
    def peak_isometric_force(self):
        r"""Symbol or value corresponding to the peak isometric force constant.

        Explanation
        ===========

        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalize tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.

        The alias ``F_M_max`` can also be used to access the same attribute.

        """
        # 返回属性 self._F_M_max，表示最大等长收缩力常量
        return self._F_M_max

    @property
    def F_M_max(self):
        r"""Symbol or value corresponding to the peak isometric force constant.

        Explanation
        ===========

        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalize tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.

        The alias ``peak_isometric_force`` can also be used to access the same
        attribute.

        """
        # 返回属性 self._F_M_max，表示最大等长收缩力常量（别名）
        return self._F_M_max

    @property
    def optimal_fiber_length(self):
        r"""Symbol or value corresponding to the optimal fiber length constant.

        Explanation
        ===========

        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.

        The alias ``l_M_opt`` can also be used to access the same attribute.

        """
        # 返回对象内部保存的最佳肌纤维长度常量
        return self._l_M_opt

    @property
    def l_M_opt(self):
        r"""Symbol or value corresponding to the optimal fiber length constant.

        Explanation
        ===========

        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.

        The alias ``optimal_fiber_length`` can also be used to access the same
        attribute.

        """
        # 返回对象内部保存的最佳肌纤维长度常量
        return self._l_M_opt

    @property
    def maximal_fiber_velocity(self):
        r"""Symbol or value corresponding to the maximal fiber velocity constant.

        Explanation
        ===========

        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.

        The alias ``v_M_max`` can also be used to access the same attribute.

        """
        # 返回对象内部保存的最大肌纤维速度常量
        return self._v_M_max

    @property
    def v_M_max(self):
        r"""Symbol or value corresponding to the maximal fiber velocity constant.

        Explanation
        ===========

        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.

        The alias ``maximal_fiber_velocity`` can also be used to access the same
        attribute.

        """
        # 返回对象内部保存的最大肌纤维速度常量
        return self._v_M_max

    @property
    def optimal_pennation_angle(self):
        """Symbol or value corresponding to the optimal pennation angle
        constant.

        Explanation
        ===========

        The pennation angle when muscle fiber length equals the optimal fiber
        length.

        The alias ``alpha_opt`` can also be used to access the same attribute.

        """
        # 返回对象内部保存的最佳膜角常量
        return self._alpha_opt
    def alpha_opt(self):
        """
        返回最佳肌肉纤维长度对应的肌肉肌角的符号或值常量。

        解释
        ===========

        当肌肉纤维长度等于最佳纤维长度时的肌肉肌角。

        别名 ``optimal_pennation_angle`` 也可以用来访问同一属性。

        """
        return self._alpha_opt

    @property
    def fiber_damping_coefficient(self):
        """
        返回对应于纤维阻尼系数的符号或值常量。

        解释
        ===========

        在肌肉纤维模型中用于阻尼元素中的阻尼系数。

        别名 ``beta`` 也可以用来访问同一属性。

        """
        return self._beta

    @property
    def beta(self):
        """
        返回对应于纤维阻尼系数的符号或值常量。

        解释
        ===========

        在肌肉纤维模型中用于阻尼元素中的阻尼系数。

        别名 ``fiber_damping_coefficient`` 也可以用来访问同一属性。

        """
        return self._beta

    @property
    def activation_dynamics(self):
        """
        控制该肌腱的激活动力学模型。

        解释
        ===========

        返回一个 ``ActivationBase`` 子类的实例，该实例控制激励和激活之间的关系，
        用于表示该肌腱的激活动力学。

        """
        return self._activation_dynamics

    @property
    def excitation(self):
        """
        表示激励的动态符号。

        解释
        ===========

        别名 ``e`` 也可以用来访问同一属性。

        """
        return self._activation_dynamics._e

    @property
    def e(self):
        """
        表示激励的动态符号。

        解释
        ===========

        别名 ``excitation`` 也可以用来访问同一属性。

        """
        return self._activation_dynamics._e

    @property
    def activation(self):
        """
        表示激活的动态符号。

        解释
        ===========

        别名 ``a`` 也可以用来访问同一属性。

        """
        return self._activation_dynamics._a

    @property
    def a(self):
        """
        表示激活的动态符号。

        解释
        ===========

        别名 ``activation`` 也可以用来访问同一属性。

        """
        return self._activation_dynamics._a
    def musculotendon_dynamics(self):
        """
        返回当前用于肌腱动力学的模型类型或状态。

        Explanation
        ===========

        定义应该在内部使用的肌腱动力学模型，可以是刚性或弹性肌腱模型，以及肌腱状态等。此属性必须是整数枚举“MusculotendonFormulation”的成员之一，
        或者可以转换为其中一个成员的整数值。要使用刚性肌腱模型，请将其设置为“MusculotendonFormulation.RIGID_TENDON”（或整数值“0”，将转换为枚举成员）。
        弹性肌腱模型有四种可能的公式。要使用以肌纤维长度为状态的显式公式，请将其设置为“MusculotendonFormulation.FIBER_LENGTH_EXPLICIT”（或整数值“1”）。
        要使用以肌腱力为状态的显式公式，请将其设置为“MusculotendonFormulation.TENDON_FORCE_EXPLICIT”（或整数值“2”）。
        要使用以肌纤维长度为状态的隐式公式，请将其设置为“MusculotendonFormulation.FIBER_LENGTH_IMPLICIT”（或整数值“3”）。
        要使用以肌腱力为状态的隐式公式，请将其设置为“MusculotendonFormulation.TENDON_FORCE_IMPLICIT”（或整数值“4”）。
        默认值为“MusculotendonFormulation.RIGID_TENDON”，对应于刚性肌腱模型。

        """
        return self._musculotendon_dynamics
    def _rigid_tendon_musculotendon_dynamics(self):
        """Defines the dynamics of a rigid tendon musculotendon."""
        # Extract lengths and velocities from pathway object
        self._l_MT = self.pathway.length
        self._v_MT = self.pathway.extension_velocity
        # Initialize tendon lengths
        self._l_T = self._l_T_slack
        self._l_T_tilde = Integer(1)
        # Calculate musculotendon length and normalize
        self._l_M = sqrt((self._l_MT - self._l_T)**2 + (self._l_M_opt*sin(self._alpha_opt))**2)
        self._l_M_tilde = self._l_M/self._l_M_opt
        # Calculate musculotendon velocity and normalize
        self._v_M = self._v_MT*(self._l_MT - self._l_T_slack)/self._l_M
        self._v_M_tilde = self._v_M/self._v_M_max
        # Set force-length relationships based on whether using defaults
        if self._with_defaults:
            self._fl_T = self.curves.tendon_force_length.with_defaults(self._l_T_tilde)
            self._fl_M_pas = self.curves.fiber_force_length_passive.with_defaults(self._l_M_tilde)
            self._fl_M_act = self.curves.fiber_force_length_active.with_defaults(self._l_M_tilde)
            self._fv_M = self.curves.fiber_force_velocity.with_defaults(self._v_M_tilde)
        else:
            # Define constants for non-default force-length relationships
            fl_T_constants = symbols(f'c_0:4_fl_T_{self.name}')
            self._fl_T = self.curves.tendon_force_length(self._l_T_tilde, *fl_T_constants)
            fl_M_pas_constants = symbols(f'c_0:2_fl_M_pas_{self.name}')
            self._fl_M_pas = self.curves.fiber_force_length_passive(self._l_M_tilde, *fl_M_pas_constants)
            fl_M_act_constants = symbols(f'c_0:12_fl_M_act_{self.name}')
            self._fl_M_act = self.curves.fiber_force_length_active(self._l_M_tilde, *fl_M_act_constants)
            fv_M_constants = symbols(f'c_0:4_fv_M_{self.name}')
            self._fv_M = self.curves.fiber_force_velocity(self._v_M_tilde, *fv_M_constants)
        # Calculate scaled musculotendon force
        self._F_M_tilde = self.a*self._fl_M_act*self._fv_M + self._fl_M_pas + self._beta*self._v_M_tilde
        self._F_T_tilde = self._F_M_tilde
        # Calculate actual musculotendon force
        self._F_M = self._F_M_tilde*self._F_M_max
        # Calculate cosine of optimal angle
        self._cos_alpha = cos(self._alpha_opt)
        # Calculate tendon force
        self._F_T = self._F_M*self._cos_alpha

        # Containers for state variables, input variables, state equations, and curve constants
        self._state_vars = zeros(0, 1)
        self._input_vars = zeros(0, 1)
        self._state_eqns = zeros(0, 1)
        # Assign curve constants based on whether using defaults
        self._curve_constants = Matrix(
            fl_T_constants
            + fl_M_pas_constants
            + fl_M_act_constants
            + fv_M_constants
        ) if not self._with_defaults else zeros(0, 1)
    def _fiber_length_explicit_musculotendon_dynamics(self):
        """Calculate explicit dynamics of musculotendon with elastic tendon."""
        # Define a dynamic symbol for muscle tendon length scaled by optimal length
        self._l_M_tilde = dynamicsymbols(f'l_M_tilde_{self.name}')
        # Retrieve current musculotendon length and extension velocity from pathway object
        self._l_MT = self.pathway.length
        self._v_MT = self.pathway.extension_velocity
        # Calculate muscle length based on scaled length
        self._l_M = self._l_M_tilde * self._l_M_opt
        # Calculate tendon length using musculotendon and optimal angle
        self._l_T = self._l_MT - sqrt(self._l_M**2 - (self._l_M_opt*sin(self._alpha_opt))**2)
        # Scale tendon length by slack length
        self._l_T_tilde = self._l_T / self._l_T_slack
        # Calculate cosine of optimal angle
        self._cos_alpha = (self._l_MT - self._l_T) / self._l_M
        # Determine tendon force-length relation using defaults or specified constants
        if self._with_defaults:
            self._fl_T = self.curves.tendon_force_length.with_defaults(self._l_T_tilde)
            self._fl_M_pas = self.curves.fiber_force_length_passive.with_defaults(self._l_M_tilde)
            self._fl_M_act = self.curves.fiber_force_length_active.with_defaults(self._l_M_tilde)
        else:
            fl_T_constants = symbols(f'c_0:4_fl_T_{self.name}')
            self._fl_T = self.curves.tendon_force_length(self._l_T_tilde, *fl_T_constants)
            fl_M_pas_constants = symbols(f'c_0:2_fl_M_pas_{self.name}')
            self._fl_M_pas = self.curves.fiber_force_length_passive(self._l_M_tilde, *fl_M_pas_constants)
            fl_M_act_constants = symbols(f'c_0:12_fl_M_act_{self.name}')
            self._fl_M_act = self.curves.fiber_force_length_active(self._l_M_tilde, *fl_M_act_constants)
        # Calculate tendon force and muscle force
        self._F_T_tilde = self._fl_T
        self._F_T = self._F_T_tilde * self._F_M_max
        self._F_M = self._F_T / self._cos_alpha
        self._F_M_tilde = self._F_M / self._F_M_max
        # Calculate normalized fiber velocity
        self._fv_M = (self._F_M_tilde - self._fl_M_pas) / (self.a * self._fl_M_act)
        # Determine inverse of fiber force-velocity relation using defaults or specified constants
        if self._with_defaults:
            self._v_M_tilde = self.curves.fiber_force_velocity_inverse.with_defaults(self._fv_M)
        else:
            fv_M_constants = symbols(f'c_0:4_fv_M_{self.name}')
            self._v_M_tilde = self.curves.fiber_force_velocity_inverse(self._fv_M, *fv_M_constants)
        # Calculate time derivative of scaled muscle tendon length
        self._dl_M_tilde_dt = (self._v_M_max / self._l_M_opt) * self._v_M_tilde

        # Define state variables, input variables, state equations, and curve constants
        self._state_vars = Matrix([self._l_M_tilde])
        self._input_vars = zeros(0, 1)
        self._state_eqns = Matrix([self._dl_M_tilde_dt])
        self._curve_constants = Matrix(
            fl_T_constants
            + fl_M_pas_constants
            + fl_M_act_constants
            + fv_M_constants
        ) if not self._with_defaults else zeros(0, 1)
    # 定义一个方法，用于计算显式肌腱肌腱动力学的弹性肌腱，使用 `F_T_tilde` 作为状态变量
    def _tendon_force_explicit_musculotendon_dynamics(self):
        """Elastic tendon musculotendon using `F_T_tilde` as a state."""
        # 定义 `_F_T_tilde` 为动力符号（符号函数的特殊类型），表示弹性肌腱的力
        self._F_T_tilde = dynamicsymbols(f'F_T_tilde_{self.name}')
        # 设置 `_l_MT` 为路径的长度
        self._l_MT = self.pathway.length
        # 设置 `_v_MT` 为路径的伸展速度
        self._v_MT = self.pathway.extension_velocity
        # 将 `_fl_T` 初始化为 `_F_T_tilde`
        self._fl_T = self._F_T_tilde
        # 如果设置了 `_with_defaults` 标志，使用默认参数创建 `_fl_T_inv`
        if self._with_defaults:
            self._fl_T_inv = self.curves.tendon_force_length_inverse.with_defaults(self._fl_T)
        else:
            # 否则，创建包含常数的符号并使用它们创建 `_fl_T_inv`
            fl_T_constants = symbols(f'c_0:4_fl_T_{self.name}')
            self._fl_T_inv = self.curves.tendon_force_length_inverse(self._fl_T, *fl_T_constants)
        # 设置 `_l_T_tilde` 为 `_fl_T_inv`
        self._l_T_tilde = self._fl_T_inv
        # 计算 `_l_T`，表示肌腱长度乘以松弛长度比例
        self._l_T = self._l_T_tilde*self._l_T_slack
        # 计算 `_l_M`，表示肌肉长度
        self._l_M = sqrt((self._l_MT - self._l_T)**2 + (self._l_M_opt*sin(self._alpha_opt))**2)
        # 计算 `_l_M_tilde`，表示肌肉长度与最佳肌肉长度的比值
        self._l_M_tilde = self._l_M/self._l_M_opt
        # 如果设置了 `_with_defaults` 标志，使用默认参数创建 `_fl_M_pas` 和 `_fl_M_act`
        if self._with_defaults:
            self._fl_M_pas = self.curves.fiber_force_length_passive.with_defaults(self._l_M_tilde)
            self._fl_M_act = self.curves.fiber_force_length_active.with_defaults(self._l_M_tilde)
        else:
            # 否则，创建包含常数的符号并使用它们创建 `_fl_M_pas` 和 `_fl_M_act`
            fl_M_pas_constants = symbols(f'c_0:2_fl_M_pas_{self.name}')
            self._fl_M_pas = self.curves.fiber_force_length_passive(self._l_M_tilde, *fl_M_pas_constants)
            fl_M_act_constants = symbols(f'c_0:12_fl_M_act_{self.name}')
            self._fl_M_act = self.curves.fiber_force_length_active(self._l_M_tilde, *fl_M_act_constants)
        # 计算 `_cos_alpha`，表示肌肉与肌腱长度之比的余弦值
        self._cos_alpha = (self._l_MT - self._l_T)/self._l_M
        # 计算 `_F_T`，表示肌腱的力
        self._F_T = self._F_T_tilde*self._F_M_max
        # 计算 `_F_M`，表示肌肉的力
        self._F_M = self._F_T/self._cos_alpha
        # 计算 `_F_M_tilde`，表示肌肉力与最大肌肉力之比
        self._F_M_tilde = self._F_M/self._F_M_max
        # 计算 `_fv_M`，表示肌肉力与肌肉速度之比
        self._fv_M = (self._F_M_tilde - self._fl_M_pas)/(self.a*self._fl_M_act)
        # 如果设置了 `_with_defaults` 标志，使用默认参数创建 `_fv_M_inv`
        if self._with_defaults:
            self._fv_M_inv = self.curves.fiber_force_velocity_inverse.with_defaults(self._fv_M)
        else:
            # 否则，创建包含常数的符号并使用它们创建 `_fv_M_inv`
            fv_M_constants = symbols(f'c_0:4_fv_M_{self.name}')
            self._fv_M_inv = self.curves.fiber_force_velocity_inverse(self._fv_M, *fv_M_constants)
        # 计算 `_v_M_tilde`，表示肌肉速度与肌肉最大速度之比
        self._v_M_tilde = self._fv_M_inv
        # 计算 `_v_M`，表示肌肉速度
        self._v_M = self._v_M_tilde*self._v_M_max
        # 计算 `_v_T`，表示肌腱速度
        self._v_T = self._v_MT - (self._v_M/self._cos_alpha)
        # 计算 `_v_T_tilde`，表示肌腱速度与松弛速度之比
        self._v_T_tilde = self._v_T/self._l_T_slack
        # 如果设置了 `_with_defaults` 标志，使用默认参数创建 `_fl_T`
        if self._with_defaults:
            self._fl_T = self.curves.tendon_force_length.with_defaults(self._l_T_tilde)
        else:
            # 否则，使用之前定义的常数创建 `_fl_T`
            self._fl_T = self.curves.tendon_force_length(self._l_T_tilde, *fl_T_constants)
        # 计算 `_dF_T_tilde_dt`，表示肌腱力随时间的变化率
        self._dF_T_tilde_dt = self._fl_T.diff(dynamicsymbols._t).subs({self._l_T_tilde.diff(dynamicsymbols._t): self._v_T_tilde})

        # 定义状态变量 `_state_vars` 为包含 `_F_T_tilde` 的矩阵
        self._state_vars = Matrix([self._F_T_tilde])
        # 定义输入变量 `_input_vars` 为空的矩阵
        self._input_vars = zeros(0, 1)
        # 定义状态方程 `_state_eqns` 为包含 `_dF_T_tilde_dt` 的矩阵
        self._state_eqns = Matrix([self._dF_T_tilde_dt])
        # 定义曲线常数 `_curve_constants` 为包含所有常数的矩阵，如果设置了 `_with_defaults` 标志，则为空的矩阵
        self._curve_constants = Matrix(
            fl_T_constants
            + fl_M_pas_constants
            + fl_M_act_constants
            + fv_M_constants
        ) if not self._with_defaults else zeros(0, 1)
    # 定义一个未实现的方法，用于计算隐式肌腱动力学的纤维长度
    def _fiber_length_implicit_musculotendon_dynamics(self):
        raise NotImplementedError

    # 定义一个未实现的方法，用于计算隐式肌腱动力学的肌腱力
    def _tendon_force_implicit_musculotendon_dynamics(self):
        raise NotImplementedError

    # 定义属性 state_vars，返回按时间函数排列的状态变量的列矩阵
    @property
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``x`` can also be used to access the same attribute.

        """
        # 初始状态变量列表包含 self._state_vars
        state_vars = [self._state_vars]
        # 遍历所有子对象，获取它们的状态变量，并添加到列表中
        for child in self._child_objects:
            state_vars.append(child.state_vars)
        # 使用 Matrix.vstack 方法将所有状态变量组成的列表竖直堆叠成一个矩阵，并返回
        return Matrix.vstack(*state_vars)

    # 定义属性 x，是 state_vars 属性的别名，返回按时间函数排列的状态变量的列矩阵
    @property
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``state_vars`` can also be used to access the same attribute.

        """
        # 初始状态变量列表包含 self._state_vars
        state_vars = [self._state_vars]
        # 遍历所有子对象，获取它们的状态变量，并添加到列表中
        for child in self._child_objects:
            state_vars.append(child.state_vars)
        # 使用 Matrix.vstack 方法将所有状态变量组成的列表竖直堆叠成一个矩阵，并返回
        return Matrix.vstack(*state_vars)

    # 定义属性 input_vars，返回按时间函数排列的输入变量的列矩阵
    @property
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``r`` can also be used to access the same attribute.

        """
        # 初始输入变量列表包含 self._input_vars
        input_vars = [self._input_vars]
        # 遍历所有子对象，获取它们的输入变量，并添加到列表中
        for child in self._child_objects:
            input_vars.append(child.input_vars)
        # 使用 Matrix.vstack 方法将所有输入变量组成的列表竖直堆叠成一个矩阵，并返回
        return Matrix.vstack(*input_vars)

    # 定义属性 r，是 input_vars 属性的别名，返回按时间函数排列的输入变量的列矩阵
    @property
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``input_vars`` can also be used to access the same attribute.

        """
        # 初始输入变量列表包含 self._input_vars
        input_vars = [self._input_vars]
        # 遍历所有子对象，获取它们的输入变量，并添加到列表中
        for child in self._child_objects:
            input_vars.append(child.input_vars)
        # 使用 Matrix.vstack 方法将所有输入变量组成的列表竖直堆叠成一个矩阵，并返回
        return Matrix.vstack(*input_vars)
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Explanation
        ===========

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        The alias ``p`` can also be used to access the same attribute.

        """
        # List of musculotendon constants that are symbols (not numeric)
        musculotendon_constants = [
            self._l_T_slack,
            self._F_M_max,
            self._l_M_opt,
            self._v_M_max,
            self._alpha_opt,
            self._beta,
        ]
        # Filter out numeric constants, leaving only symbolic constants
        musculotendon_constants = [
            c for c in musculotendon_constants if not c.is_number
        ]
        # Create a Matrix of symbolic constants or an empty Matrix if none
        constants = [
            Matrix(musculotendon_constants)
            if musculotendon_constants
            else zeros(0, 1)
        ]
        # Append constants from child objects
        for child in self._child_objects:
            constants.append(child.constants)
        # Append curve constants
        constants.append(self._curve_constants)
        # Return a vertically stacked Matrix of all constants
        return Matrix.vstack(*constants)

    @property
    def p(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Explanation
        ===========

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        The alias ``constants`` can also be used to access the same attribute.

        """
        # List of musculotendon constants that are symbols (not numeric)
        musculotendon_constants = [
            self._l_T_slack,
            self._F_M_max,
            self._l_M_opt,
            self._v_M_max,
            self._alpha_opt,
            self._beta,
        ]
        # Filter out numeric constants, leaving only symbolic constants
        musculotendon_constants = [
            c for c in musculotendon_constants if not c.is_number
        ]
        # Create a Matrix of symbolic constants or an empty Matrix if none
        constants = [
            Matrix(musculotendon_constants)
            if musculotendon_constants
            else zeros(0, 1)
        ]
        # Append constants from child objects
        for child in self._child_objects:
            constants.append(child.constants)
        # Append curve constants
        constants.append(self._curve_constants)
        # Return a vertically stacked Matrix of all constants
        return Matrix.vstack(*constants)
    # 定义方法 M，返回矩阵 M，用于线性系统 ``M x' = F`` 的左手边系数矩阵

    M = [eye(len(self._state_vars))]  # 创建一个单位矩阵，维度为状态变量的长度
    for child in self._child_objects:
        M.append(child.M)  # 将子对象的 M 方法返回的矩阵添加到 M 中
    return diag(*M)  # 返回 M 中各矩阵对角线组成的对角矩阵

    @property
    # 定义属性 F，返回矩阵 F，用于线性系统 ``M x' = F`` 的右手边方程列向量

    F = [self._state_eqns]  # 将状态方程作为初始 F
    for child in self._child_objects:
        F.append(child.F)  # 将子对象的 F 属性返回的矩阵添加到 F 中
    return Matrix.vstack(*F)  # 垂直堆叠 F 中的所有矩阵，形成一个列向量矩阵

    # 定义方法 rhs，返回用于解 ``M x' = F`` 的方程列向量矩阵

    is_explicit = (
        MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
        MusculotendonFormulation.TENDON_FORCE_EXPLICIT,
    )
    if self.musculotendon_dynamics is MusculotendonFormulation.RIGID_TENDON:
        child_rhs = [child.rhs() for child in self._child_objects]  # 递归调用子对象的 rhs 方法并收集结果
        return Matrix.vstack(*child_rhs)  # 垂直堆叠所有子对象的 rhs 方法返回的矩阵

    elif self.musculotendon_dynamics in is_explicit:
        rhs = self._state_eqns  # 获取当前对象的状态方程作为 rhs
        child_rhs = [child.rhs() for child in self._child_objects]  # 递归调用子对象的 rhs 方法并收集结果
        return Matrix.vstack(rhs, *child_rhs)  # 垂直堆叠当前对象的 rhs 和所有子对象的 rhs 方法返回的矩阵

    return self.M.solve(self.F)  # 使用矩阵 M 的 solve 方法解线性系统 ``M x' = F``
    # 返回一个字符串表示，用于重新实例化模型
    def __repr__(self):
        """Returns a string representation to reinstantiate the model."""
        # 使用类名和各个属性的字符串表示形式构造一个包含模型信息的字符串
        return (
            f'{self.__class__.__name__}({self.name!r}, '  # 模型类名及其名称
            f'pathway={self.pathway!r}, '  # 模型的路径
            f'activation_dynamics={self.activation_dynamics!r}, '  # 模型的激活动力学
            f'musculotendon_dynamics={self.musculotendon_dynamics}, '  # 肌腱动力学
            f'tendon_slack_length={self._l_T_slack!r}, '  # 肌腱松弛长度
            f'peak_isometric_force={self._F_M_max!r}, '  # 峰值等长力
            f'optimal_fiber_length={self._l_M_opt!r}, '  # 最佳纤维长度
            f'maximal_fiber_velocity={self._v_M_max!r}, '  # 最大纤维速度
            f'optimal_pennation_angle={self._alpha_opt!r}, '  # 最佳羽毛角
            f'fiber_damping_coefficient={self._beta!r})'  # 纤维阻尼系数
        )

    # 返回关于肌腱肌力的表达式的字符串表示形式
    def __str__(self):
        """Returns a string representation of the expression for musculotendon
        force."""
        # 返回肌腱肌力的字符串表示形式
        return str(self.force)
class MusculotendonDeGroote2016(MusculotendonBase):
    r"""Musculotendon model using the curves of De Groote et al., 2016 [1]_.

    Examples
    ========

    This class models the musculotendon actuator parametrized by the
    characteristic curves described in De Groote et al., 2016 [1]_. Like all
    musculotendon models in SymPy's biomechanics module, it requires a pathway
    to define its line of action. We'll begin by creating a simple
    ``LinearPathway`` between two points that our musculotendon will follow.
    We'll create a point ``O`` to represent the musculotendon's origin and
    another ``I`` to represent its insertion.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearPathway, Point,
    ...     ReferenceFrame, dynamicsymbols)

    >>> N = ReferenceFrame('N')  # 定义一个惯性参考系 N
    >>> O, I = O, P = symbols('O, I', cls=Point)  # 创建两个点 O 和 I 作为 Point 类的实例
    >>> q, u = dynamicsymbols('q, u', real=True)  # 定义动力学符号 q 和 u
    >>> I.set_pos(O, q*N.x)  # 设置点 I 相对于点 O 的位置
    >>> O.set_vel(N, 0)  # 设置点 O 在参考系 N 中的速度为零
    >>> I.set_vel(N, u*N.x)  # 设置点 I 在参考系 N 中的速度为 u*N.x
    >>> pathway = LinearPathway(O, I)  # 创建一个 LinearPathway 对象，连接点 O 和 I
    >>> pathway.attachments  # 获取 LinearPathway 对象的 attachments 属性，应为 (O, I)
    (O, I)
    >>> pathway.length  # 获取 LinearPathway 对象的 length 属性，应为 Abs(q(t))
    Abs(q(t))
    >>> pathway.extension_velocity  # 获取 LinearPathway 对象的 extension_velocity 属性，应为 sign(q(t))*Derivative(q(t), t)

    A musculotendon also takes an instance of an activation dynamics model as
    this will be used to provide symbols for the activation in the formulation
    of the musculotendon dynamics. We'll use an instance of
    ``FirstOrderActivationDeGroote2016`` to represent first-order activation
    dynamics. Note that a single name argument needs to be provided as SymPy
    will use this as a suffix.

    >>> from sympy.physics.biomechanics import FirstOrderActivationDeGroote2016

    >>> activation = FirstOrderActivationDeGroote2016('muscle')  # 创建 FirstOrderActivationDeGroote2016 类的实例，命名为 'muscle'
    >>> activation.x  # 获取 activation 对象的 x 属性，应为 Matrix([[a_muscle(t)]])
    Matrix([[a_muscle(t)]])
    >>> activation.r  # 获取 activation 对象的 r 属性，应为 Matrix([[e_muscle(t)]])
    Matrix([[e_muscle(t)]])
    >>> activation.p  # 获取 activation 对象的 p 属性，应为 Matrix([[tau_a_muscle], [tau_d_muscle], [b_muscle]])
    Matrix([
    [tau_a_muscle],
    [tau_d_muscle],
    [    b_muscle]])
    >>> activation.rhs()  # 调用 activation 对象的 rhs() 方法，返回激活动力学方程的右手边
    Matrix([[((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

    The musculotendon class requires symbols or values to be passed to represent
    the constants in the musculotendon dynamics. We'll use SymPy's ``symbols``
    function to create symbols for the maximum isometric force ``F_M_max``,
    optimal fiber length ``l_M_opt``, tendon slack length ``l_T_slack``, maximum
    fiber velocity ``v_M_max``, optimal pennation angle ``alpha_opt, and fiber
    damping coefficient ``beta``.

    >>> F_M_max = symbols('F_M_max', real=True)  # 创建实数符号 F_M_max
    >>> l_M_opt = symbols('l_M_opt', real=True)  # 创建实数符号 l_M_opt
    >>> l_T_slack = symbols('l_T_slack', real=True)  # 创建实数符号 l_T_slack
    >>> v_M_max = symbols('v_M_max', real=True)  # 创建实数符号 v_M_max
    >>> alpha_opt = symbols('alpha_opt', real=True)  # 创建实数符号 alpha_opt
    >>> beta = symbols('beta', real=True)  # 创建实数符号 beta

    We can then import the class ``MusculotendonDeGroote2016`` from the
    biomechanics module and create an instance by passing in the various objects
    we have previously instantiated. By default, a musculotendon model with
    rigid tendon musculotendon dynamics will be created.
    # 创建刚性肌腱肌肉动力学模型。

    >>> from sympy.physics.biomechanics import MusculotendonDeGroote2016

    >>> rigid_tendon_muscle = MusculotendonDeGroote2016(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ...     maximal_fiber_velocity=v_M_max,
    ...     optimal_pennation_angle=alpha_opt,
    ...     fiber_damping_coefficient=beta,
    ... )
    # 使用 Sympy 中的 MusculotendonDeGroote2016 类创建刚性肌腱肌肉模型实例，
    # 传入了必要的参数和常量。

    We can inspect the various properties of the musculotendon, including
    getting the symbolic expression describing the force it produces using its
    ``force`` attribute.

    >>> rigid_tendon_muscle.force
    -F_M_max*(beta*(-l_T_slack + Abs(q(t)))*sign(q(t))*Derivative(q(t), t)...
    # 可以通过访问 force 属性来获取描述该肌腱肌肉模型产生力的符号表达式。

    When we created the musculotendon object, we passed in an instance of an
    activation dynamics object that governs the activation within the
    musculotendon. SymPy makes a design choice here that the activation dynamics
    instance will be treated as a child object of the musculotendon dynamics.
    Therefore, if we want to inspect the state and input variables associated
    with the musculotendon model, we will also be returned the state and input
    variables associated with the child object, or the activation dynamics in
    this case. As the musculotendon model that we created here uses rigid tendon
    dynamics, no additional states or inputs relating to the musculotendon are
    introduces. Consequently, the model has a single state associated with it,
    the activation, and a single input associated with it, the excitation. The
    states and inputs can be inspected using the ``x`` and ``r`` attributes
    respectively. Note that both ``x`` and ``r`` have the alias attributes of
    ``state_vars`` and ``input_vars``.

    >>> rigid_tendon_muscle.x
    Matrix([[a_muscle(t)]])
    >>> rigid_tendon_muscle.r
    Matrix([[e_muscle(t)]])
    # 在创建肌腱肌肉对象时，我们传入了一个激活动力学对象，它控制肌腱肌肉内的激活过程。
    # SymPy 设计选择将激活动力学实例视为肌腱肌肉动力学的子对象。因此，如果要检查与肌腱肌肉模型相关的状态和输入变量，
    # 也会返回与子对象（即激活动力学）相关的状态和输入变量。
    # 由于此处创建的肌腱肌肉模型使用刚性肌腱动力学，因此不会引入与肌腱肌肉相关的额外状态或输入。
    # 因此，模型只有一个状态与之相关联，即激活，和一个与之相关联的输入，即激励。
    # 可以分别使用 ``x`` 和 ``r`` 属性来检查状态和输入。请注意，``x`` 和 ``r`` 还具有别名属性 ``state_vars`` 和 ``input_vars``。

    To see which constants are symbolic in the musculotendon model, we can use
    the ``p`` or ``constants`` attribute. This returns a ``Matrix`` populated
    by the constants that are represented by a ``Symbol`` rather than a numeric
    value.

    >>> rigid_tendon_muscle.p
    Matrix([
    [           l_T_slack],
    [             F_M_max],
    [             l_M_opt],
    [             v_M_max],
    [           alpha_opt],
    [                beta],
    [        tau_a_muscle],
    [        tau_d_muscle],
    [            b_muscle],
    [     c_0_fl_T_muscle],
    [     c_1_fl_T_muscle],
    [     c_2_fl_T_muscle],
    [     c_3_fl_T_muscle],
    [ c_0_fl_M_pas_muscle],
    [ c_1_fl_M_pas_muscle],
    [ c_0_fl_M_act_muscle],
    [ c_1_fl_M_act_muscle],
    [ c_2_fl_M_act_muscle],
    [ c_3_fl_M_act_muscle],
    [ c_4_fl_M_act_muscle],
    [ c_5_fl_M_act_muscle],
    [ c_6_fl_M_act_muscle],
    [ c_7_fl_M_act_muscle],
    [ c_8_fl_M_act_muscle],
    [ c_9_fl_M_act_muscle],
    # 要查看肌腱肌肉模型中哪些常量是符号常量，可以使用 ``p`` 或 ``constants`` 属性。
    # 这将返回一个 ``Matrix``，其中包含由 ``Symbol`` 表示而不是数值的常量。
    [c_10_fl_M_act_muscle],
    [c_11_fl_M_act_muscle],
    [     c_0_fv_M_muscle],
    [     c_1_fv_M_muscle],
    [     c_2_fv_M_muscle],
    [     c_3_fv_M_muscle]])


# 定义了一个包含多个列表的大列表，每个小列表包含了变量或常量的名称。
# 这些名称代表了肌肉-肌腱模型中的各种参数，用于描述肌肉力学特性和动力学行为。



    Finally, we can call the ``rhs`` method to return a ``Matrix`` that
    contains as its elements the righthand side of the ordinary differential
    equations corresponding to each of the musculotendon's states. Like the
    method with the same name on the ``Method`` classes in SymPy's mechanics
    module, this returns a column vector where the number of rows corresponds to
    the number of states. For our example here, we have a single state, the
    dynamic symbol ``a_muscle(t)``, so the returned value is a 1-by-1
    ``Matrix``.


# 最后，我们可以调用 ``rhs`` 方法返回一个 ``Matrix`` 对象，
# 它包含肌肉-肌腱模型各状态的常微分方程右侧的元素。
# 类似于 SymPy 力学模块中 ``Method`` 类的同名方法，
# 这个方法返回一个列向量，其行数对应于状态的数量。
# 在我们的例子中，我们有一个状态，即动态符号 ``a_muscle(t)``，
# 因此返回的值是一个 1x1 的 ``Matrix``。



    >>> rigid_tendon_muscle.rhs()
    Matrix([[((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])


# 调用 ``rhs`` 方法计算刚性肌腱模型的右手边（常微分方程的右侧），
# 返回一个包含表达式的 ``Matrix`` 对象，描述肌肉-肌腱系统的动力学行为。



    The musculotendon class supports elastic tendon musculotendon models in
    addition to rigid tendon ones. You can choose to either use the fiber length
    or tendon force as an additional state. You can also specify whether an
    explicit or implicit formulation should be used. To select a formulation,
    pass a member of the ``MusculotendonFormulation`` enumeration to the
    ``musculotendon_dynamics`` parameter when calling the constructor. This
    enumeration is an ``IntEnum``, so you can also pass an integer, however it
    is recommended to use the enumeration as it is clearer which formulation you
    are actually selecting. Below, we'll use the ``FIBER_LENGTH_EXPLICIT``
    member to create a musculotendon with an elastic tendon that will use the
    (normalized) muscle fiber length as an additional state and will produce
    the governing ordinary differential equation in explicit form.


# musculotendon 类支持弹性肌腱模型和刚性肌腱模型。
# 您可以选择将肌肉纤维长度或肌腱力作为额外的状态。
# 您还可以指定使用显式或隐式公式。要选择一个公式，
# 在调用构造函数时将 ``MusculotendonFormulation`` 枚举的成员传递给
# ``musculotendon_dynamics`` 参数。这个枚举是一个 ``IntEnum``，
# 因此也可以传递整数，但建议使用枚举，因为这样更清楚您实际选择了哪种公式。
# 下面，我们将使用 ``FIBER_LENGTH_EXPLICIT`` 成员创建一个弹性肌腱模型，
# 该模型将使用（归一化的）肌肉纤维长度作为额外的状态，并以显式形式生成主导的常微分方程。



    >>> from sympy.physics.biomechanics import MusculotendonFormulation


# 导入 SymPy 的生物力学模块中的 MusculotendonFormulation 类。



    >>> elastic_tendon_muscle = MusculotendonDeGroote2016(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     musculotendon_dynamics=MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ...     maximal_fiber_velocity=v_M_max,
    ...     optimal_pennation_angle=alpha_opt,
    ...     fiber_damping_coefficient=beta,
    ... )


# 创建一个名为 elastic_tendon_muscle 的 MusculotendonDeGroote2016 对象，
# 用于描述一个具有弹性肌腱的肌肉模型。构造函数参数包括：
# - 'muscle'：肌肉的名称
# - pathway：肌肉通路（假设是一个变量或对象）
# - activation：激活（假设是一个变量或对象）
# - musculotendon_dynamics=MusculotendonFormulation.FIBER_LENGTH_EXPLICIT：
#   使用 MusculotendonFormulation 类中的 FIBER_LENGTH_EXPLICIT 枚举成员，
#   指定肌肉-肌腱动力学的显式公式
# - tendon_slack_length=l_T_slack：肌腱松弛长度
# - peak_isometric_force=F_M_max：峰值等长力
# - optimal_fiber_length=l_M_opt：最佳纤维长度
# - maximal_fiber_velocity=v_M_max：最大纤维速度
# - optimal_pennation_angle=alpha_opt：最佳针尖角
# - fiber_damping_coefficient=beta：纤维阻尼系数



    >>> elastic_tendon_muscle.force
    -F_M_max*TendonForceLengthDeGroote2016((-sqrt(l_M_opt**2*...


# 访问 elastic_tendon_muscle 对象的 force 属性，返回描述弹性肌腱力的表达式。



    >>> elastic_tendon_muscle.x
    Matrix([
    [l_M_tilde_muscle(t)],
    [        a_muscle(t)]])


# 访问 elastic_tendon_muscle 对象的 x 属性，返回一个描述弹性肌腱模型状态的 ``Matrix`` 对象。
# 包括了归一化肌肉纤维长度（l_M_tilde_muscle(t)）和肌肉激活（a_muscle(t)）。



    >>> elastic_tendon_muscle.r
    Matrix([[e_muscle(t)]])


# 访问 elastic_tendon_muscle 对象的 r 属性，返回一个描述弹性肌腱模型的 e_muscle(t) 状态的 ``Matrix`` 对象。



    >>> elastic_tendon_muscle.p
    Matrix([
    [           l_T_slack],
    [             F_M_max],
    [             l_M_opt],
    [             v_M_max],
    [           alpha_opt],
    [                beta],
    [        tau_a_muscle],
    [        tau_d_muscle],
    [            b_muscle],
    [     c_0_fl_T_muscle],
    [     c_1_fl_T_muscle],
    [     c_2_fl_T_muscle],
    [     c_3_fl_T_muscle],
    [ c_0_fl_M_pas_muscle],


# 访问 elastic_tendon_muscle 对象的 p 属性，返回一个描述弹性肌腱模型的参数的 ``Matrix`` 对象。
# 包括了肌腱松弛长度、峰值等长力、最佳纤维长度、最大纤维速度、最佳针尖角、
# 纤维阻尼系
    [ c_1_fl_M_pas_muscle],  # 创建包含单个元素的列表，这个元素是一个标识符为 c_1_fl_M_pas_muscle 的对象
    [ c_0_fl_M_act_muscle],  # 同上，标识符为 c_0_fl_M_act_muscle
    [ c_1_fl_M_act_muscle],  # 同上，标识符为 c_1_fl_M_act_muscle
    [ c_2_fl_M_act_muscle],  # 同上，标识符为 c_2_fl_M_act_muscle
    [ c_3_fl_M_act_muscle],  # 同上，标识符为 c_3_fl_M_act_muscle
    [ c_4_fl_M_act_muscle],  # 同上，标识符为 c_4_fl_M_act_muscle
    [ c_5_fl_M_act_muscle],  # 同上，标识符为 c_5_fl_M_act_muscle
    [ c_6_fl_M_act_muscle],  # 同上，标识符为 c_6_fl_M_act_muscle
    [ c_7_fl_M_act_muscle],  # 同上，标识符为 c_7_fl_M_act_muscle
    [ c_8_fl_M_act_muscle],  # 同上，标识符为 c_8_fl_M_act_muscle
    [ c_9_fl_M_act_muscle],  # 同上，标识符为 c_9_fl_M_act_muscle
    [c_10_fl_M_act_muscle],  # 同上，标识符为 c_10_fl_M_act_muscle
    [c_11_fl_M_act_muscle],  # 同上，标识符为 c_11_fl_M_act_muscle
    [     c_0_fv_M_muscle],  # 创建包含单个元素的列表，这个元素是一个标识符为 c_0_fv_M_muscle 的对象
    [     c_1_fv_M_muscle],  # 同上，标识符为 c_1_fv_M_muscle
    [     c_2_fv_M_muscle],  # 同上，标识符为 c_2_fv_M_muscle
    [     c_3_fv_M_muscle]])  # 同上，标识符为 c_3_fv_M_muscle

# 调用 elastic_tendon_muscle 对象的 rhs() 方法，返回一个包含数学表达式的矩阵
>>> elastic_tendon_muscle.rhs()
Matrix([
[v_M_max*FiberForceVelocityInverseDeGroote2016((l_M_opt*...],
[((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

# 建议在创建实例时使用 alternate 的 with_defaults 构造函数，以确保在肌腱特征曲线中使用发布的常量。
>>> elastic_tendon_muscle = MusculotendonDeGroote2016.with_defaults(
...     'muscle',
...     pathway,
...     activation,
...     musculotendon_dynamics=MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
...     tendon_slack_length=l_T_slack,
...     peak_isometric_force=F_M_max,
...     optimal_fiber_length=l_M_opt,
... )

# 访问 elastic_tendon_muscle 对象的属性 x，返回一个包含状态变量的矩阵
>>> elastic_tendon_muscle.x
Matrix([
[l_M_tilde_muscle(t)],
[        a_muscle(t)]])

# 访问 elastic_tendon_muscle 对象的属性 r，返回一个包含外部刺激的矩阵
>>> elastic_tendon_muscle.r
Matrix([[e_muscle(t)]])

# 访问 elastic_tendon_muscle 对象的属性 p，返回一个包含参数的矩阵
>>> elastic_tendon_muscle.p
Matrix([
[   l_T_slack],
[     F_M_max],
[     l_M_opt],
[tau_a_muscle],
[tau_d_muscle],
[    b_muscle]])

# Parameters 部分，以下是关于 MusculotendonDeGroote2016 类构造函数的参数说明
name : str
    与肌肉肌腱相关的名称标识符。当自动生成符号时，此名称用作后缀。
    必须是非零长度的字符串。
pathway : PathwayBase
    执行器遵循的路径。必须是 PathwayBase 的一个具体子类的实例，如 LinearPathway。
activation_dynamics : ActivationBase
    在肌肉肌腱内部建模的激活动力学。必须是 ActivationBase 的一个具体子类的实例，如 FirstOrderActivationDeGroote2016。
    musculotendon_dynamics : MusculotendonFormulation | int
        # 肌腱动力学的配方，可以是整数枚举类型``MusculotendonFormulation``的成员，或可以转换为成员的整数值
        # 若要使用刚性肌腱模型，设为``MusculotendonFormulation.RIGID_TENDON``（或整数值``0``，将被转换为枚举成员）
        # 弹性肌腱模型有四种可能的配方。若要使用显式配方，以肌纤维长度作为状态，则设为
        # ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT``（或整数值``1``）；
        # 以肌腱力作为状态，则设为``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``（或整数值``2``）；
        # 若要使用隐式配方，以肌纤维长度作为状态，则设为
        # ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT``（或整数值``3``）；
        # 以肌腱力作为状态，则设为``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``（或整数值``4``）。
        # 默认为``MusculotendonFormulation.RIGID_TENDON``，对应刚性肌腱模型。

    tendon_slack_length : Expr | None
        # 肌腱松弛长度，在肌腱处于无负荷状态时的长度。
        # 在刚性肌腱模型中，肌腱长度即为肌腱松弛长度。
        # 在所有肌腱模型中，肌腱松弛长度用于归一化肌腱长度，得到：math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`。

    peak_isometric_force : Expr | None
        # 肌纤维等长收缩时能产生的最大力量。
        # 在所有肌腱模型中，最大等长收缩力被用来归一化肌腱和肌纤维力量，得到：math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`。

    optimal_fiber_length : Expr | None
        # 肌纤维长度，当肌纤维不产生任何被动力和其最大主动力时。
        # 在所有肌腱模型中，最佳肌纤维长度被用来归一化肌纤维长度，得到：math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`。

    maximal_fiber_velocity : Expr | None
        # 肌纤维最大速度，在肌纤维缩短期间，肌纤维无法产生任何主动力时的速度。
        # 在所有肌腱模型中，最大肌纤维速度被用来归一化肌纤维扩展速度，得到：math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`。

    optimal_pennation_angle : Expr | None
        # 最佳肌纤维长度时的肌肉腱角度。
    fiber_damping_coefficient : Expr | None
        The coefficient of damping to be used in the damping element in the
        muscle fiber model.
    with_defaults : bool
        Whether ``with_defaults`` alternate constructors should be used when
        automatically constructing child classes. Default is ``False``.

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    # 创建一个包含不同特征曲线类的集合对象，用于描述肌肉模型中的力-长度和力-速度关系
    curves = CharacteristicCurveCollection(
        tendon_force_length=TendonForceLengthDeGroote2016,
        tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,
        fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,
        fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,
        fiber_force_length_active=FiberForceLengthActiveDeGroote2016,
        fiber_force_velocity=FiberForceVelocityDeGroote2016,
        fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,
    )
```