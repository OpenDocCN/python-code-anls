# `D:\src\scipysrc\scipy\scipy\constants\_codata.py`

```
"""
Fundamental Physical Constants
------------------------------

These constants are taken from CODATA Recommended Values of the Fundamental
Physical Constants 2018.

Object
------
physical_constants : dict
    A dictionary containing physical constants. Keys are the names of physical
    constants, values are tuples (value, units, precision).

Functions
---------
value(key):
    Returns the value of the physical constant(key).
unit(key):
    Returns the units of the physical constant(key).
precision(key):
    Returns the relative precision of the physical constant(key).
find(sub):
    Prints or returns list of keys containing the string sub, default is all.

Source
------
The values of the constants provided at this site are recommended for
international use by CODATA and are the latest available. Termed the "2018
CODATA recommended values," they are generally recognized worldwide for use in
all fields of science and technology. The values became available on 20 May
2019 and replaced the 2014 CODATA set. Also available is an introduction to the
constants for non-experts at

https://physics.nist.gov/cuu/Constants/introduction.html

References
----------
Theoretical and experimental publications relevant to the fundamental constants
and closely related precision measurements published since the mid 1980s, but
also including many older papers of particular interest, some of which date
back to the 1800s. To search the bibliography, visit

https://physics.nist.gov/cuu/Constants/
"""

# Compiled by Charles Harris, dated October 3, 2002
# updated to 2002 values by BasSw, 2006
# Updated to 2006 values by Vincent Davis June 2010
# Updated to 2014 values by Joseph Booker, 2015
# Updated to 2018 values by Jakob Jakobson, 2019

from __future__ import annotations

import warnings

from typing import Any

__all__ = ['physical_constants', 'value', 'unit', 'precision', 'find',
           'ConstantWarning']

"""
Source:  https://physics.nist.gov/cuu/Constants/

The values of the constants provided at this site are recommended for
international use by CODATA and are the latest available. Termed the "2018
CODATA recommended values," they are generally recognized worldwide for use in
all fields of science and technology. The values became available on 20 May
2019 and replaced the 2014 CODATA set.
"""

#
# Source:  https://physics.nist.gov/cuu/Constants/
#

# Quantity                                             Value                 Uncertainty          Unit
# ---------------------------------------------------- --------------------- -------------------- -------------
# 定义包含基本物理常数的文本，包括名称、值、不确定性和单位
txt2002 = """\
Wien displacement law constant                         2.897 7685e-3         0.000 0051e-3         m K
atomic unit of 1st hyperpolarizablity                  3.206 361 51e-53      0.000 000 28e-53      C^3 m^3 J^-2
atomic unit of 2nd hyperpolarizablity                  6.235 3808e-65        0.000 0011e-65        C^4 m^4 J^-3
"""
# atomic unit of electric dipole moment                  8.478 353 09e-30      0.000 000 73e-30      C m
atomic_unit_electric_dipole_moment = 8.47835309e-30  # 原子单位电偶极矩，单位为库仑米，值为8.47835309e-30 C m

# atomic unit of electric polarizablity                  1.648 777 274e-41     0.000 000 016e-41     C^2 m^2 J^-1
atomic_unit_electric_polarizability = 1.648777274e-41  # 原子单位电极化率，单位为库仑平方米每焦耳，值为1.648777274e-41 C^2 m^2 J^-1

# atomic unit of electric quadrupole moment              4.486 551 24e-40      0.000 000 39e-40      C m^2
atomic_unit_electric_quadrupole_moment = 4.48655124e-40  # 原子单位电四极矩，单位为库仑米平方，值为4.48655124e-40 C m^2

# atomic unit of magn. dipole moment                     1.854 801 90e-23      0.000 000 16e-23      J T^-1
atomic_unit_magnetic_dipole_moment = 1.85480190e-23  # 原子单位磁偶极矩，单位为焦耳每特斯拉，值为1.85480190e-23 J T^-1

# atomic unit of magn. flux density                      2.350 517 42e5        0.000 000 20e5        T
atomic_unit_magnetic_flux_density = 2.35051742e5  # 原子单位磁通密度，单位为特斯拉，值为2.35051742e5 T

# deuteron magn. moment                                  0.433 073 482e-26     0.000 000 038e-26     J T^-1
deuteron_magnetic_moment = 0.433073482e-26  # 氘核磁矩，单位为焦耳每特斯拉，值为0.433073482e-26 J T^-1

# deuteron magn. moment to Bohr magneton ratio           0.466 975 4567e-3     0.000 000 0050e-3
deuteron_to_Bohr_magneton_ratio = 0.4669754567e-3  # 氘核磁矩与玻尔磁子比值，值为0.4669754567e-3

# deuteron magn. moment to nuclear magneton ratio        0.857 438 2329        0.000 000 0092
deuteron_to_nuclear_magneton_ratio = 0.8574382329  # 氘核磁矩与核磁子比值，值为0.8574382329

# deuteron-electron magn. moment ratio                   -4.664 345 548e-4     0.000 000 050e-4
deuteron_to_electron_magnetic_moment_ratio = -4.664345548e-4  # 氘-电子磁矩比值，值为-4.664345548e-4

# deuteron-proton magn. moment ratio                     0.307 012 2084        0.000 000 0045
deuteron_to_proton_magnetic_moment_ratio = 0.3070122084  # 氘-质子磁矩比值，值为0.3070122084

# deuteron-neutron magn. moment ratio                    -0.448 206 52         0.000 000 11
deuteron_to_neutron_magnetic_moment_ratio = -0.44820652  # 氘-中子磁矩比值，值为-0.44820652

# electron gyromagn. ratio                               1.760 859 74e11       0.000 000 15e11       s^-1 T^-1
electron_gyromagnetic_ratio = 1.76085974e11  # 电子旋磁比，单位为每秒每特斯拉，值为1.76085974e11 s^-1 T^-1

# electron gyromagn. ratio over 2 pi                     28 024.9532           0.0024                MHz T^-1
electron_gyromagnetic_ratio_over_2pi = 28024.9532  # 电子旋磁比除以2π，单位为兆赫每特斯拉，值为28024.9532 MHz T^-1

# electron magn. moment                                  -928.476 412e-26      0.000 080e-26         J T^-1
electron_magnetic_moment = -928.476412e-26  # 电子磁矩，单位为焦耳每特斯拉，值为-928.476412e-26 J T^-1

# electron magn. moment to Bohr magneton ratio           -1.001 159 652 1859   0.000 000 000 0038
electron_to_Bohr_magneton_ratio = -1.0011596521859  # 电子磁矩与玻尔磁子比值，值为-1.0011596521859

# electron magn. moment to nuclear magneton ratio        -1838.281 971 07      0.000 000 85
electron_to_nuclear_magneton_ratio = -1838.28197107  # 电子磁矩与核磁子比值，值为-1838.28197107

# electron magn. moment anomaly                          1.159 652 1859e-3     0.000 000 0038e-3
electron_magnetic_moment_anomaly = 1.1596521859e-3  # 电子磁矩异常，值为1.1596521859e-3

# electron to shielded proton magn. moment ratio         -658.227 5956         0.000 0071
electron_to_shielded_proton_magnetic_moment_ratio = -658.2275956  # 电子到屏蔽质子磁矩比值，值为-658.2275956

# electron to shielded helion magn. moment ratio         864.058 255           0.000 010
electron_to_shielded_helion_magnetic_moment_ratio = 864.058255  # 电子到屏蔽氦三磁矩比值，值为864.058255

# electron-deuteron magn. moment ratio                   -2143.923 493         0.000 023
electron_to_deuteron_magnetic_moment_ratio = -2143.923493  # 电子-氘磁矩比值，值为-2143.923493

# electron-muon magn. moment ratio                       206.766 9894          0.000 0054
electron_to_muon_magnetic_moment_ratio = 206.7669894  # 电子-μ子磁矩比值，值为206.7669894

# electron-neutron magn. moment ratio                    960.920 50            0.000 23
electron_to_neutron_magnetic_moment_ratio = 960.92050  # 电子-中子磁矩比值，值为960.92050

# electron-proton magn. moment ratio                     -658.210 6862         0.000 0066
electron_to_proton_magnetic_moment_ratio = -658.2106862  # 电子-质子磁矩比值，值为-658.2106862

# magn. constant                                         12.566 370 614...e-7  0                     N A^-2
magnetic_constant = 12.566370614e-7  # 磁常数，单位为牛顿每安培平方，值为12.566370614e-7 N A^-2

# magn. flux quantum                                     2.067 833 72e-15      0.000 000 18e-15      Wb
magnetic_flux_quantum = 2.06783372e-15  # 磁通量量子，单位为韦伯，值为
# neutron gyromagn. ratio over 2 pi
neutron_gyromagn_ratio = 29.1646950           # 基本数据：29.1646950，单位：MHz T^-1
neutron_gyromagn_ratio_error = 0.0000073      # 数据误差：0.0000073，单位：MHz T^-1

# neutron magn. moment
neutron_magnetic_moment = -0.96623645e-26     # 基本数据：-0.96623645 × 10^-26，单位：J T^-1
neutron_magnetic_moment_error = 0.00000024e-26# 数据误差：0.00000024 × 10^-26，单位：J T^-1

# neutron magn. moment to Bohr magneton ratio
neutron_to_bohr_magneton_ratio = -1.04187563e-3   # 基本数据：-1.04187563 × 10^-3
neutron_to_bohr_magneton_ratio_error = 0.00000025e-3 # 数据误差：0.00000025 × 10^-3

# neutron magn. moment to nuclear magneton ratio
neutron_to_nuclear_magneton_ratio = -1.91304273    # 基本数据：-1.91304273
neutron_to_nuclear_magneton_ratio_error = 0.00000045 # 数据误差：0.00000045

# neutron to shielded proton magn. moment ratio
neutron_to_shielded_proton_magneton_ratio = -0.68499694   # 基本数据：-0.68499694
neutron_to_shielded_proton_magneton_ratio_error = 0.00000016 # 数据误差：0.00000016

# neutron-electron magn. moment ratio
neutron_to_electron_magneton_ratio = 1.04066882e-3    # 基本数据：1.04066882 × 10^-3
neutron_to_electron_magneton_ratio_error = 0.00000025e-3  # 数据误差：0.00000025 × 10^-3

# neutron-proton magn. moment ratio
neutron_to_proton_magneton_ratio = -0.68497934       # 基本数据：-0.68497934
neutron_to_proton_magneton_ratio_error = 0.00000016   # 数据误差：0.00000016

# proton gyromagn. ratio
proton_gyromagn_ratio = 2.67522205e8        # 基本数据：2.67522205 × 10^8，单位：s^-1 T^-1
proton_gyromagn_ratio_error = 0.00000023e8  # 数据误差：0.00000023 × 10^8，单位：s^-1 T^-1

# proton gyromagn. ratio over 2 pi
proton_gyromagn_ratio_over_2pi = 42.5774813  # 基本数据：42.5774813，单位：MHz T^-1
proton_gyromagn_ratio_over_2pi_error = 0.0000037  # 数据误差：0.0000037，单位：MHz T^-1

# proton magn. moment
proton_magnetic_moment = 1.41060671e-26     # 基本数据：1.41060671 × 10^-26，单位：J T^-1
proton_magnetic_moment_error = 0.00000012e-26   # 数据误差：0.00000012 × 10^-26，单位：J T^-1

# proton magn. moment to Bohr magneton ratio
proton_to_bohr_magneton_ratio = 1.521032206e-3   # 基本数据：1.521032206 × 10^-3
proton_to_bohr_magneton_ratio_error = 0.000000015e-3  # 数据误差：0.000000015 × 10^-3

# proton magn. moment to nuclear magneton ratio
proton_to_nuclear_magneton_ratio = 2.792847351    # 基本数据：2.792847351
proton_to_nuclear_magneton_ratio_error = 0.000000028 # 数据误差：0.000000028

# proton magn. shielding correction
proton_magnetic_shielding_correction = 25.689e-6    # 基本数据：25.689 × 10^-6
proton_magnetic_shielding_correction_error = 0.015e-6  # 数据误差：0.015 × 10^-6

# proton-neutron magn. moment ratio
proton_to_neutron_magneton_ratio = -1.45989805    # 基本数据：-1.45989805
proton_to_neutron_magneton_ratio_error = 0.00000034  # 数据误差：0.00000034

# shielded helion gyromagn. ratio
shielded_helion_gyromagn_ratio = 2.03789470e8    # 基本数据：2.03789470 × 10^8，单位：s^-1 T^-1
shielded_helion_gyromagn_ratio_error = 0.00000018e8  # 数据误差：0.00000018 × 10^8，单位：s^-1 T^-1

# shielded helion gyromagn. ratio over 2 pi
shielded_helion_gyromagn_ratio_over_2pi = 32.4341015   # 基本数据：32.4341015，单位：MHz T^-1
shielded_helion_gyromagn_ratio_over_2pi_error = 0.0000028   # 数据误差：0.0000028，单位：MHz T^-1

# shielded helion magn. moment
shielded_helion_magnetic_moment = -1.074553024e-26   # 基本数据：-1.074553024 × 10^-26，单位：J T^-1
shielded_helion_magnetic_moment_error = 0.000000093e-26 # 数据误差：0.000000093 × 10^-26，单位：J T^-1

# shielded helion magn. moment to Bohr magneton ratio
shielded_helion_to_bohr_magneton_ratio = -1.158671474e-3   # 基本数据：-1.158671474 × 10^-3
shielded_helion_to_bohr_magneton_ratio_error = 0.000000014e-3 # 数据误差：0.000000014 × 10^-3

# shielded helion magn. moment to nuclear magneton ratio
shielded_helion_to_nuclear_magneton_ratio = -2.127497723   # 基本数据：-2.127497723
shielded_helion_to_nuclear_magneton_ratio_error = 0.000000025  # 数据误差：0.000000025

# shielded helion to proton magn. moment ratio
shielded_helion_to_proton_magneton_ratio = -0.761766562    # 基本数据：-0.761766562
shielded_helion_to_proton_magneton_ratio_error = 0.000000012  # 数据误差：0.000000012

# shielded helion to shielded proton magn. moment ratio
shielded_helion_to_shielded_proton_magneton_ratio = -0.7617861313   # 基本数据：-0.7617861313
shielded_helion_to_shielded_proton_magneton_ratio_error = 0.0000000033 # 数据误差：0.0000000033

# shielded proton magn. moment
shielded_proton_magnetic_moment = 1.41057047e-26     # 基本数据：1.41057047 × 10^-26，单位：J T^-1
shielded_proton_magnetic_moment_error = 0.00000012e-26   # 数据误差：0.00000012 × 10^-26，单位：J T^-1

# shielded proton magn. moment to Bohr magneton ratio
shielded_proton_to_bohr_magneton_ratio = 1.520993132e-3   # 基本数据：1.520993132 × 10^-3
shielded_proton_to_bohr_magneton_ratio_error = 0.000000016e-3 # 数据误差：0.000000016 × 10^-3

# shielded proton magn. moment to nuclear magneton ratio
shielded_proton_to_nuclear_magneton_ratio = 2.792775604     # 基本数据：2.792775604
shielded_proton_to_nuclear_magneton_ratio_error = 0.000000030   # 数据误
# alpha particle mass energy equivalent                  5.971 919 17 e-10     0.000 000 30 e-10     J
# alpha particle mass energy equivalent in MeV           3727.379 109          0.000 093             MeV
# alpha particle mass in u                               4.001 506 179 127     0.000 000 000 062     u
# alpha particle molar mass                              4.001 506 179 127 e-3 0.000 000 000 062 e-3 kg mol^-1
# alpha particle-proton mass ratio                       3.972 599 689 51      0.000 000 000 41
# Angstrom star                                          1.000 014 98 e-10     0.000 000 90 e-10     m
# atomic mass constant                                   1.660 538 782 e-27    0.000 000 083 e-27    kg
# atomic mass constant energy equivalent                 1.492 417 830 e-10    0.000 000 074 e-10    J
# atomic mass constant energy equivalent in MeV          931.494 028           0.000 023             MeV
# atomic mass unit-electron volt relationship            931.494 028 e6        0.000 023 e6          eV
# atomic mass unit-hartree relationship                  3.423 177 7149 e7     0.000 000 0049 e7     E_h
# atomic mass unit-hertz relationship                    2.252 342 7369 e23    0.000 000 0032 e23    Hz
# atomic mass unit-inverse meter relationship            7.513 006 671 e14     0.000 000 011 e14     m^-1
# atomic mass unit-joule relationship                    1.492 417 830 e-10    0.000 000 074 e-10    J
# atomic mass unit-kelvin relationship                   1.080 9527 e13        0.000 0019 e13        K
# atomic mass unit-kilogram relationship                 1.660 538 782 e-27    0.000 000 083 e-27    kg
# atomic unit of 1st hyperpolarizability                 3.206 361 533 e-53    0.000 000 081 e-53    C^3 m^3 J^-2
# atomic unit of 2nd hyperpolarizability                 6.235 380 95 e-65     0.000 000 31 e-65     C^4 m^4 J^-3
# atomic unit of action                                  1.054 571 628 e-34    0.000 000 053 e-34    J s
# atomic unit of charge                                  1.602 176 487 e-19    0.000 000 040 e-19    C
# atomic unit of charge density                          1.081 202 300 e12     0.000 000 027 e12     C m^-3
# atomic unit of current                                 6.623 617 63 e-3      0.000 000 17 e-3      A
# atomic unit of electric dipole mom.                    8.478 352 81 e-30     0.000 000 21 e-30     C m
# atomic unit of electric field                          5.142 206 32 e11      0.000 000 13 e11      V m^-1
# atomic unit of electric field gradient                 9.717 361 66 e21      0.000 000 24 e21      V m^-2
# atomic unit of electric polarizability                 1.648 777 2536 e-41   0.000 000 0034 e-41   C^2 m^2 J^-1
# atomic unit of electric potential                      27.211 383 86         0.000 000 68          V
# atomic unit of electric quadrupole mom.                4.486 551 07 e-40     0.000 000 11 e-40     C m^2
# atomic unit of energy                                  4.359 743 94 e-18     0.000 000 22 e-18     J
# atomic unit of force                                   8.238 722 06 e-8      0.000 000 41 e-8      N
atomic_unit_of_force = 8.23872206e-8  # 定义原子力单位，单位为牛顿

# atomic unit of length                                  0.529 177 208 59 e-10 0.000 000 000 36 e-10 m
atomic_unit_of_length = 0.52917720859e-10  # 定义原子长度单位，单位为米

# atomic unit of mag. dipole mom.                        1.854 801 830 e-23    0.000 000 046 e-23    J T^-1
atomic_unit_of_dipole_moment = 1.854801830e-23  # 定义原子磁偶极矩单位，单位为焦耳每特斯拉

# atomic unit of mag. flux density                       2.350 517 382 e5      0.000 000 059 e5      T
atomic_unit_of_flux_density = 2.350517382e5  # 定义原子磁通密度单位，单位为特斯拉

# atomic unit of magnetizability                         7.891 036 433 e-29    0.000 000 027 e-29    J T^-2
atomic_unit_of_magnetizability = 7.891036433e-29  # 定义原子磁化率单位，单位为焦耳每平方特斯拉

# atomic unit of mass                                    9.109 382 15 e-31     0.000 000 45 e-31     kg
atomic_unit_of_mass = 9.10938215e-31  # 定义原子质量单位，单位为千克

# atomic unit of momentum                                1.992 851 565 e-24    0.000 000 099 e-24    kg m s^-1
atomic_unit_of_momentum = 1.992851565e-24  # 定义原子动量单位，单位为千克米每秒

# atomic unit of permittivity                            1.112 650 056... e-10 (exact)               F m^-1
atomic_unit_of_permittivity = 1.112650056e-10  # 定义原子介电常数单位，单位为法拉每米

# atomic unit of time                                    2.418 884 326 505 e-17 0.000 000 000 016 e-17 s
atomic_unit_of_time = 2.418884326505e-17  # 定义原子时间单位，单位为秒

# atomic unit of velocity                                2.187 691 2541 e6     0.000 000 0015 e6     m s^-1
atomic_unit_of_velocity = 2.1876912541e6  # 定义原子速度单位，单位为米每秒

# Avogadro constant                                      6.022 141 79 e23      0.000 000 30 e23      mol^-1
Avogadro_constant = 6.02214179e23  # 定义阿伏伽德罗常数，单位为摩尔的倒数

# Bohr magneton                                          927.400 915 e-26      0.000 023 e-26        J T^-1
Bohr_magneton = 927.400915e-26  # 定义玻尔磁子，单位为焦耳每特斯拉

# Bohr magneton in eV/T                                  5.788 381 7555 e-5    0.000 000 0079 e-5    eV T^-1
Bohr_magneton_in_eV_per_T = 5.7883817555e-5  # 定义玻尔磁子单位（电子伏特每特斯拉）

# Bohr magneton in Hz/T                                  13.996 246 04 e9      0.000 000 35 e9       Hz T^-1
Bohr_magneton_in_Hz_per_T = 13.99624604e9  # 定义玻尔磁子单位（赫兹每特斯拉）

# Bohr magneton in inverse meters per tesla              46.686 4515           0.000 0012            m^-1 T^-1
Bohr_magneton_in_inverse_m_per_T = 46.6864515  # 定义玻尔磁子单位（每特斯拉的倒数米）

# Bohr magneton in K/T                                   0.671 7131            0.000 0012            K T^-1
Bohr_magneton_in_K_per_T = 0.6717131  # 定义玻尔磁子单位（每特斯拉的开尔文）

# Bohr radius                                            0.529 177 208 59 e-10 0.000 000 000 36 e-10 m
Bohr_radius = 0.52917720859e-10  # 定义玻尔半径，单位为米

# Boltzmann constant                                     1.380 6504 e-23       0.000 0024 e-23       J K^-1
Boltzmann_constant = 1.3806504e-23  # 定义玻尔兹曼常数，单位为焦耳每开尔文

# Boltzmann constant in eV/K                             8.617 343 e-5         0.000 015 e-5         eV K^-1
Boltzmann_constant_in_eV_per_K = 8.617343e-5  # 定义玻尔兹曼常数单位（电子伏特每开尔文）

# Boltzmann constant in Hz/K                             2.083 6644 e10        0.000 0036 e10        Hz K^-1
Boltzmann_constant_in_Hz_per_K = 2.0836644e10  # 定义玻尔兹曼常数单位（赫兹每开尔文）

# Boltzmann constant in inverse meters per kelvin        69.503 56             0.000 12              m^-1 K^-1
Boltzmann_constant_in_inverse_m_per_K = 69.50356  # 定义玻尔兹曼常数单位（每开尔文的倒数米）

# characteristic impedance of vacuum                     376.730 313 461...    (exact)               ohm
characteristic_impedance_of_vacuum = 376.730313461  # 定义真空特征阻抗，单位为欧姆

# classical electron radius                              2.817 940 2894 e-15   0.000 000 0058 e-15   m
classical_electron_radius = 2.8179402894e-15  # 定义经典电子半径，单位为米

# Compton wavelength                                     2.426 310 2175 e-12   0.000 000 0033 e-12   m
Compton_wavelength = 2.4263102175e-12  # 定义康普顿波长，单位为米

# Compton wavelength over 2 pi                           386.159 264 59 e-15   0.000 000 53 e-15     m
Compton_wavelength_over_2pi = 386.15926459e-15  # 定义康普顿波长除以2π，单位为米

# conductance quantum                                    7.748 091 7004 e-5    0.000 000 0053 e-5    S
conductance_quantum = 7.7480917004e-5  # 定义电导量子，单位为西门子

# conventional value of Josephson constant               483 597.9 e9          (exact)               Hz V^-1
conventional_Josephson_constant = 483597.9e9  # 定义约瑟夫森常数，单位为赫兹每伏特

# conventional value of von Klitzing constant            25 812.807            (exact)               ohm
conventional_von_Klitzing_constant = 25812.807  # 定义冯·克里青常数，单位为欧姆
# Cu x unit                                              1.002 076 99 e-13     0.000 000 28 e-13     m
Cu_x_unit = 1.00207699e-13  # 定义Cu x单位的数值，单位为米，不确定性为±0.000 000 28 e-13 米

# deuteron-electron mag. mom. ratio                      -4.664 345 537 e-4    0.000 000 039 e-4
deuteron_electron_mag_mom_ratio = -4.664345537e-4  # 定义氘-电子磁矩比的数值，不确定性为±0.000 000 039 e-4

# deuteron-electron mass ratio                           3670.482 9654         0.000 0016
deuteron_electron_mass_ratio = 3670.4829654  # 定义氘-电子质量比的数值，不确定性为±0.000 0016

# deuteron g factor                                      0.857 438 2308        0.000 000 0072
deuteron_g_factor = 0.8574382308  # 定义氘的g因子的数值，不确定性为±0.000 000 0072

# deuteron mag. mom.                                     0.433 073 465 e-26    0.000 000 011 e-26    J T^-1
deuteron_mag_mom = 0.433073465e-26  # 定义氘的磁矩的数值，单位为J T^-1，不确定性为±0.000 000 011 e-26 J T^-1

# deuteron mag. mom. to Bohr magneton ratio              0.466 975 4556 e-3    0.000 000 0039 e-3
deuteron_mag_mom_to_Bohr_magneton_ratio = 0.4669754556e-3  # 定义氘的磁矩对玻尔磁子比值的数值，不确定性为±0.000 000 0039 e-3

# deuteron mag. mom. to nuclear magneton ratio           0.857 438 2308        0.000 000 0072
deuteron_mag_mom_to_nuclear_magneton_ratio = 0.8574382308  # 定义氘的磁矩对核磁子比值的数值，不确定性为±0.000 000 0072

# deuteron mass                                          3.343 583 20 e-27     0.000 000 17 e-27     kg
deuteron_mass = 3.34358320e-27  # 定义氘的质量的数值，单位为千克，不确定性为±0.000 000 17 e-27 千克

# deuteron mass energy equivalent                        3.005 062 72 e-10     0.000 000 15 e-10     J
deuteron_mass_energy_equiv = 3.00506272e-10  # 定义氘的质能转换值的数值，单位为J，不确定性为±0.000 000 15 e-10 J

# deuteron mass energy equivalent in MeV                 1875.612 793          0.000 047             MeV
deuteron_mass_energy_equiv_MeV = 1875.612793  # 定义氘的质能转换值的数值，单位为MeV，不确定性为±0.000 047 MeV

# deuteron mass in u                                     2.013 553 212 724     0.000 000 000 078     u
deuteron_mass_in_u = 2.013553212724  # 定义氘的质量的数值，单位为原子质量单位u，不确定性为±0.000 000 000 078 u

# deuteron molar mass                                    2.013 553 212 724 e-3 0.000 000 000 078 e-3 kg mol^-1
deuteron_molar_mass = 2.013553212724e-3  # 定义氘的摩尔质量的数值，单位为kg mol^-1，不确定性为±0.000 000 000 078 e-3 kg mol^-1

# deuteron-neutron mag. mom. ratio                       -0.448 206 52         0.000 000 11
deuteron_neutron_mag_mom_ratio = -0.44820652  # 定义氘-中子磁矩比的数值，不确定性为±0.000 000 11

# deuteron-proton mag. mom. ratio                        0.307 012 2070        0.000 000 0024
deuteron_proton_mag_mom_ratio = 0.3070122070  # 定义氘-质子磁矩比的数值，不确定性为±0.000 000 0024

# deuteron-proton mass ratio                             1.999 007 501 08      0.000 000 000 22
deuteron_proton_mass_ratio = 1.99900750108  # 定义氘-质子质量比的数值，不确定性为±0.000 000 000 22

# deuteron rms charge radius                             2.1402 e-15           0.0028 e-15           m
deuteron_rms_charge_radius = 2.1402e-15  # 定义氘的均方根电荷半径的数值，单位为米，不确定性为±0.0028 e-15 米

# electric constant                                      8.854 187 817... e-12 (exact)               F m^-1
electric_constant = 8.854187817e-12  # 定义电常数的数值，单位为F m^-1，精确值

# electron charge to mass quotient                       -1.758 820 150 e11    0.000 000 044 e11     C kg^-1
electron_charge_to_mass_quotient = -1.758820150e11  # 定义电子电荷质量比的数值，单位为C kg^-1，不确定性为±0.000 000 044 e11 C kg^-1

# electron-deuteron mag. mom. ratio                      -2143.923 498         0.000 018
electron_deuteron_mag_mom_ratio = -2143.923498  # 定义电子-氘磁矩比的数值，不确定性为±0.000 018

# electron-deuteron mass ratio                           2.724 437 1093 e-4    0.000 000 0012 e-4
electron_deuteron_mass_ratio = 2.7244371093e-4  # 定义电子-氘质量比的数值，不确定性为±0.000 000 0012 e-4

# electron g factor                                      -2.002 319 304 3622   0.000 000 000 0015
electron_g_factor = -2.0023193043622  # 定义电子的g因子的数值，不确定性为±0.000 000 000 0015

# electron gyromag. ratio                                1.760 859 770 e11     0.000 000 044 e11     s^-1 T^-1
electron_gyromag_ratio = 1.760859770e11  # 定义电子的旋磁比的数值，单位为s^-1 T^-1，不确定性为±0.000 000 044 e11 s^-1 T^-1

# electron gyromag. ratio over 2 pi                      28 024.953 64         0.000 70              MHz T^-1
electron_gyromag_ratio_over_2pi = 28024.95364  # 定义电子的2π倍旋磁比的数值，单位为MHz T^-1，不确定性为±0.000 70 MHz T^-1

# electron mag. mom.                                     -928.476 377 e-26     0.000 023 e-26        J T^-1
electron_mag_mom = -928.476377e-26  # 定义电子的磁矩的数值，单位为J T^-1，不确定性为±0.000 023 e-26 J T^-
# 电子的质量，单位为原子质量单位（atomic mass unit，u）
electron mass in u                                     5.485 799 0943 e-4    0.000 000 0023 e-4    u
# 电子的摩尔质量，单位为千克每摩尔（kilograms per mole，kg mol^-1）
electron molar mass                                    5.485 799 0943 e-7    0.000 000 0023 e-7    kg mol^-1
# 电子和μ子的磁矩比
electron-muon mag. mom. ratio                          206.766 9877          0.000 0052
# 电子和μ子的质量比
electron-muon mass ratio                               4.836 331 71 e-3      0.000 000 12 e-3
# 电子和中子的磁矩比
electron-neutron mag. mom. ratio                       960.920 50            0.000 23
# 电子和中子的质量比
electron-neutron mass ratio                            5.438 673 4459 e-4    0.000 000 0033 e-4
# 电子和质子的磁矩比
electron-proton mag. mom. ratio                        -658.210 6848         0.000 0054
# 电子和质子的质量比
electron-proton mass ratio                             5.446 170 2177 e-4    0.000 000 0024 e-4
# 电子和τ子的质量比
electron-tau mass ratio                                2.875 64 e-4          0.000 47 e-4
# 电子和α粒子的质量比
electron to alpha particle mass ratio                  1.370 933 555 70 e-4  0.000 000 000 58 e-4
# 电子与屏蔽氦离子的磁矩比
electron to shielded helion mag. mom. ratio            864.058 257           0.000 010
# 电子与屏蔽质子的磁矩比
electron to shielded proton mag. mom. ratio            -658.227 5971         0.000 0072
# 电子伏特，单位焦耳（joules，J）
electron volt                                          1.602 176 487 e-19    0.000 000 040 e-19    J
# 电子伏特与原子质量单位之间的关系
electron volt-atomic mass unit relationship            1.073 544 188 e-9     0.000 000 027 e-9     u
# 电子伏特与哈特里之间的关系
electron volt-hartree relationship                     3.674 932 540 e-2     0.000 000 092 e-2     E_h
# 电子伏特与赫兹之间的关系
electron volt-hertz relationship                       2.417 989 454 e14     0.000 000 060 e14     Hz
# 电子伏特与逆米之间的关系
electron volt-inverse meter relationship               8.065 544 65 e5       0.000 000 20 e5       m^-1
# 电子伏特与焦耳之间的关系
electron volt-joule relationship                       1.602 176 487 e-19    0.000 000 040 e-19    J
# 电子伏特与开尔文之间的关系
electron volt-kelvin relationship                      1.160 4505 e4         0.000 0020 e4         K
# 电子伏特与千克之间的关系
electron volt-kilogram relationship                    1.782 661 758 e-36    0.000 000 044 e-36    kg
# 元电荷，单位库伦（coulombs，C）
elementary charge                                      1.602 176 487 e-19    0.000 000 040 e-19    C
# 元电荷与普朗克常数之比
elementary charge over h                               2.417 989 454 e14     0.000 000 060 e14     A J^-1
# 法拉第常数，单位库伦每摩尔（coulombs per mole，C mol^-1）
Faraday constant                                       96 485.3399           0.0024                C mol^-1
# 用于传统电流的法拉第常数，单位库伦（90克分子，C_90 mol^-1）
Faraday constant for conventional electric current     96 485.3401           0.0048                C_90 mol^-1
# 费米耦合常数，单位GeV^-2
Fermi coupling constant                                1.166 37 e-5          0.000 01 e-5          GeV^-2
# 精细结构常数
fine-structure constant                                7.297 352 5376 e-3    0.000 000 0050 e-3
# 第一辐射常数，单位瓦特每平方米（watts per square meter，W m^2）
first radiation constant                               3.741 771 18 e-16     0.000 000 19 e-16     W m^2
# 用于光谱辐射度的第一辐射常数，单位瓦特每平方米每立体角（watts per square meter per steradian，W m^2 sr^-1）
first radiation constant for spectral radiance         1.191 042 759 e-16    0.000 000 059 e-16    W m^2 sr^-1
# 哈特里与原子质量单位之间的关系
hartree-atomic mass unit relationship                  2.921 262 2986 e-8    0.000 000 0042 e-8    u
# 哈特里与电子伏特之间的关系
hartree-electron volt relationship                     27.211 383 86         0.000 000 68          eV
# Hartree energy                                         4.359 743 94 e-18     0.000 000 22 e-18     J
Hartree_energy = 4.35974394e-18  # 定义Hartree能量（Joules）
Hartree_energy_uncertainty = 0.00000022e-18  # Hartree能量的不确定度（Joules）

# Hartree energy in eV                                   27.211 383 86         0.000 000 68          eV
Hartree_energy_eV = 27.21138386  # Hartree能量对应的电子伏特能量（eV）
Hartree_energy_eV_uncertainty = 0.00000068  # 电子伏特能量的不确定度（eV）

# hartree-hertz relationship                             6.579 683 920 722 e15 0.000 000 000 044 e15 Hz
hartree_hertz_relationship = 6.579683920722e15  # Hartree与赫兹之间的关系（Hz）
hartree_hertz_relationship_uncertainty = 0.000000000044e15  # 关系的不确定度（Hz）

# hartree-inverse meter relationship                     2.194 746 313 705 e7  0.000 000 000 015 e7  m^-1
hartree_inverse_meter_relationship = 2.194746313705e7  # Hartree与逆米之间的关系（m^-1）
hartree_inverse_meter_relationship_uncertainty = 0.000000000015e7  # 关系的不确定度（m^-1）

# hartree-joule relationship                             4.359 743 94 e-18     0.000 000 22 e-18     J
hartree_joule_relationship = 4.35974394e-18  # Hartree与焦耳之间的关系（Joules）
hartree_joule_relationship_uncertainty = 0.00000022e-18  # 关系的不确定度（Joules）

# hartree-kelvin relationship                            3.157 7465 e5         0.000 0055 e5         K
hartree_kelvin_relationship = 3.1577465e5  # Hartree与开尔文之间的关系（Kelvin）
hartree_kelvin_relationship_uncertainty = 0.0000055e5  # 关系的不确定度（Kelvin）

# hartree-kilogram relationship                          4.850 869 34 e-35     0.000 000 24 e-35     kg
hartree_kilogram_relationship = 4.85086934e-35  # Hartree与千克之间的关系（kg）
hartree_kilogram_relationship_uncertainty = 0.00000024e-35  # 关系的不确定度（kg）

# helion-electron mass ratio                             5495.885 2765         0.000 0052
helion_electron_mass_ratio = 5495.8852765  # 氦离子与电子质量比
helion_electron_mass_ratio_uncertainty = 0.0000052  # 比率的不确定度

# helion mass                                            5.006 411 92 e-27     0.000 000 25 e-27     kg
helion_mass = 5.00641192e-27  # 氦离子质量（kg）
helion_mass_uncertainty = 0.00000025e-27  # 氦离子质量的不确定度（kg）

# helion mass energy equivalent                          4.499 538 64 e-10     0.000 000 22 e-10     J
helion_mass_energy_equivalent = 4.49953864e-10  # 氦离子质量能量等效值（Joules）
helion_mass_energy_equivalent_uncertainty = 0.00000022e-10  # 能量等效值的不确定度（Joules）

# helion mass energy equivalent in MeV                   2808.391 383          0.000 070             MeV
helion_mass_energy_equivalent_MeV = 2808.391383  # 氦离子质量能量等效值（兆电子伏特）
helion_mass_energy_equivalent_MeV_uncertainty = 0.000070  # 能量等效值的不确定度（MeV）

# helion mass in u                                       3.014 932 2473        0.000 000 0026        u
helion_mass_u = 3.0149322473  # 氦离子质量单位质量（原子质量单位）
helion_mass_u_uncertainty = 0.0000000026  # 单位质量的不确定度（原子质量单位）

# helion molar mass                                      3.014 932 2473 e-3    0.000 000 0026 e-3    kg mol^-1
helion_molar_mass = 3.0149322473e-3  # 氦离子摩尔质量（kg/mol）
helion_molar_mass_uncertainty = 0.0000000026e-3  # 摩尔质量的不确定度（kg/mol）

# helion-proton mass ratio                               2.993 152 6713        0.000 000 0026
helion_proton_mass_ratio = 2.9931526713  # 氦离子与质子质量比
helion_proton_mass_ratio_uncertainty = 0.0000000026  # 比率的不确定度

# hertz-atomic mass unit relationship                    4.439 821 6294 e-24   0.000 000 0064 e-24   u
hertz_atomic_mass_unit_relationship = 4.4398216294e-24  # 赫兹与原子质量单位之间的关系（原子质量单位）
hertz_atomic_mass_unit_relationship_uncertainty = 0.0000000064e-24  # 关系的不确定度（原子质量单位）

# hertz-electron volt relationship                       4.135 667 33 e-15     0.000 000 10 e-15     eV
hertz_electron_volt_relationship = 4.13566733e-15  # 赫兹与电子伏特之间的关系（eV）
hertz_electron_volt_relationship_uncertainty = 0.00000010e-15  # 关系的不确定度（eV）

# hertz-hartree relationship                             1.519 829 846 006 e-16 0.000 000 000010e-16 E_h
hertz_hartree_relationship = 1.519829846006e-16  # 赫兹与Hartree之间的关系（Hartree）
hertz_hartree_relationship_uncertainty = 0.000000000010e-16  # 关系的不确定度（Hartree）

# hertz-inverse meter relationship                       3.335 640 951... e-9  (exact)               m^-1
hertz_inverse_meter_relationship = 3.335640951e-9  # 赫兹与逆米之间的关系（m^-1）
# 该关系为精确值，没有不确定度

# hertz-joule relationship                               6.626 068 96 e-34     0.000 000 33 e-34     J
hertz_joule_relationship = 6.62606896e-34  # 赫兹与焦耳之间的关系（Joules）
hertz_joule_relationship_uncertainty = 0.00000033e-34  # 关系的不确定度（Joules）

# hertz-kelvin relationship                              4.799 2374 e-11       0.000 0084 e-11       K
hertz_kelvin_relationship = 4.7992374e-11  # 赫兹与开尔文之间的关系（Kelvin）
hertz_kelvin_relationship_uncertainty = 0.0000084e-11  # 关系的不确定度（Kelvin）

# hertz-kilogram relationship                            7.372 496 00 e-51     0.000 000 37 e-51     kg
hertz_kilogram_relationship = 7.37249600e-51  # 赫兹与千克之间的关系（kg）
hertz_kilogram_relationship_uncertainty = 0.00000037e-51  # 关系的不确定度（kg）

# inverse fine-structure
# inverse of conductance quantum                         12 906.403 7787       0.000 0088            ohm
# 电导量子的倒数，单位是欧姆
inverse_of_conductance_quantum = 12_906.403_7787

# Josephson constant                                     483 597.891 e9        0.012 e9              Hz V^-1
# 约瑟夫森常数，单位是赫兹每伏特
Josephson_constant = 483_597.891e9

# joule-atomic mass unit relationship                    6.700 536 41 e9       0.000 000 33 e9       u
# 能量与原子质量单位的关系，单位是原子质量单位
joule_atomic_mass_unit_relationship = 6.700_536_41e9

# joule-electron volt relationship                       6.241 509 65 e18      0.000 000 16 e18      eV
# 能量与电子伏特的关系，单位是电子伏特
joule_electron_volt_relationship = 6.241_509_65e18

# joule-hartree relationship                             2.293 712 69 e17      0.000 000 11 e17      E_h
# 能量与哈特里的关系，单位是哈特里
joule_hartree_relationship = 2.293_712_69e17

# joule-hertz relationship                               1.509 190 450 e33     0.000 000 075 e33     Hz
# 能量与赫兹的关系，单位是赫兹
joule_hertz_relationship = 1.509_190_450e33

# joule-inverse meter relationship                       5.034 117 47 e24      0.000 000 25 e24      m^-1
# 能量与米的倒数的关系，单位是米的倒数
joule_inverse_meter_relationship = 5.034_117_47e24

# joule-kelvin relationship                              7.242 963 e22         0.000 013 e22         K
# 能量与开尔文的关系，单位是开尔文
joule_kelvin_relationship = 7.242_963e22

# joule-kilogram relationship                            1.112 650 056... e-17 (exact)               kg
# 能量与千克的关系，单位是千克
joule_kilogram_relationship = 1.112_650_056e-17

# kelvin-atomic mass unit relationship                   9.251 098 e-14        0.000 016 e-14        u
# 开尔文与原子质量单位的关系，单位是原子质量单位
kelvin_atomic_mass_unit_relationship = 9.251_098e-14

# kelvin-electron volt relationship                      8.617 343 e-5         0.000 015 e-5         eV
# 开尔文与电子伏特的关系，单位是电子伏特
kelvin_electron_volt_relationship = 8.617_343e-5

# kelvin-hartree relationship                            3.166 8153 e-6        0.000 0055 e-6        E_h
# 开尔文与哈特里的关系，单位是哈特里
kelvin_hartree_relationship = 3.166_8153e-6

# kelvin-hertz relationship                              2.083 6644 e10        0.000 0036 e10        Hz
# 开尔文与赫兹的关系，单位是赫兹
kelvin_hertz_relationship = 2.083_6644e10

# kelvin-inverse meter relationship                      69.503 56             0.000 12              m^-1
# 开尔文与米的倒数的关系，单位是米的倒数
kelvin_inverse_meter_relationship = 69.503_56

# kelvin-joule relationship                              1.380 6504 e-23       0.000 0024 e-23       J
# 开尔文与能量的关系，单位是焦耳
kelvin_joule_relationship = 1.380_6504e-23

# kelvin-kilogram relationship                           1.536 1807 e-40       0.000 0027 e-40       kg
# 开尔文与千克的关系，单位是千克
kelvin_kilogram_relationship = 1.536_1807e-40

# kilogram-atomic mass unit relationship                 6.022 141 79 e26      0.000 000 30 e26      u
# 千克与原子质量单位的关系，单位是原子质量单位
kilogram_atomic_mass_unit_relationship = 6.022_141_79e26

# kilogram-electron volt relationship                    5.609 589 12 e35      0.000 000 14 e35      eV
# 千克与电子伏特的关系，单位是电子伏特
kilogram_electron_volt_relationship = 5.609_589_12e35

# kilogram-hartree relationship                          2.061 486 16 e34      0.000 000 10 e34      E_h
# 千克与哈特里的关系，单位是哈特里
kilogram_hartree_relationship = 2.061_486_16e34

# kilogram-hertz relationship                            1.356 392 733 e50     0.000 000 068 e50     Hz
# 千克与赫兹的关系，单位是赫兹
kilogram_hertz_relationship = 1.356_392_733e50

# kilogram-inverse meter relationship                    4.524 439 15 e41      0.000 000 23 e41      m^-1
# 千克与米的倒数的关系，单位是米的倒数
kilogram_inverse_meter_relationship = 4.524_439_15e41

# kilogram-joule relationship                            8.987 551 787... e16  (exact)               J
# 千克与能量的关系，单位是焦耳
kilogram_joule_relationship = 8.987_551_787e16

# kilogram-kelvin relationship                           6.509 651 e39         0.000 011 e39         K
# 千克与开尔文的关系，单位是开尔文
kilogram_kelvin_relationship = 6.509_651e39

# lattice parameter of silicon                           543.102 064 e-12      0.000 014 e-12        m
# 硅的晶格常数，单位是米
lattice_parameter_of_silicon = 543.102_064e-12

# Loschmidt constant (273.15 K, 101.325 kPa)             2.686 7774 e25        0.000 0047 e25        m^-3
# 洛希米特常数 (273.15 开尔文, 101.325 千帕)，单位是米的倒数
Loschmidt_constant = 2.686_7774e25

# mag. constant                                          12.566 370 614... e-7 (exact)               N A^-2
# 磁常数，单位是牛顿每安培的平方
magnetic_constant = 12.566_370_614e-7

# mag. flux quantum                                      2.067 833 667 e-15    0.000 000 052 e-15    Wb
# 磁通量子，单位是韦伯
magnetic_flux_quantum = 2.067_833_667e-15

# molar gas constant                                     8.314 472             0.000 015             J mol^-1 K^-1
# 摩尔气体常数，单位是焦耳每摩尔每开尔文
molar_gas_constant = 8.314_472

# molar mass constant                                    1 e-3                 (exact)               kg mol^-1
# 摩尔质量常数，单位是千克每摩尔
molar_mass_constant = 1e-3
# 碳-12的摩尔质量
molar mass of carbon-12                                12 e-3                (exact)               kg mol^-1

# 摩尔普朗克常数
molar Planck constant                                  3.990 312 6821 e-10   0.000 000 0057 e-10   J s mol^-1

# 摩尔普朗克常数乘以光速
molar Planck constant times c                          0.119 626 564 72      0.000 000 000 17      J m mol^-1

# 理想气体在标准条件下（273.15 K, 100 kPa）的摩尔体积
molar volume of ideal gas (273.15 K, 100 kPa)          22.710 981 e-3        0.000 040 e-3         m^3 mol^-1

# 理想气体在标准条件下（273.15 K, 101.325 kPa）的摩尔体积
molar volume of ideal gas (273.15 K, 101.325 kPa)      22.413 996 e-3        0.000 039 e-3         m^3 mol^-1

# 硅的摩尔体积
molar volume of silicon                                12.058 8349 e-6       0.000 0011 e-6        m^3 mol^-1

# Mo x 单位
Mo x unit                                              1.002 099 55 e-13     0.000 000 53 e-13     m

# 缪子的康普顿波长
muon Compton wavelength                                11.734 441 04 e-15    0.000 000 30 e-15     m

# 缪子的康普顿波长除以2π
muon Compton wavelength over 2 pi                      1.867 594 295 e-15    0.000 000 047 e-15    m

# 缪子与电子质量比
muon-electron mass ratio                               206.768 2823          0.000 0052

# 缪子的g因子
muon g factor                                          -2.002 331 8414       0.000 000 0012

# 缪子的磁矩
muon mag. mom.                                         -4.490 447 86 e-26    0.000 000 16 e-26     J T^-1

# 缪子的磁矩异常
muon mag. mom. anomaly                                 1.165 920 69 e-3      0.000 000 60 e-3

# 缪子的磁矩与玻尔磁子比
muon mag. mom. to Bohr magneton ratio                  -4.841 970 49 e-3     0.000 000 12 e-3

# 缪子的磁矩与核磁子比
muon mag. mom. to nuclear magneton ratio               -8.890 597 05         0.000 000 23

# 缪子的质量
muon mass                                              1.883 531 30 e-28     0.000 000 11 e-28     kg

# 缪子的质能等效
muon mass energy equivalent                            1.692 833 510 e-11    0.000 000 095 e-11    J

# 缪子的质能等效（单位为兆电子伏特）
muon mass energy equivalent in MeV                     105.658 3668          0.000 0038            MeV

# 缪子的质量单位
muon mass in u                                         0.113 428 9256        0.000 000 0029        u

# 缪子的摩尔质量
muon molar mass                                        0.113 428 9256 e-3    0.000 000 0029 e-3    kg mol^-1

# 缪子与中子质量比
muon-neutron mass ratio                                0.112 454 5167        0.000 000 0029

# 缪子与质子磁矩比
muon-proton mag. mom. ratio                            -3.183 345 137        0.000 000 085

# 缪子与质子质量比
muon-proton mass ratio                                 0.112 609 5261        0.000 000 0029

# 缪子与τ子质量比
muon-tau mass ratio                                    5.945 92 e-2          0.000 97 e-2

# 自然单位的作用量（普朗克常数）
natural unit of action                                 1.054 571 628 e-34    0.000 000 053 e-34    J s

# 自然单位的作用量（单位为电子伏特秒）
natural unit of action in eV s                         6.582 118 99 e-16     0.000 000 16 e-16     eV s

# 自然单位的能量
natural unit of energy                                 8.187 104 38 e-14     0.000 000 41 e-14     J

# 自然单位的能量（单位为兆电子伏特）
natural unit of energy in MeV                          0.510 998 910         0.000 000 013         MeV

# 自然单位的长度
natural unit of length                                 386.159 264 59 e-15   0.000 000 53 e-15     m
# 自然单位制下的质量单位（电子质量）
natural unit of mass                                   9.109 382 15 e-31     0.000 000 45 e-31     kg
# 自然单位制下的动量单位（质量 × 速度）
natural unit of momentum                               2.730 924 06 e-22     0.000 000 14 e-22     kg m s^-1
# 自然单位制下的动量单位（以 MeV/c 为单位）
natural unit of momentum in MeV/c                      0.510 998 910         0.000 000 013         MeV/c
# 自然单位制下的时间单位
natural unit of time                                   1.288 088 6570 e-21   0.000 000 0018 e-21   s
# 自然单位制下的速度单位（光速）
natural unit of velocity                               299 792 458           (exact)               m s^-1
# 中子康普顿波长
neutron Compton wavelength                             1.319 590 8951 e-15   0.000 000 0020 e-15   m
# 中子康普顿波长除以 2π
neutron Compton wavelength over 2 pi                   0.210 019 413 82 e-15 0.000 000 000 31 e-15 m
# 中子-电子磁矩比
neutron-electron mag. mom. ratio                       1.040 668 82 e-3      0.000 000 25 e-3
# 中子-电子质量比
neutron-electron mass ratio                            1838.683 6605         0.000 0011
# 中子的 g 因子
neutron g factor                                       -3.826 085 45         0.000 000 90
# 中子的旋磁比
neutron gyromag. ratio                                 1.832 471 85 e8       0.000 000 43 e8       s^-1 T^-1
# 中子的旋磁比除以 2π
neutron gyromag. ratio over 2 pi                       29.164 6954           0.000 0069            MHz T^-1
# 中子的磁矩
neutron mag. mom.                                      -0.966 236 41 e-26    0.000 000 23 e-26     J T^-1
# 中子磁矩与玻尔磁子磁矩比
neutron mag. mom. to Bohr magneton ratio               -1.041 875 63 e-3     0.000 000 25 e-3
# 中子磁矩与核磁子磁矩比
neutron mag. mom. to nuclear magneton ratio            -1.913 042 73         0.000 000 45
# 中子的质量
neutron mass                                           1.674 927 211 e-27    0.000 000 084 e-27    kg
# 中子的质能关系
neutron mass energy equivalent                         1.505 349 505 e-10    0.000 000 075 e-10    J
# 中子的质能关系（以 MeV 为单位）
neutron mass energy equivalent in MeV                  939.565 346           0.000 023             MeV
# 中子的质量（以原子单位为单位）
neutron mass in u                                      1.008 664 915 97      0.000 000 000 43      u
# 中子的摩尔质量
neutron molar mass                                     1.008 664 915 97 e-3  0.000 000 000 43 e-3  kg mol^-1
# 中子-μ子质量比
neutron-muon mass ratio                                8.892 484 09          0.000 000 23
# 中子-质子磁矩比
neutron-proton mag. mom. ratio                         -0.684 979 34         0.000 000 16
# 中子-质子质量比
neutron-proton mass ratio                              1.001 378 419 18      0.000 000 000 46
# 中子-τ子质量比
neutron-tau mass ratio                                 0.528 740             0.000 086
# 中子到屏蔽质子磁矩比
neutron to shielded proton mag. mom. ratio             -0.684 996 94         0.000 000 16
# 牛顿引力常数
Newtonian constant of gravitation                      6.674 28 e-11         0.000 67 e-11         m^3 kg^-1 s^-2
# 牛顿引力常数除以约化 Planck 常数乘以光速的幂次
Newtonian constant of gravitation over h-bar c         6.708 81 e-39         0.000 67 e-39         (GeV/c^2)^-2
# 核磁子
nuclear magneton                                       5.050 783 24 e-27     0.000 000 13 e-27     J T^-1
# 核磁子（以电子伏特/特斯拉为单位）
nuclear magneton in eV/T                               3.152 451 2326 e-8    0.000 000 0045 e-8    eV T^-1
# 核磁子每特斯拉的逆米数
nuclear magneton in inverse meters per tesla           2.542 623 616 e-2     0.000 000 064 e-2     m^-1 T^-1

# 核磁子每开尔文特斯拉
nuclear magneton in K/T                                3.658 2637 e-4        0.000 0064 e-4        K T^-1

# 核磁子每兆赫特斯拉
nuclear magneton in MHz/T                              7.622 593 84          0.000 000 19          MHz T^-1

# 普朗克常数
Planck constant                                        6.626 068 96 e-34     0.000 000 33 e-34     J s

# 以电子伏秒为单位的普朗克常数
Planck constant in eV s                                4.135 667 33 e-15     0.000 000 10 e-15     eV s

# 2π除以普朗克常数
Planck constant over 2 pi                              1.054 571 628 e-34    0.000 000 053 e-34    J s

# 以电子伏秒为单位的2π除以普朗克常数
Planck constant over 2 pi in eV s                      6.582 118 99 e-16     0.000 000 16 e-16     eV s

# 以兆电子伏特费米为单位的2π除以普朗克常数乘以光速
Planck constant over 2 pi times c in MeV fm            197.326 9631          0.000 0049            MeV fm

# 普朗克长度
Planck length                                          1.616 252 e-35        0.000 081 e-35        m

# 普朗克质量
Planck mass                                            2.176 44 e-8          0.000 11 e-8          kg

# 以吉电子伏特为单位的普朗克质量能量等效值
Planck mass energy equivalent in GeV                   1.220 892 e19         0.000 061 e19         GeV

# 普朗克温度
Planck temperature                                     1.416 785 e32         0.000 071 e32         K

# 普朗克时间
Planck time                                            5.391 24 e-44         0.000 27 e-44         s

# 质子电荷质量比
proton charge to mass quotient                         9.578 833 92 e7       0.000 000 24 e7       C kg^-1

# 质子康普顿波长
proton Compton wavelength                              1.321 409 8446 e-15   0.000 000 0019 e-15   m

# 2π除以质子康普顿波长
proton Compton wavelength over 2 pi                    0.210 308 908 61 e-15 0.000 000 000 30 e-15 m

# 质子电子质量比
proton-electron mass ratio                             1836.152 672 47       0.000 000 80

# 质子g因子
proton g factor                                        5.585 694 713         0.000 000 046

# 质子旋磁比
proton gyromag. ratio                                  2.675 222 099 e8      0.000 000 070 e8      s^-1 T^-1

# 2π除以质子旋磁比
proton gyromag. ratio over 2 pi                        42.577 4821           0.000 0011            MHz T^-1

# 质子磁矩
proton mag. mom.                                       1.410 606 662 e-26    0.000 000 037 e-26    J T^-1

# 质子磁矩对玻尔磁子比
proton mag. mom. to Bohr magneton ratio                1.521 032 209 e-3     0.000 000 012 e-3

# 质子磁矩对核磁子比
proton mag. mom. to nuclear magneton ratio             2.792 847 356         0.000 000 023

# 质子磁矩屏蔽修正
proton mag. shielding correction                       25.694 e-6            0.014 e-6

# 质子质量
proton mass                                            1.672 621 637 e-27    0.000 000 083 e-27    kg

# 质子质量能量等效值
proton mass energy equivalent                          1.503 277 359 e-10    0.000 000 075 e-10    J

# 以兆电子伏特为单位的质子质量能量等效值
proton mass energy equivalent in MeV                   938.272 013           0.000 023             MeV

# 质子的质量数
proton mass in u                                       1.007 276 466 77      0.000 000 000 10      u

# 质子的摩尔质量
proton molar mass                                      1.007 276 466 77 e-3  0.000 000 000 10 e-3  kg mol^-1
# 质子与μ子质量比
proton-muon mass ratio                                 8.880 243 39          0.000 000 23

# 质子与中子磁矩比
proton-neutron mag. mom. ratio                         -1.459 898 06         0.000 000 34

# 质子与中子质量比
proton-neutron mass ratio                              0.998 623 478 24      0.000 000 000 46

# 质子的有效电荷半径的均方根
proton rms charge radius                               0.8768 e-15           0.0069 e-15           m

# 质子与τ子质量比
proton-tau mass ratio                                  0.528 012             0.000 086

# 循环量子
quantum of circulation                                 3.636 947 5199 e-4    0.000 000 0050 e-4    m^2 s^-1

# 量子循环的两倍
quantum of circulation times 2                         7.273 895 040 e-4     0.000 000 010 e-4     m^2 s^-1

# 瑞利常数（基态氢原子的光谱学常数）
Rydberg constant                                       10 973 731.568 527    0.000 073             m^-1

# 瑞利常数乘以光速（以赫兹为单位）
Rydberg constant times c in Hz                         3.289 841 960 361 e15 0.000 000 000 022 e15 Hz

# 瑞利常数乘以普朗克常数和光速（以电子伏特为单位）
Rydberg constant times hc in eV                        13.605 691 93         0.000 000 34          eV

# 瑞利常数乘以普朗克常数和光速（以焦耳为单位）
Rydberg constant times hc in J                         2.179 871 97 e-18     0.000 000 11 e-18     J

# Sackur-Tetrode常数（1K时，100千帕）
Sackur-Tetrode constant (1 K, 100 kPa)                 -1.151 7047           0.000 0044

# Sackur-Tetrode常数（1K时，101.325千帕）
Sackur-Tetrode constant (1 K, 101.325 kPa)             -1.164 8677           0.000 0044

# 第二辐射常数（光谱辐射定律中的常数）
second radiation constant                              1.438 7752 e-2        0.000 0025 e-2        m K

# 有屏蔽的氦核陀螺磁比率
shielded helion gyromag. ratio                         2.037 894 730 e8      0.000 000 056 e8      s^-1 T^-1

# 有屏蔽的氦核陀螺磁比率除以2π
shielded helion gyromag. ratio over 2 pi               32.434 101 98         0.000 000 90          MHz T^-1

# 有屏蔽的氦核磁矩（对玻尔磁子的比率）
shielded helion mag. mom.                              -1.074 552 982 e-26   0.000 000 030 e-26    J T^-1

# 有屏蔽的氦核磁矩对玻尔磁子的比率
shielded helion mag. mom. to Bohr magneton ratio       -1.158 671 471 e-3    0.000 000 014 e-3

# 有屏蔽的氦核磁矩对核磁子的比率
shielded helion mag. mom. to nuclear magneton ratio    -2.127 497 718        0.000 000 025

# 有屏蔽的氦核对质子磁矩比率
shielded helion to proton mag. mom. ratio              -0.761 766 558        0.000 000 011

# 有屏蔽的氦核对有屏蔽的质子磁矩比率
shielded helion to shielded proton mag. mom. ratio     -0.761 786 1313       0.000 000 0033

# 有屏蔽的质子陀螺磁比率
shielded proton gyromag. ratio                         2.675 153 362 e8      0.000 000 073 e8      s^-1 T^-1

# 有屏蔽的质子陀螺磁比率除以2π
shielded proton gyromag. ratio over 2 pi               42.576 3881           0.000 0012            MHz T^-1

# 有屏蔽的质子磁矩（对玻尔磁子的比率）
shielded proton mag. mom.                              1.410 570 419 e-26    0.000 000 038 e-26    J T^-1

# 有屏蔽的质子磁矩对玻尔磁子的比率
shielded proton mag. mom. to Bohr magneton ratio       1.520 993 128 e-3     0.000 000 017 e-3

# 有屏蔽的质子磁矩对核磁子的比率
shielded proton mag. mom. to nuclear magneton ratio    2.792 775 598         0.000 000 030

# 真空中的光速
speed of light in vacuum                               299 792 458           (exact)               m s^-1

# 标准重力加速度
standard acceleration of gravity                       9.806 65              (exact)               m s^-2

# 标准大气压
standard atmosphere                                    101 325               (exact)               Pa

# Stefan-Boltzmann常数（热辐射定律中的常数）
Stefan-Boltzmann constant                              5.670 400 e-8         0.000 040 e-8         W m^-2 K^-4
# tau Compton wavelength                                 0.697 72 e-15         0.000 11 e-15         m
tau_compton_wavelength = 0.69772e-15                     # tau康普顿波长，单位为米
# tau Compton wavelength over 2 pi                       0.111 046 e-15        0.000 018 e-15        m
tau_compton_wavelength_over_2pi = 0.111046e-15           # tau康普顿波长除以2π，单位为米
# tau-electron mass ratio                                3477.48               0.57
tau_electron_mass_ratio = 3477.48                        # tau电子质量比
# tau mass                                               3.167 77 e-27         0.000 52 e-27         kg
tau_mass = 3.16777e-27                                   # tau质量，单位为千克
# tau mass energy equivalent                             2.847 05 e-10         0.000 46 e-10         J
tau_mass_energy_equivalent = 2.84705e-10                  # tau质量能量等效，单位为焦耳
# tau mass energy equivalent in MeV                      1776.99               0.29                  MeV
tau_mass_energy_equivalent_mev = 1776.99                  # tau质量能量等效，单位为兆电子伏特
# tau mass in u                                          1.907 68              0.000 31              u
tau_mass_in_u = 1.90768                                  # tau质量，单位为原子质量单位
# tau molar mass                                         1.907 68 e-3          0.000 31 e-3          kg mol^-1
tau_molar_mass = 1.90768e-3                              # tau摩尔质量，单位为千克每摩尔
# tau-muon mass ratio                                    16.8183               0.0027
tau_muon_mass_ratio = 16.8183                             # tau-μ子质量比
# tau-neutron mass ratio                                 1.891 29              0.000 31
tau_neutron_mass_ratio = 1.89129                          # tau-中子质量比
# tau-proton mass ratio                                  1.893 90              0.000 31
tau_proton_mass_ratio = 1.89390                           # tau-质子质量比
# Thomson cross section                                  0.665 245 8558 e-28   0.000 000 0027 e-28   m^2
thomson_cross_section = 0.6652458558e-28                  # 汤姆孙散射截面，单位为平方米
# triton-electron mag. mom. ratio                        -1.620 514 423 e-3    0.000 000 021 e-3
triton_electron_mag_mom_ratio = -1.620514423e-3          # 氚-电子磁矩比
# triton-electron mass ratio                             5496.921 5269         0.000 0051
triton_electron_mass_ratio = 5496.9215269                # 氚-电子质量比
# triton g factor                                        5.957 924 896         0.000 000 076
triton_g_factor = 5.957924896                            # 氚的g因子
# triton mag. mom.                                       1.504 609 361 e-26    0.000 000 042 e-26    J T^-1
triton_mag_mom = 1.504609361e-26                         # 氚的磁矩，单位为焦耳每特斯拉
# triton mag. mom. to Bohr magneton ratio                1.622 393 657 e-3     0.000 000 021 e-3
triton_mag_mom_to_bohr_magneton_ratio = 1.622393657e-3   # 氚的磁矩与玻尔磁子比值
# triton mag. mom. to nuclear magneton ratio             2.978 962 448         0.000 000 038
triton_mag_mom_to_nuclear_magneton_ratio = 2.978962448   # 氚的磁矩与核磁子比值
# triton mass                                            5.007 355 88 e-27     0.000 000 25 e-27     kg
triton_mass = 5.00735588e-27                             # 氚的质量，单位为千克
# triton mass energy equivalent                          4.500 387 03 e-10     0.000 000 22 e-10     J
triton_mass_energy_equivalent = 4.50038703e-10            # 氚的质量能量等效，单位为焦耳
# triton mass energy equivalent in MeV                   2808.920 906          0.000 070             MeV
triton_mass_energy_equivalent_mev = 2808.920906           # 氚的质量能量等效，单位为兆电子伏特
# triton mass in u                                       3.015 500 7134        0.000 000 0025        u
triton_mass_in_u = 3.0155007134                          # 氚的质量，单位为原子质量单位
# triton molar mass                                      3.015 500 7134 e-3    0.000 000 0025 e-3    kg mol^-1
triton_molar_mass = 3.0155007134e-3                      # 氚的摩尔质量，单位为千克每摩尔
# triton-neutron mag. mom. ratio                         -1.557 185 53         0.000 000 37
triton_neutron_mag_mom_ratio = -1.55718553               # 氚-中子磁矩比
# triton-proton mag. mom. ratio                          1.066 639 908         0.000 000 010
triton_proton_mag_mom_ratio = 1.066639908                # 氚-质子磁矩比
# triton-proton mass ratio                               2.993 717 0309        0.000 000 0025
triton_proton_mass_ratio = 2.9937170309                  # 氚-质子质量比
# unified atomic mass unit                               1.660 538 782 e-27    0.000 000 083 e-27    kg
unified_atomic_mass_unit = 1.660538782e-27               # 统一原子质量单位，单位为千克
# von Klitzing constant                                  25 812.807 557        0.000 018             ohm
von_klitzing_constant = 25812.807557                     # 冯·克利青常数，单位为欧姆
# weak mixing angle                                      0.222 55              0.000 56
weak_mixing_angle = 0.22255                              # 弱相互作用角
# Wien frequency displacement law constant               5.878 933 e10         0.000 010 e10         Hz K^-1
wien_frequency_displacement_law_constant = 5.878933e10   # 维恩频率位移定律常数，单位为赫兹每开尔文
# 定义 Wien 波长位移定律常数，单位为米·开尔文
Wien wavelength displacement law constant              2.897 7685 e-3        0.000 0051 e-3        m K"""

# 定义包含多个物理常数的文本块，每行包含一个常数及其不确定性
txt2010 = """\
# 220 晶格间距离硅的数值，单位为米
{220} lattice spacing of silicon                       192.015 5714 e-12     0.000 0032 e-12       m
# α粒子和电子质量比的数值
alpha particle-electron mass ratio                     7294.299 5361         0.000 0029
# α粒子的质量，单位为千克
alpha particle mass                                    6.644 656 75 e-27     0.000 000 29 e-27     kg
# α粒子的质能转换，单位为焦耳
alpha particle mass energy equivalent                  5.971 919 67 e-10     0.000 000 26 e-10     J
# α粒子的质能转换，单位为兆电子伏特
alpha particle mass energy equivalent in MeV           3727.379 240          0.000 082             MeV
# α粒子的质量，单位为原子质量单位（Dalton）
alpha particle mass in u                               4.001 506 179 125     0.000 000 000 062     u
# α粒子的摩尔质量，单位为千克每摩尔
alpha particle molar mass                              4.001 506 179 125 e-3 0.000 000 000 062 e-3 kg mol^-1
# α粒子和质子质量比的数值
alpha particle-proton mass ratio                       3.972 599 689 33      0.000 000 000 36
# Å星常数，即埃曼常数，单位为米
Angstrom star                                          1.000 014 95 e-10     0.000 000 90 e-10     m
# 原子质量常数，单位为千克
atomic mass constant                                   1.660 538 921 e-27    0.000 000 073 e-27    kg
# 原子质量常数的质能转换，单位为焦耳
atomic mass constant energy equivalent                 1.492 417 954 e-10    0.000 000 066 e-10    J
# 原子质量常数的质能转换，单位为兆电子伏特
atomic mass constant energy equivalent in MeV          931.494 061           0.000 021             MeV
# 原子质量单位与电子伏特关系，单位为电子伏特
atomic mass unit-electron volt relationship            931.494 061 e6        0.000 021 e6          eV
# 原子质量单位与哈特里关系，单位为哈特里
atomic mass unit-hartree relationship                  3.423 177 6845 e7     0.000 000 0024 e7     E_h
# 原子质量单位与赫兹关系，单位为赫兹
atomic mass unit-hertz relationship                    2.252 342 7168 e23    0.000 000 0016 e23    Hz
# 原子质量单位与米的倒数关系，单位为米的倒数
atomic mass unit-inverse meter relationship            7.513 006 6042 e14    0.000 000 0053 e14    m^-1
# 原子质量单位与焦耳关系，单位为焦耳
atomic mass unit-joule relationship                    1.492 417 954 e-10    0.000 000 066 e-10    J
# 原子质量单位与开尔文关系，单位为开尔文
atomic mass unit-kelvin relationship                   1.080 954 08 e13      0.000 000 98 e13      K
# 原子质量单位与千克关系，单位为千克
atomic mass unit-kilogram relationship                 1.660 538 921 e-27    0.000 000 073 e-27    kg
# 一级超极化率的原子单位，单位为库仑立方米每焦耳平方
atomic unit of 1st hyperpolarizability                 3.206 361 449 e-53    0.000 000 071 e-53    C^3 m^3 J^-2
# 二级超极化率的原子单位，单位为库仑的四次方米每焦耳立方米
atomic unit of 2nd hyperpolarizability                 6.235 380 54 e-65     0.000 000 28 e-65     C^4 m^4 J^-3
# 动作的原子单位，单位为焦耳秒
atomic unit of action                                  1.054 571 726 e-34    0.000 000 047 e-34    J s
# 电荷的原子单位，单位为库仑
atomic unit of charge                                  1.602 176 565 e-19    0.000 000 035 e-19    C
# 电荷密度的原子单位，单位为库仑每立方米
atomic unit of charge density                          1.081 202 338 e12     0.000 000 024 e12     C m^-3
# 电流的原子单位，单位为安培
atomic unit of current                                 6.623 617 95 e-3      0.000 000 15 e-3      A
# 电偶极矩的原子单位，单位为库仑米
atomic unit of electric dipole mom.                    8.478 353 26 e-30     0.000 000 19 e-30     C m
# 电场的原子单位，单位为伏特每米
atomic unit of electric field                          5.142 206 52 e11      0.000 000 11 e11      V m^-1
# 电场梯度的原子单位，单位为伏特每平方米
atomic unit of electric field gradient                 9.717 362 00 e21      0.000 000 21 e21      V m^-2
# atomic unit of electric polarizability
1.648 777 2754 e-41   0.000 000 0016 e-41   C^2 m^2 J^-1
# atomic unit of electric potential
27.211 385 05         0.000 000 60          V
# atomic unit of electric quadrupole mom.
4.486 551 331 e-40    0.000 000 099 e-40    C m^2
# atomic unit of energy
4.359 744 34 e-18     0.000 000 19 e-18     J
# atomic unit of force
8.238 722 78 e-8      0.000 000 36 e-8      N
# atomic unit of length
0.529 177 210 92 e-10 0.000 000 000 17 e-10 m
# atomic unit of mag. dipole mom.
1.854 801 936 e-23    0.000 000 041 e-23    J T^-1
# atomic unit of mag. flux density
2.350 517 464 e5      0.000 000 052 e5      T
# atomic unit of magnetizability
7.891 036 607 e-29    0.000 000 013 e-29    J T^-2
# atomic unit of mass
9.109 382 91 e-31     0.000 000 40 e-31     kg
# atomic unit of mom.um
1.992 851 740 e-24    0.000 000 088 e-24    kg m s^-1
# atomic unit of permittivity
1.112 650 056... e-10 (exact)               F m^-1
# atomic unit of time
2.418 884 326 502e-17 0.000 000 000 012e-17 s
# atomic unit of velocity
2.187 691 263 79 e6   0.000 000 000 71 e6   m s^-1
# Avogadro constant
6.022 141 29 e23      0.000 000 27 e23      mol^-1
# Bohr magneton
927.400 968 e-26      0.000 020 e-26        J T^-1
# Bohr magneton in eV/T
5.788 381 8066 e-5    0.000 000 0038 e-5    eV T^-1
# Bohr magneton in Hz/T
13.996 245 55 e9      0.000 000 31 e9       Hz T^-1
# Bohr magneton in inverse meters per tesla
46.686 4498           0.000 0010            m^-1 T^-1
# Bohr magneton in K/T
0.671 713 88          0.000 000 61          K T^-1
# Bohr radius
0.529 177 210 92 e-10 0.000 000 000 17 e-10 m
# Boltzmann constant
1.380 6488 e-23       0.000 0013 e-23       J K^-1
# Boltzmann constant in eV/K
8.617 3324 e-5        0.000 0078 e-5        eV K^-1
# Boltzmann constant in Hz/K
2.083 6618 e10        0.000 0019 e10        Hz K^-1
# Boltzmann constant in inverse meters per kelvin
69.503 476            0.000 063             m^-1 K^-1
# characteristic impedance of vacuum
376.730 313 461...    (exact)               ohm
# classical electron radius
2.817 940 3267 e-15   0.000 000 0027 e-15   m
# Compton wavelength
2.426 310 2389 e-12   0.000 000 0016 e-12   m
# Compton wavelength over 2 pi
# 386.159 268 00 e-15   0.000 000 25 e-15     m

# conductance quantum
# 7.748 091 7346 e-5    0.000 000 0025 e-5    S

# conventional value of Josephson constant
# 483 597.9 e9          (exact)               Hz V^-1

# conventional value of von Klitzing constant
# 25 812.807            (exact)               ohm

# Cu x unit
# 1.002 076 97 e-13     0.000 000 28 e-13     m

# deuteron-electron mag. mom. ratio
# -4.664 345 537 e-4    0.000 000 039 e-4

# deuteron-electron mass ratio
# 3670.482 9652         0.000 0015

# deuteron g factor
# 0.857 438 2308        0.000 000 0072

# deuteron mag. mom.
# 0.433 073 489 e-26    0.000 000 010 e-26    J T^-1

# deuteron mag. mom. to Bohr magneton ratio
# 0.466 975 4556 e-3    0.000 000 0039 e-3

# deuteron mag. mom. to nuclear magneton ratio
# 0.857 438 2308        0.000 000 0072

# deuteron mass
# 3.343 583 48 e-27     0.000 000 15 e-27     kg

# deuteron mass energy equivalent
# 3.005 062 97 e-10     0.000 000 13 e-10     J

# deuteron mass energy equivalent in MeV
# 1875.612 859          0.000 041             MeV

# deuteron mass in u
# 2.013 553 212 712     0.000 000 000 077     u

# deuteron molar mass
# 2.013 553 212 712 e-3 0.000 000 000 077 e-3 kg mol^-1

# deuteron-neutron mag. mom. ratio
# -0.448 206 52         0.000 000 11

# deuteron-proton mag. mom. ratio
# 0.307 012 2070        0.000 000 0024

# deuteron-proton mass ratio
# 1.999 007 500 97      0.000 000 000 18

# deuteron rms charge radius
# 2.1424 e-15           0.0021 e-15           m

# electric constant
# 8.854 187 817... e-12 (exact)               F m^-1

# electron charge to mass quotient
# -1.758 820 088 e11    0.000 000 039 e11     C kg^-1

# electron-deuteron mag. mom. ratio
# -2143.923 498         0.000 018

# electron-deuteron mass ratio
# 2.724 437 1095 e-4    0.000 000 0011 e-4

# electron g factor
# -2.002 319 304 361 53 0.000 000 000 000 53

# electron gyromag. ratio
# 1.760 859 708 e11     0.000 000 039 e11     s^-1 T^-1

# electron gyromag. ratio over 2 pi
# 28 024.952 66         0.000 62              MHz T^-1

# electron-helion mass ratio
# 1.819 543 0761 e-4    0.000 000 0017 e-4

# electron mag. mom.
# -928.476 430 e-26     0.000 021 e-26        J T^-1

# electron mag. mom. anomaly
# 1.159 652 180 76 e-3  0.000 000 000 27 e-3
# 电子磁矩到玻尔磁子比
electron mag. mom. to Bohr magneton ratio              -1.001 159 652 180 76 0.000 000 000 000 27

# 电子磁矩到核磁子比
electron mag. mom. to nuclear magneton ratio           -1838.281 970 90      0.000 000 75

# 电子质量
electron mass                                          9.109 382 91 e-31     0.000 000 40 e-31     kg

# 电子质量能量等效
electron mass energy equivalent                        8.187 105 06 e-14     0.000 000 36 e-14     J

# 电子质量能量等效，以兆电子伏计
electron mass energy equivalent in MeV                 0.510 998 928         0.000 000 011         MeV

# 电子质量，以原子质量单位（原子质量单位 u 是 1/12 碳-12原子的质量）
electron mass in u                                     5.485 799 0946 e-4    0.000 000 0022 e-4    u

# 电子摩尔质量
electron molar mass                                    5.485 799 0946 e-7    0.000 000 0022 e-7    kg mol^-1

# 电子-μ子磁矩比
electron-muon mag. mom. ratio                          206.766 9896          0.000 0052

# 电子-μ子质量比
electron-muon mass ratio                               4.836 331 66 e-3      0.000 000 12 e-3

# 电子-中子磁矩比
electron-neutron mag. mom. ratio                       960.920 50            0.000 23

# 电子-中子质量比
electron-neutron mass ratio                            5.438 673 4461 e-4    0.000 000 0032 e-4

# 电子-质子磁矩比
electron-proton mag. mom. ratio                        -658.210 6848         0.000 0054

# 电子-质子质量比
electron-proton mass ratio                             5.446 170 2178 e-4    0.000 000 0022 e-4

# 电子-τ质量比
electron-tau mass ratio                                2.875 92 e-4          0.000 26 e-4

# 电子到α粒子质量比
electron to alpha particle mass ratio                  1.370 933 555 78 e-4  0.000 000 000 55 e-4

# 电子到屏蔽氦核磁矩比
electron to shielded helion mag. mom. ratio            864.058 257           0.000 010

# 电子到屏蔽质子磁矩比
electron to shielded proton mag. mom. ratio            -658.227 5971         0.000 0072

# 电子-氚质量比
electron-triton mass ratio                             1.819 200 0653 e-4    0.000 000 0017 e-4

# 电子伏特（电子静电单位）
electron volt                                          1.602 176 565 e-19    0.000 000 035 e-19    J

# 电子伏特到原子质量单位关系
electron volt-atomic mass unit relationship            1.073 544 150 e-9     0.000 000 024 e-9     u

# 电子伏特到哈特里关系
electron volt-hartree relationship                     3.674 932 379 e-2     0.000 000 081 e-2     E_h

# 电子伏特到赫兹关系
electron volt-hertz relationship                       2.417 989 348 e14     0.000 000 053 e14     Hz

# 电子伏特到逆米关系
electron volt-inverse meter relationship               8.065 544 29 e5       0.000 000 18 e5       m^-1

# 电子伏特到焦耳关系
electron volt-joule relationship                       1.602 176 565 e-19    0.000 000 035 e-19    J

# 电子伏特到开尔文关系
electron volt-kelvin relationship                      1.160 4519 e4         0.000 0011 e4         K

# 电子伏特到千克关系
electron volt-kilogram relationship                    1.782 661 845 e-36    0.000 000 039 e-36    kg

# 元电荷
elementary charge                                      1.602 176 565 e-19    0.000 000 035 e-19    C

# 元电荷与普朗克常数之比
elementary charge over h                               2.417 989 348 e14     0.000 000 053 e14     A J^-1

# 法拉第常数（化学中电量的量纲）
Faraday constant                                       96 485.3365           0.0021                C mol^-1

# 传统电流单位下的法拉第常数
Faraday constant for conventional electric current     96 485.3321           0.0043                C_90 mol^-1
# Fermi coupling constant                                1.166 364 e-5         0.000 005 e-5         GeV^-2
# 费米耦合常数，以GeV^-2为单位

# fine-structure constant                                7.297 352 5698 e-3    0.000 000 0024 e-3
# 精细结构常数

# first radiation constant                               3.741 771 53 e-16     0.000 000 17 e-16     W m^2
# 第一辐射常数，单位为W m^2

# first radiation constant for spectral radiance         1.191 042 869 e-16    0.000 000 053 e-16    W m^2 sr^-1
# 用于光谱辐射亮度的第一辐射常数，单位为W m^2 sr^-1

# hartree-atomic mass unit relationship                  2.921 262 3246 e-8    0.000 000 0021 e-8    u
# 哈特里-原子质量单位关系，单位为原子质量单位(u)

# hartree-electron volt relationship                     27.211 385 05         0.000 000 60          eV
# 哈特里-电子伏特关系，单位为电子伏特(eV)

# Hartree energy                                         4.359 744 34 e-18     0.000 000 19 e-18     J
# 哈特里能量，单位为焦耳(J)

# Hartree energy in eV                                   27.211 385 05         0.000 000 60          eV
# 哈特里能量，单位为电子伏特(eV)

# hartree-hertz relationship                             6.579 683 920 729 e15 0.000 000 000 033 e15 Hz
# 哈特里-赫兹关系，单位为赫兹(Hz)

# hartree-inverse meter relationship                     2.194 746 313 708 e7  0.000 000 000 011 e7  m^-1
# 哈特里-米的倒数关系，单位为米的倒数(m^-1)

# hartree-joule relationship                             4.359 744 34 e-18     0.000 000 19 e-18     J
# 哈特里-焦耳关系，单位为焦耳(J)

# hartree-kelvin relationship                            3.157 7504 e5         0.000 0029 e5         K
# 哈特里-开尔文关系，单位为开尔文(K)

# hartree-kilogram relationship                          4.850 869 79 e-35     0.000 000 21 e-35     kg
# 哈特里-千克关系，单位为千克(kg)

# helion-electron mass ratio                             5495.885 2754         0.000 0050
# 氚离子-电子质量比

# helion g factor                                        -4.255 250 613        0.000 000 050
# 氚离子g因子

# helion mag. mom.                                       -1.074 617 486 e-26   0.000 000 027 e-26    J T^-1
# 氚离子磁矩，单位为焦耳每特斯拉(J T^-1)

# helion mag. mom. to Bohr magneton ratio                -1.158 740 958 e-3    0.000 000 014 e-3
# 氚离子磁矩与玻尔磁子比值

# helion mag. mom. to nuclear magneton ratio             -2.127 625 306        0.000 000 025
# 氚离子磁矩与核磁子比值

# helion mass                                            5.006 412 34 e-27     0.000 000 22 e-27     kg
# 氚离子质量，单位为千克(kg)

# helion mass energy equivalent                          4.499 539 02 e-10     0.000 000 20 e-10     J
# 氚离子质能等效，单位为焦耳(J)

# helion mass energy equivalent in MeV                   2808.391 482          0.000 062             MeV
# 氚离子质能等效，单位为兆电子伏特(MeV)

# helion mass in u                                       3.014 932 2468        0.000 000 0025        u
# 氚离子质量，单位为原子质量单位(u)

# helion molar mass                                      3.014 932 2468 e-3    0.000 000 0025 e-3    kg mol^-1
# 氚离子摩尔质量，单位为千克每摩尔(kg mol^-1)

# helion-proton mass ratio                               2.993 152 6707        0.000 000 0025
# 氚离子-质子质量比

# hertz-atomic mass unit relationship                    4.439 821 6689 e-24   0.000 000 0031 e-24   u
# 赫兹-原子质量单位关系，单位为原子质量单位(u)

# hertz-electron volt relationship                       4.135 667 516 e-15    0.000 000 091 e-15    eV
# 赫兹-电子伏特关系，单位为电子伏特(eV)

# hertz-hartree relationship                             1.519 829 8460045e-16 0.000 000 0000076e-16 E_h
# 赫兹-哈特里关系，单位为哈特里(E_h)

# hertz-inverse meter relationship                       3.335 640 951... e-9  (exact)               m^-1
# 赫兹-米的倒数关系，单位为米的倒数(m^-1)

# hertz-joule relationship                               6.626 069 57 e-34     0.000 000 29 e-34     J
# 赫兹-焦耳关系，单位为焦耳(J)
# Hertz-Kelvin 关系
4.799 2434 e-11       0.000 0044 e-11       K
# Hertz-Kilogram 关系
7.372 496 68 e-51     0.000 000 33 e-51     kg
# 精细结构常数的倒数
137.035 999 074       0.000 000 044
# 米和原子质量单位的倒数关系
1.331 025 051 20 e-15 0.000 000 000 94 e-15 u
# 米和电子伏特的倒数关系
1.239 841 930 e-6     0.000 000 027 e-6     eV
# 米和哈特里的倒数关系
4.556 335 252 755 e-8 0.000 000 000 023 e-8 E_h
# 米和赫兹的倒数关系
299 792 458           (exact)               Hz
# 米和焦耳的倒数关系
1.986 445 684 e-25    0.000 000 088 e-25    J
# 米和开尔文的倒数关系
1.438 7770 e-2        0.000 0013 e-2        K
# 米和千克的倒数关系
2.210 218 902 e-42    0.000 000 098 e-42    kg
# 导电量子的倒数
12 906.403 7217       0.000 0042            ohm
# 约瑟夫森常数
483 597.870 e9        0.011 e9              Hz V^-1
# 焦耳和原子质量单位的关系
6.700 535 85 e9       0.000 000 30 e9       u
# 焦耳和电子伏特的关系
6.241 509 34 e18      0.000 000 14 e18      eV
# 焦耳和哈特里的关系
2.293 712 48 e17      0.000 000 10 e17      E_h
# 焦耳和赫兹的关系
1.509 190 311 e33     0.000 000 067 e33     Hz
# 焦耳和米的倒数关系
5.034 117 01 e24      0.000 000 22 e24      m^-1
# 焦耳和开尔文的关系
7.242 9716 e22        0.000 0066 e22        K
# 焦耳和千克的关系
1.112 650 056... e-17 (exact)               kg
# 开尔文和原子质量单位的关系
9.251 0868 e-14       0.000 0084 e-14       u
# 开尔文和电子伏特的关系
8.617 3324 e-5        0.000 0078 e-5        eV
# 开尔文和哈特里的关系
3.166 8114 e-6        0.000 0029 e-6        E_h
# 开尔文和赫兹的关系
2.083 6618 e10        0.000 0019 e10        Hz
# 开尔文和米的倒数关系
69.503 476            0.000 063             m^-1
# 开尔文和焦耳的关系
1.380 6488 e-23       0.000 0013 e-23       J
# 开尔文和千克的关系
1.536 1790 e-40       0.000 0014 e-40       kg
# 千克和原子质量单位的关系
6.022 141 29 e26      0.000 000 27 e26      u
# 千克和电子伏特的关系
5.609 588 85 e35      0.000 000 12 e35      eV
# 千克和哈特里的关系
2.061 485 968 e34     0.000 000 091 e34     E_h
# kilogram-hertz relationship                            1.356 392 608 e50     0.000 000 060 e50     Hz
kilogram_hertz_relationship = 1.356392608e50

# kilogram-inverse meter relationship                    4.524 438 73 e41      0.000 000 20 e41      m^-1
kilogram_inverse_meter_relationship = 4.52443873e41

# kilogram-joule relationship                            8.987 551 787... e16  (exact)               J
kilogram_joule_relationship = 8.987551787e16  # Exact value in joules

# kilogram-kelvin relationship                           6.509 6582 e39        0.000 0059 e39        K
kilogram_kelvin_relationship = 6.5096582e39

# lattice parameter of silicon                           543.102 0504 e-12     0.000 0089 e-12       m
lattice_parameter_silicon = 543.1020504e-12

# Loschmidt constant (273.15 K, 100 kPa)                 2.651 6462 e25        0.000 0024 e25        m^-3
Loschmidt_constant_273_15_100kPa = 2.6516462e25

# Loschmidt constant (273.15 K, 101.325 kPa)             2.686 7805 e25        0.000 0024 e25        m^-3
Loschmidt_constant_273_15_101325kPa = 2.6867805e25

# mag. constant                                          12.566 370 614... e-7 (exact)               N A^-2
magnetic_constant = 12.566370614e-7  # Exact value in N A^-2

# mag. flux quantum                                      2.067 833 758 e-15    0.000 000 046 e-15    Wb
magnetic_flux_quantum = 2.067833758e-15

# molar gas constant                                     8.314 4621            0.000 0075            J mol^-1 K^-1
molar_gas_constant = 8.3144621

# molar mass constant                                    1 e-3                 (exact)               kg mol^-1
molar_mass_constant = 1e-3  # Exact value in kg mol^-1

# molar mass of carbon-12                                12 e-3                (exact)               kg mol^-1
molar_mass_carbon_12 = 12e-3  # Exact value in kg mol^-1

# molar Planck constant                                  3.990 312 7176 e-10   0.000 000 0028 e-10   J s mol^-1
molar_Planck_constant = 3.9903127176e-10

# molar Planck constant times c                          0.119 626 565 779     0.000 000 000 084     J m mol^-1
molar_Planck_constant_times_c = 0.119626565779

# molar volume of ideal gas (273.15 K, 100 kPa)          22.710 953 e-3        0.000 021 e-3         m^3 mol^-1
molar_volume_ideal_gas_273_15_100kPa = 22.710953e-3

# molar volume of ideal gas (273.15 K, 101.325 kPa)      22.413 968 e-3        0.000 020 e-3         m^3 mol^-1
molar_volume_ideal_gas_273_15_101325kPa = 22.413968e-3

# molar volume of silicon                                12.058 833 01 e-6     0.000 000 80 e-6      m^3 mol^-1
molar_volume_silicon = 12.05883301e-6

# Mo x unit                                              1.002 099 52 e-13     0.000 000 53 e-13     m
Mo_x_unit = 1.00209952e-13

# muon Compton wavelength                                11.734 441 03 e-15    0.000 000 30 e-15     m
muon_Compton_wavelength = 11.73444103e-15

# muon Compton wavelength over 2 pi                      1.867 594 294 e-15    0.000 000 047 e-15    m
muon_Compton_wavelength_over_2pi = 1.867594294e-15

# muon-electron mass ratio                               206.768 2843          0.000 0052
muon_electron_mass_ratio = 206.7682843

# muon g factor                                          -2.002 331 8418       0.000 000 0013
muon_g_factor = -2.0023318418

# muon mag. mom.                                         -4.490 448 07 e-26    0.000 000 15 e-26     J T^-1
muon_magnetic_moment = -4.49044807e-26

# muon mag. mom. anomaly                                 1.165 920 91 e-3      0.000 000 63 e-3
muon_magnetic_moment_anomaly = 1.16592091e-3

# muon mag. mom. to Bohr magneton ratio                  -4.841 970 44 e-3     0.000 000 12 e-3
muon_magnetic_moment_to_Bohr_magneton_ratio = -4.84197044e-3

# muon mag. mom. to nuclear magneton ratio               -8.890 596 97         0.000 000 22
muon_magnetic_moment_to_nuclear_magneton_ratio = -8.89059697

# muon mass                                              1.883 531 475 e-28    0.000 000 096 e-28    kg
muon_mass = 1.883531475e-28

# muon mass energy equivalent                            1.692 833 667 e-11    0.000 000 086 e-11    J
muon_mass_energy_equivalent = 1.692833667e-11

# muon mass energy equivalent in MeV                     105.658 3715          0.000 0035            MeV
muon_mass_energy_equivalent_MeV = 105.6583715
# muon mass in u
# The mass of a muon expressed in atomic mass units (u).
muon_mass_u = 0.113 428 9267        # value of muon mass in u
muon_mass_u_uncertainty = 0.000 000 0029        # uncertainty in the muon mass in u

# muon molar mass
# The molar mass of a muon in kilograms per mole (kg mol^-1).
muon_molar_mass = 0.113 428 9267 e-3    # value of muon molar mass in kg mol^-1
muon_molar_mass_uncertainty = 0.000 000 0029 e-3    # uncertainty in the muon molar mass in kg mol^-1

# muon-neutron mass ratio
# The ratio of the mass of a muon to the mass of a neutron.
muon_neutron_mass_ratio = 0.112 454 5177        # value of muon-neutron mass ratio
muon_neutron_mass_ratio_uncertainty = 0.000 000 0028        # uncertainty in the muon-neutron mass ratio

# muon-proton mag. mom. ratio
# The ratio of the magnetic moment of a muon to the magnetic moment of a proton.
muon_proton_mag_mom_ratio = -3.183 345 107        # value of muon-proton magnetic moment ratio
muon_proton_mag_mom_ratio_uncertainty = 0.000 000 084        # uncertainty in the muon-proton magnetic moment ratio

# muon-proton mass ratio
# The ratio of the mass of a muon to the mass of a proton.
muon_proton_mass_ratio = 0.112 609 5272        # value of muon-proton mass ratio
muon_proton_mass_ratio_uncertainty = 0.000 000 0028        # uncertainty in the muon-proton mass ratio

# muon-tau mass ratio
# The ratio of the mass of a muon to the mass of a tau lepton.
muon_tau_mass_ratio = 5.946 49 e-2          # value of muon-tau mass ratio
muon_tau_mass_ratio_uncertainty = 0.000 54 e-2          # uncertainty in the muon-tau mass ratio

# natural unit of action
# The value of the natural unit of action in joule-seconds (J s).
natural_unit_of_action = 1.054 571 726 e-34    # value of natural unit of action in J s
natural_unit_of_action_uncertainty = 0.000 000 047 e-34    # uncertainty in the natural unit of action in J s

# natural unit of action in eV s
# The value of the natural unit of action in electron-volt seconds (eV s).
natural_unit_of_action_eV = 6.582 119 28 e-16     # value of natural unit of action in eV s
natural_unit_of_action_eV_uncertainty = 0.000 000 15 e-16     # uncertainty in the natural unit of action in eV s

# natural unit of energy
# The value of the natural unit of energy in joules (J).
natural_unit_of_energy = 8.187 105 06 e-14     # value of natural unit of energy in J
natural_unit_of_energy_uncertainty = 0.000 000 36 e-14     # uncertainty in the natural unit of energy in J

# natural unit of energy in MeV
# The value of the natural unit of energy in mega-electron volts (MeV).
natural_unit_of_energy_MeV = 0.510 998 928         # value of natural unit of energy in MeV
natural_unit_of_energy_MeV_uncertainty = 0.000 000 011         # uncertainty in the natural unit of energy in MeV

# natural unit of length
# The value of the natural unit of length in meters (m).
natural_unit_of_length = 386.159 268 00 e-15   # value of natural unit of length in m
natural_unit_of_length_uncertainty = 0.000 000 25 e-15   # uncertainty in the natural unit of length in m

# natural unit of mass
# The value of the natural unit of mass in kilograms (kg).
natural_unit_of_mass = 9.109 382 91 e-31     # value of natural unit of mass in kg
natural_unit_of_mass_uncertainty = 0.000 000 40 e-31     # uncertainty in the natural unit of mass in kg

# natural unit of momentum
# The value of the natural unit of momentum in kilogram meters per second (kg m s^-1).
natural_unit_of_momentum = 2.730 924 29 e-22     # value of natural unit of momentum in kg m s^-1
natural_unit_of_momentum_uncertainty = 0.000 000 12 e-22     # uncertainty in the natural unit of momentum in kg m s^-1

# natural unit of momentum in MeV/c
# The value of the natural unit of momentum in mega-electron volts per speed of light (MeV/c).
natural_unit_of_momentum_MeVc = 0.510 998 928         # value of natural unit of momentum in MeV/c
natural_unit_of_momentum_MeVc_uncertainty = 0.000 000 011         # uncertainty in the natural unit of momentum in MeV/c

# natural unit of time
# The value of the natural unit of time in seconds (s).
natural_unit_of_time = 1.288 088 668 33 e-21    # value of natural unit of time in s
natural_unit_of_time_uncertainty = 0.000 000 000 83 e-21    # uncertainty in the natural unit of time in s

# natural unit of velocity
# The exact value of the natural unit of velocity in meters per second (m s^-1).
natural_unit_of_velocity = 299 792 458           # exact value of natural unit of velocity in m s^-1

# neutron Compton wavelength
# The value of the neutron Compton wavelength in meters (m).
neutron_Compton_wavelength = 1.319 590 9068 e-15     # value of neutron Compton wavelength in m
neutron_Compton_wavelength_uncertainty = 0.000 000 0011 e-15   # uncertainty in the neutron Compton wavelength in m

# neutron Compton wavelength over 2 pi
# The value of the neutron Compton wavelength divided by 2 pi in meters (m).
neutron_Compton_wavelength_over_2pi = 0.210 019 415 68 e-15   # value of neutron Compton wavelength over 2 pi in m
neutron_Compton_wavelength_over_2pi_uncertainty = 0.000 000 000 17 e-15   # uncertainty in the neutron Compton wavelength over 2 pi in m

# neutron-electron mag. mom. ratio
# The ratio of the magnetic moment of a neutron to the magnetic moment of an electron.
neutron_electron_mag_mom_ratio = 1.040 668 82 e-3    # value of neutron-electron magnetic moment ratio
neutron_electron_mag_mom_ratio_uncertainty = 0.000 000 25 e-3    # uncertainty in the neutron-electron magnetic moment ratio

# neutron-electron mass ratio
# The ratio of the mass of a neutron to the mass of an electron.
neutron_electron_mass_ratio = 1838.683 6605       # value of neutron-electron mass ratio
neutron_electron_mass_ratio_uncertainty = 0.000 0011       # uncertainty in the neutron-electron mass ratio

# neutron g factor
# The g-factor (gyromagnetic ratio) of a neutron.
neutron_g_factor = -3.826 085 45         # value of neutron g factor
neutron_g_factor_uncertainty = 0.000 000 90         # uncertainty in the neutron g factor

# neutron gyromagnetic ratio
# The gyromagnetic ratio of a neutron in seconds^-1 tesla^-1.
neutron_gyromag_ratio = 1.832 471 79 e8       # value of neutron gyromagnetic ratio in s^-1 T^-1
neutron_gyromag_ratio_uncertainty = 0.000 000 43 e8       # uncertainty in the neutron gyromagnetic ratio in s^-1 T^-1

# neutron gyromagnetic ratio over 2 pi
# The gyromagnetic ratio of a neutron divided by 2 pi in megahertz per tesla (MHz T^-1).
neutron_gyromag_ratio_over_2pi = 29.164 6943         # value of neutron gyromagnetic ratio over 2 pi in MHz T^-1
neutron_gyromag_ratio_over_2pi_uncertainty = 0.000 0069         # uncertainty in the neutron gyromagnetic ratio over 2 pi in MHz T^-1

# neutron magnetic moment
# The magnetic moment of a neutron in joules per tesla (J T^-1).
neutron_mag_moment = -0.966 236 47 e-26      # value of neutron magnetic moment in J T^-1
neutron_mag_moment_uncertainty = 0.000 000 23 e-26      # uncertainty in the neutron
# neutron molar mass
# 中子的摩尔质量
neutron molar mass                                     1.008 664 916 00 e-3  0.000 000 000 43 e-3  kg mol^-1

# neutron-muon mass ratio
# 中子与μ子的质量比
neutron-muon mass ratio                                8.892 484 00          0.000 000 22

# neutron-proton mag. mom. ratio
# 中子与质子的磁矩比
neutron-proton mag. mom. ratio                         -0.684 979 34         0.000 000 16

# neutron-proton mass difference
# 中子与质子的质量差
neutron-proton mass difference                         2.305 573 92 e-30     0.000 000 76 e-30

# neutron-proton mass difference energy equivalent
# 中子与质子的质量差能量等效值
neutron-proton mass difference energy equivalent       2.072 146 50 e-13     0.000 000 68 e-13

# neutron-proton mass difference energy equivalent in MeV
# 中子与质子的质量差能量等效值（单位为兆电子伏）
neutron-proton mass difference energy equivalent in MeV 1.293 332 17          0.000 000 42

# neutron-proton mass difference in u
# 中子与质子的质量差（单位为原子质量单位）
neutron-proton mass difference in u                    0.001 388 449 19      0.000 000 000 45

# neutron-proton mass ratio
# 中子与质子的质量比
neutron-proton mass ratio                              1.001 378 419 17      0.000 000 000 45

# neutron-tau mass ratio
# 中子与τ子的质量比
neutron-tau mass ratio                                 0.528 790             0.000 048

# neutron to shielded proton mag. mom. ratio
# 中子到屏蔽质子的磁矩比
neutron to shielded proton mag. mom. ratio             -0.684 996 94         0.000 000 16

# Newtonian constant of gravitation
# 牛顿引力常数
Newtonian constant of gravitation                      6.673 84 e-11         0.000 80 e-11         m^3 kg^-1 s^-2

# Newtonian constant of gravitation over h-bar c
# 牛顿引力常数与约化普朗克常数乘以光速的比值
Newtonian constant of gravitation over h-bar c         6.708 37 e-39         0.000 80 e-39         (GeV/c^2)^-2

# nuclear magneton
# 核磁子
nuclear magneton                                       5.050 783 53 e-27     0.000 000 11 e-27     J T^-1

# nuclear magneton in eV/T
# 核磁子的电子伏特每特斯拉
nuclear magneton in eV/T                               3.152 451 2605 e-8    0.000 000 0022 e-8    eV T^-1

# nuclear magneton in inverse meters per tesla
# 核磁子的每特斯拉的米的倒数
nuclear magneton in inverse meters per tesla           2.542 623 527 e-2     0.000 000 056 e-2     m^-1 T^-1

# nuclear magneton in K/T
# 核磁子的每特斯拉的开尔文
nuclear magneton in K/T                                3.658 2682 e-4        0.000 0033 e-4        K T^-1

# nuclear magneton in MHz/T
# 核磁子的每特斯拉的兆赫兹
nuclear magneton in MHz/T                              7.622 593 57          0.000 000 17          MHz T^-1

# Planck constant
# 普朗克常数
Planck constant                                        6.626 069 57 e-34     0.000 000 29 e-34     J s

# Planck constant in eV s
# 普朗克常数的电子伏特秒
Planck constant in eV s                                4.135 667 516 e-15    0.000 000 091 e-15    eV s

# Planck constant over 2 pi
# 普朗克常数除以2π
Planck constant over 2 pi                              1.054 571 726 e-34    0.000 000 047 e-34    J s

# Planck constant over 2 pi in eV s
# 普朗克常数除以2π的电子伏特秒
Planck constant over 2 pi in eV s                      6.582 119 28 e-16     0.000 000 15 e-16     eV s

# Planck constant over 2 pi times c in MeV fm
# 普朗克常数除以2π乘以光速的乘积的兆电子伏特·费米
Planck constant over 2 pi times c in MeV fm            197.326 9718          0.000 0044            MeV fm

# Planck length
# 普朗克长度
Planck length                                          1.616 199 e-35        0.000 097 e-35        m

# Planck mass
# 普朗克质量
Planck mass                                            2.176 51 e-8          0.000 13 e-8          kg

# Planck mass energy equivalent in GeV
# 普朗克质量的能量等效值（单位为吉电子伏特）
Planck mass energy equivalent in GeV                   1.220 932 e19         0.000 073 e19         GeV

# Planck temperature
# 普朗克温度
Planck temperature                                     1.416 833 e32         0.000 085 e32         K

# Planck time
# 普朗克时间
Planck time                                            5.391 06 e-44         0.000 32 e-44         s

# proton charge to mass quotient
# 质子电荷与质量的比值
proton charge to mass quotient                         9.578 833 58 e7       0.000 000 21 e7       C kg^-1

# proton Compton wavelength
# 质子的康普顿波长
proton Compton wavelength                              1.321 409 856 23 e-15 0.000 000 000 94 e-15 m
# 定义变量：质子 Compton 波长除以 2π
proton_Compton_wavelength_over_2pi = 0.21030891047e-15  # 单位：m

# 定义变量：质子电子质量比
proton_electron_mass_ratio = 1836.15267245  # 无单位

# 定义变量：质子 g 因子
proton_g_factor = 5.585694713  # 无单位

# 定义变量：质子陀螺磁比率
proton_gyromag_ratio = 2.675222005e8  # 单位：s^-1 T^-1

# 定义变量：质子陀螺磁比率除以 2π
proton_gyromag_ratio_over_2pi = 42.5774806  # 单位：MHz T^-1

# 定义变量：质子磁矩
proton_mag_mom = 1.410606743e-26  # 单位：J T^-1

# 定义变量：质子磁矩对玻尔磁子比值
proton_mag_mom_to_Bohr_magneton_ratio = 1.521032210e-3  # 无单位

# 定义变量：质子磁矩对核磁子比值
proton_mag_mom_to_nuclear_magneton_ratio = 2.792847356  # 无单位

# 定义变量：质子磁矩的屏蔽修正
proton_mag_shielding_correction = 25.694e-6  # 单位：无单位

# 定义变量：质子质量
proton_mass = 1.672621777e-27  # 单位：kg

# 定义变量：质子质能等效
proton_mass_energy_equivalent = 1.503277484e-10  # 单位：J

# 定义变量：质子质能等效转换为 MeV
proton_mass_energy_equivalent_in_MeV = 938.272046  # 单位：MeV

# 定义变量：质子质量单位原子质量单位（u）
proton_mass_in_u = 1.007276466812  # 单位：u

# 定义变量：质子摩尔质量
proton_molar_mass = 1.007276466812e-3  # 单位：kg mol^-1

# 定义变量：质子-μ子质量比
proton_muon_mass_ratio = 8.88024331  # 无单位

# 定义变量：质子-中子磁矩比
proton_neutron_mag_mom_ratio = -1.45989806  # 无单位

# 定义变量：质子-中子质量比
proton_neutron_mass_ratio = 0.99862347826  # 无单位

# 定义变量：质子有效电荷半径（均方根）
proton_rms_charge_radius = 0.8775e-15  # 单位：m

# 定义变量：质子-τ子质量比
proton_tau_mass_ratio = 0.528063  # 无单位

# 定义变量：量子循环
quantum_of_circulation = 3.6369475520e-4  # 单位：m^2 s^-1

# 定义变量：量子循环的两倍
quantum_of_circulation_times_2 = 7.2738951040e-4  # 单位：m^2 s^-1

# 定义变量：Rydberg 常数
Rydberg_constant = 10973731.568539  # 单位：m^-1

# 定义变量：Rydberg 常数乘以 c 的频率
Rydberg_constant_times_c_in_Hz = 3.289841960364e15  # 单位：Hz

# 定义变量：Rydberg 常数乘以 Planck 常数 h 乘以光速 c 的电子伏特单位
Rydberg_constant_times_hc_in_eV = 13.60569253  # 单位：eV

# 定义变量：Rydberg 常数乘以 Planck 常数 h 乘以光速 c 的焦耳单位
Rydberg_constant_times_hc_in_J = 2.179872171e-18  # 单位：J

# 定义变量：Sackur-Tetrode 常数（1K, 100kPa）
Sackur_Tetrode_constant_1K_100kPa = -1.1517078  # 无单位

# 定义变量：Sackur-Tetrode 常数（1K, 101.325kPa）
Sackur_Tetrode_constant_1K_101325kPa = -1.1648708  # 无单位

# 定义变量：第二辐射常数
second_radiation_constant = 1.4387770e-2  # 单位：m K

# 定义变量：屏蔽后的氦核陀螺磁比率
shielded_helion_gyromag_ratio = 2.037894659e8  # 单位：s^-1 T^-1

# 定义变量：屏蔽后的氦核陀螺磁比率除以 2π
shielded_helion_gyromag_ratio_over_2pi = 32.43410084  # 单位：MHz T^-1
# 稳定质子到电子磁动量比
shielded helion mag. mom.                              -1.074 553 044 e-26   0.000 000 027 e-26    J T^-1
# 稳定质子到玻尔磁子比
shielded helion mag. mom. to Bohr magneton ratio       -1.158 671 471 e-3    0.000 000 014 e-3
# 稳定质子到核磁子比
shielded helion mag. mom. to nuclear magneton ratio    -2.127 497 718        0.000 000 025
# 稳定质子到质子磁动量比
shielded helion to proton mag. mom. ratio              -0.761 766 558        0.000 000 011
# 稳定质子到屏蔽质子磁动量比
shielded helion to shielded proton mag. mom. ratio     -0.761 786 1313       0.000 000 0033
# 屏蔽质子陀螺磁比率
shielded proton gyromag. ratio                         2.675 153 268 e8      0.000 000 066 e8      s^-1 T^-1
# 屏蔽质子陀螺磁比率除以2π
shielded proton gyromag. ratio over 2 pi               42.576 3866           0.000 0010            MHz T^-1
# 稳定质子磁动量
shielded proton mag. mom.                              1.410 570 499 e-26    0.000 000 035 e-26    J T^-1
# 稳定质子到玻尔磁子比
shielded proton mag. mom. to Bohr magneton ratio       1.520 993 128 e-3     0.000 000 017 e-3
# 稳定质子到核磁子比
shielded proton mag. mom. to nuclear magneton ratio    2.792 775 598         0.000 000 030
# 真空中的光速
speed of light in vacuum                               299 792 458           (exact)               m s^-1
# 标准重力加速度
standard acceleration of gravity                       9.806 65              (exact)               m s^-2
# 标准大气压
standard atmosphere                                    101 325               (exact)               Pa
# 标准状态压力
standard-state pressure                                100 000               (exact)               Pa
# Stefan-Boltzmann常数
Stefan-Boltzmann constant                              5.670 373 e-8         0.000 021 e-8         W m^-2 K^-4
# τ子康普顿波长
tau Compton wavelength                                 0.697 787 e-15        0.000 063 e-15        m
# τ子康普顿波长除以2π
tau Compton wavelength over 2 pi                       0.111 056 e-15        0.000 010 e-15        m
# τ电子质量比
tau-electron mass ratio                                3477.15               0.31
# τ子质量
tau mass                                               3.167 47 e-27         0.000 29 e-27         kg
# τ子质量能量等效
tau mass energy equivalent                             2.846 78 e-10         0.000 26 e-10         J
# τ子质量能量等效（单位：兆电子伏特）
tau mass energy equivalent in MeV                      1776.82               0.16                  MeV
# τ子质量（单位：原子质量单位）
tau mass in u                                          1.907 49              0.000 17              u
# τ子摩尔质量
tau molar mass                                         1.907 49 e-3          0.000 17 e-3          kg mol^-1
# τ-μ子质量比
tau-muon mass ratio                                    16.8167               0.0015
# τ-中子质量比
tau-neutron mass ratio                                 1.891 11              0.000 17
# τ-质子质量比
tau-proton mass ratio                                  1.893 72              0.000 17
# 汤姆逊散射截面
Thomson cross section                                  0.665 245 8734 e-28   0.000 000 0013 e-28   m^2
# 氚-电子质量比
triton-electron mass ratio                             5496.921 5267         0.000 0050
# 氚g因子
triton g factor                                        5.957 924 896         0.000 000 076
# 氚磁动量
triton mag. mom.                                       1.504 609 447 e-26    0.000 000 038 e-26    J T^-1
# triton mag. mom. to Bohr magneton ratio
# Triton的磁矩与玻尔磁子比例
triton_mag_mom_to_bohr_magneton_ratio = 1.622393657e-3    # 值及其标准误差

# triton mag. mom. to nuclear magneton ratio
# Triton的磁矩与核磁子比例
triton_mag_mom_to_nuclear_magneton_ratio = 2.978962448    # 值及其标准误差

# triton mass
# Triton的质量
triton_mass = 5.00735630e-27    # 千克，及其标准误差

# triton mass energy equivalent
# Triton的质能转换值
triton_mass_energy_equivalent = 4.50038741e-10    # 焦耳，及其标准误差

# triton mass energy equivalent in MeV
# Triton的质能转换值（以兆电子伏特为单位）
triton_mass_energy_equivalent_in_MeV = 2808.921005    # 兆电子伏特，及其标准误差

# triton mass in u
# Triton的质量（以原子单位为单位）
triton_mass_in_u = 3.0155007134    # 原子单位，及其标准误差

# triton molar mass
# Triton的摩尔质量
triton_molar_mass = 3.0155007134e-3    # 千克每摩尔，及其标准误差

# triton-proton mass ratio
# Triton与质子质量的比值
triton_proton_mass_ratio = 2.9937170308    # 值及其标准误差

# unified atomic mass unit
# 统一原子质量单位
unified_atomic_mass_unit = 1.660538921e-27    # 千克，及其标准误差

# von Klitzing constant
# 冯·克里兹常数
von_Klitzing_constant = 25812.8074434    # 欧姆，及其标准误差

# weak mixing angle
# 弱作用混合角
weak_mixing_angle = 0.2223    # 值及其标准误差

# Wien frequency displacement law constant
# 维恩频率位移定律常数
Wien_frequency_displacement_law_constant = 5.8789254e10    # 赫兹·开尔文^-1，及其标准误差

# Wien wavelength displacement law constant
# 维恩波长位移定律常数
Wien_wavelength_displacement_law_constant = 2.8977721e-3    # 米·开尔文^-1，及其标准误差

# txt2014: a string containing additional physical constants
# txt2014: 包含额外物理常数的字符串
txt2014 = """\
{220} lattice spacing of silicon                       192.015 5714 e-12     0.000 0032 e-12       m
alpha particle-electron mass ratio                     7294.299 541 36       0.000 000 24
alpha particle mass                                    6.644 657 230 e-27    0.000 000 082 e-27    kg
alpha particle mass energy equivalent                  5.971 920 097 e-10    0.000 000 073 e-10    J
alpha particle mass energy equivalent in MeV           3727.379 378          0.000 023             MeV
alpha particle mass in u                               4.001 506 179 127     0.000 000 000 063     u
alpha particle molar mass                              4.001 506 179 127 e-3 0.000 000 000 063 e-3 kg mol^-1
alpha particle-proton mass ratio                       3.972 599 689 07      0.000 000 000 36
Angstrom star                                          1.000 014 95 e-10     0.000 000 90 e-10     m
atomic mass constant                                   1.660 539 040 e-27    0.000 000 020 e-27    kg
atomic mass constant energy equivalent                 1.492 418 062 e-10    0.000 000 018 e-10    J
atomic mass constant energy equivalent in MeV          931.494 0954          0.000 0057            MeV
atomic mass unit-electron volt relationship            931.494 0954 e6       0.000 0057 e6         eV
atomic mass unit-hartree relationship                  3.423 177 6902 e7     0.000 000 0016 e7     E_h
atomic mass unit-hertz relationship                    2.252 342 7206 e23    0.000 000 0010 e23    Hz
atomic mass unit-inverse meter relationship            7.513 006 6166 e14    0.000 000 0034 e14    m^-1
"""
# 原子质量单位与焦耳关系，第一列为数值，第二列为不确定度，第三列为单位
atomic mass unit-joule relationship                    1.492 418 062 e-10    0.000 000 018 e-10    J
# 原子质量单位与开尔文关系，第一列为数值，第二列为不确定度，第三列为单位
atomic mass unit-kelvin relationship                   1.080 954 38 e13      0.000 000 62 e13      K
# 原子质量单位与千克关系，第一列为数值，第二列为不确定度，第三列为单位
atomic mass unit-kilogram relationship                 1.660 539 040 e-27    0.000 000 020 e-27    kg
# 第一超极化率的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of 1st hyperpolarizability                 3.206 361 329 e-53    0.000 000 020 e-53    C^3 m^3 J^-2
# 第二超极化率的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of 2nd hyperpolarizability                 6.235 380 085 e-65    0.000 000 077 e-65    C^4 m^4 J^-3
# 动量的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of action                                  1.054 571 800 e-34    0.000 000 013 e-34    J s
# 电荷的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of charge                                  1.602 176 6208 e-19   0.000 000 0098 e-19   C
# 电荷密度的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of charge density                          1.081 202 3770 e12    0.000 000 0067 e12    C m^-3
# 电流的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of current                                 6.623 618 183 e-3     0.000 000 041 e-3     A
# 电偶极矩的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of electric dipole mom.                    8.478 353 552 e-30    0.000 000 052 e-30    C m
# 电场的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of electric field                          5.142 206 707 e11     0.000 000 032 e11     V m^-1
# 电场梯度的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of electric field gradient                 9.717 362 356 e21     0.000 000 060 e21     V m^-2
# 电极化率的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of electric polarizability                 1.648 777 2731 e-41   0.000 000 0011 e-41   C^2 m^2 J^-1
# 电位的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of electric potential                      27.211 386 02         0.000 000 17          V
# 电四极矩的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of electric quadrupole mom.                4.486 551 484 e-40    0.000 000 028 e-40    C m^2
# 能量的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of energy                                  4.359 744 650 e-18    0.000 000 054 e-18    J
# 力的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of force                                   8.238 723 36 e-8      0.000 000 10 e-8      N
# 长度的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of length                                  0.529 177 210 67 e-10 0.000 000 000 12 e-10 m
# 磁偶极矩的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of mag. dipole mom.                        1.854 801 999 e-23    0.000 000 011 e-23    J T^-1
# 磁通量密度的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of mag. flux density                       2.350 517 550 e5      0.000 000 014 e5      T
# 磁化率的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of magnetizability                         7.891 036 5886 e-29   0.000 000 0090 e-29   J T^-2
# 质量的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of mass                                    9.109 383 56 e-31     0.000 000 11 e-31     kg
# 动量的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of mom.um                                  1.992 851 882 e-24    0.000 000 024 e-24    kg m s^-1
# 介电常数的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of permittivity                            1.112 650 056... e-10 (exact)               F m^-1
# 时间的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of time                                    2.418 884 326509e-17  0.000 000 000014e-17  s
# 速度的原子单位，第一列为数值，第二列为不确定度，第三列为单位
atomic unit of velocity                                2.187 691 262 77 e6   0.000 000 000 50 e6   m s^-1
# 阿伏伽德罗常数，第一列为数值，第二列为不确定度，第三列为单位
Avogadro constant                                      6.022 140 857 e23     0.000 000 074 e23     mol^-1
# 玻尔磁子，第一列为数值，第二列为不确定度，第三列为单位
Bohr magneton                                          927.400 9994 e-26     0.000 0057 e-26       J T^-1
# Bohr magneton in eV/T                                  5.788 381 8012 e-5    0.000 000 0026 e-5    eV T^-1
# 定义 Bohr 磁子在电子伏特每特斯拉单位下的值及其不确定性，单位为电子伏特每特斯拉
Bohr_magneton_in_eV_per_T = 5.7883818012e-5
Bohr_magneton_in_eV_per_T_error = 0.0000000026e-5

# Bohr magneton in Hz/T                                  13.996 245 042 e9     0.000 000 086 e9      Hz T^-1
# 定义 Bohr 磁子在赫兹每特斯拉单位下的值及其不确定性，单位为赫兹每特斯拉
Bohr_magneton_in_Hz_per_T = 13.996245042e9
Bohr_magneton_in_Hz_per_T_error = 0.000000086e9

# Bohr magneton in inverse meters per tesla              46.686 448 14         0.000 000 29          m^-1 T^-1
# 定义 Bohr 磁子在每特斯拉下每米单位倒数的值及其不确定性，单位为每米每特斯拉倒数
Bohr_magneton_in_inverse_m_per_T = 46.68644814
Bohr_magneton_in_inverse_m_per_T_error = 0.00000029

# Bohr magneton in K/T                                   0.671 714 05          0.000 000 39          K T^-1
# 定义 Bohr 磁子在开尔文每特斯拉单位下的值及其不确定性，单位为开尔文每特斯拉
Bohr_magneton_in_K_per_T = 0.67171405
Bohr_magneton_in_K_per_T_error = 0.00000039

# Bohr radius                                            0.529 177 210 67 e-10 0.000 000 000 12 e-10 m
# 定义 Bohr 半径的值及其不确定性，单位为米
Bohr_radius = 0.52917721067e-10
Bohr_radius_error = 0.00000000012e-10

# Boltzmann constant                                     1.380 648 52 e-23     0.000 000 79 e-23     J K^-1
# 定义玻尔兹曼常数的值及其不确定性，单位为焦耳每开尔文
Boltzmann_constant = 1.38064852e-23
Boltzmann_constant_error = 0.00000079e-23

# Boltzmann constant in eV/K                             8.617 3303 e-5        0.000 0050 e-5        eV K^-1
# 定义玻尔兹曼常数在电子伏特每开尔文单位下的值及其不确定性，单位为电子伏特每开尔文
Boltzmann_constant_in_eV_per_K = 8.6173303e-5
Boltzmann_constant_in_eV_per_K_error = 0.0000050e-5

# Boltzmann constant in Hz/K                             2.083 6612 e10        0.000 0012 e10        Hz K^-1
# 定义玻尔兹曼常数在赫兹每开尔文单位下的值及其不确定性，单位为赫兹每开尔文
Boltzmann_constant_in_Hz_per_K = 2.0836612e10
Boltzmann_constant_in_Hz_per_K_error = 0.0000012e10

# Boltzmann constant in inverse meters per kelvin        69.503 457            0.000 040             m^-1 K^-1
# 定义玻尔兹曼常数在每开尔文下每米单位倒数的值及其不确定性，单位为每米每开尔文倒数
Boltzmann_constant_in_inverse_m_per_K = 69.503457
Boltzmann_constant_in_inverse_m_per_K_error = 0.000040

# characteristic impedance of vacuum                     376.730 313 461...    (exact)               ohm
# 定义真空中的特征阻抗值，以欧姆为单位，精确值
characteristic_impedance_of_vacuum = 376.730313461

# classical electron radius                              2.817 940 3227 e-15   0.000 000 0019 e-15   m
# 定义经典电子半径的值及其不确定性，单位为米
classical_electron_radius = 2.8179403227e-15
classical_electron_radius_error = 0.0000000019e-15

# Compton wavelength                                     2.426 310 2367 e-12   0.000 000 0011 e-12   m
# 定义康普顿波长的值及其不确定性，单位为米
Compton_wavelength = 2.4263102367e-12
Compton_wavelength_error = 0.0000000011e-12

# Compton wavelength over 2 pi                           386.159 267 64 e-15   0.000 000 18 e-15     m
# 定义2π下的康普顿波长的值及其不确定性，单位为米
Compton_wavelength_over_2pi = 386.15926764e-15
Compton_wavelength_over_2pi_error = 0.00000018e-15

# conductance quantum                                    7.748 091 7310 e-5    0.000 000 0018 e-5    S
# 定义导电量子的值及其不确定性，单位为西门子
conductance_quantum = 7.7480917310e-5
conductance_quantum_error = 0.0000000018e-5

# conventional value of Josephson constant               483 597.9 e9          (exact)               Hz V^-1
# 定义约瑟夫森常数的常规值，单位为赫兹每伏特，精确值
conventional_value_of_Josephson_constant = 483597.9e9

# conventional value of von Klitzing constant            25 812.807            (exact)               ohm
# 定义冯·克里青常数的常规值，单位为欧姆，精确值
conventional_value_of_von_Klitzing_constant = 25812.807

# Cu x unit                                              1.002 076 97 e-13     0.000 000 28 e-13     m
# 定义铜的 X 单位值及其不确定性，单位为米
Cu_x_unit = 1.00207697e-13
Cu_x_unit_error = 0.00000028e-13

# deuteron-electron mag. mom. ratio                      -4.664 345 535 e-4    0.000 000 026 e-4
# 定义质子与电子磁矩比的值及其不确定性
deuteron_electron_mag_mom_ratio = -4.664345535e-4
deuteron_electron_mag_mom_ratio_error = 0.000000026e-4

# deuteron-electron mass ratio                           3670.482 967 85       0.000 000 13
# 定义质子与电子质量比的值及其不确定性
deuteron_electron_mass_ratio = 3670.48296785
deuteron_electron_mass_ratio_error = 0.00000013

# deuteron g factor                                      0.857 438 2311        0.000 000 0048
# 定义质子 g 因子的值及其不确定性
deuteron_g_factor = 0.8574382311
deuteron_g_factor_error = 0.0000000048

# deuteron mag. mom.                                     0.433 073 5040 e-26   0.000 000 0036 e-26   J T^-1
# 定义质子磁矩的值及其不确定性，单位为焦耳每特斯拉
deuteron_mag_mom = 0.4330735040e-26
deuteron
# deuteron-proton mag. mom. ratio
deuteron_proton_mag_mom_ratio = 0.3070122077    # 氘-质子磁矩比

# deuteron-proton mass ratio
deuteron_proton_mass_ratio = 1.99900750087     # 氘-质子质量比

# deuteron rms charge radius
deuteron_rms_charge_radius = 2.1413e-15        # 氘的均方根电荷半径，单位为米

# electric constant
electric_constant = 8.854187817e-12             # 电常数，单位为法拉第每米，精确值

# electron charge to mass quotient
electron_charge_to_mass_quotient = -1.758820024e11  # 电子电荷质量比，单位为库仑每千克

# electron-deuteron mag. mom. ratio
electron_deuteron_mag_mom_ratio = -2143.923499  # 电子-氘磁矩比

# electron-deuteron mass ratio
electron_deuteron_mass_ratio = 2.724437107484e-4  # 电子-氘质量比

# electron g factor
electron_g_factor = -2.00231930436182           # 电子g因子

# electron gyromag. ratio
electron_gyromag_ratio = 1.760859644e11         # 电子旋磁比，单位为秒每特斯拉

# electron gyromag. ratio over 2 pi
electron_gyromag_ratio_over_2pi = 28024.95164   # 电子旋磁比除以2π，单位为兆赫每特斯拉

# electron-helion mass ratio
electron_helion_mass_ratio = 1.819543074854e-4  # 电子-氦三质量比

# electron mag. mom.
electron_mag_mom = -928.4764620e-26             # 电子磁矩，单位为焦耳每特斯拉

# electron mag. mom. anomaly
electron_mag_mom_anomaly = 1.15965218091e-3     # 电子磁矩异常

# electron mag. mom. to Bohr magneton ratio
electron_mag_mom_to_Bohr_magneton_ratio = -1.00115965218091  # 电子磁矩到玻尔磁子比

# electron mag. mom. to nuclear magneton ratio
electron_mag_mom_to_nuclear_magneton_ratio = -1838.28197234  # 电子磁矩到核磁子比

# electron mass
electron_mass = 9.10938356e-31                   # 电子质量，单位为千克

# electron mass energy equivalent
electron_mass_energy_equivalent = 8.18710565e-14  # 电子质量能量等效，单位为焦耳

# electron mass energy equivalent in MeV
electron_mass_energy_equivalent_MeV = 0.5109989461  # 电子质量能量等效，单位为兆电子伏特

# electron mass in u
electron_mass_in_u = 5.48579909070e-4           # 电子质量，单位为原子质量单位

# electron molar mass
electron_molar_mass = 5.48579909070e-7          # 电子摩尔质量，单位为千克每摩尔

# electron-muon mag. mom. ratio
electron_muon_mag_mom_ratio = 206.7669880       # 电子-μ子磁矩比

# electron-muon mass ratio
electron_muon_mass_ratio = 4.83633170e-3        # 电子-μ子质量比

# electron-neutron mag. mom. ratio
electron_neutron_mag_mom_ratio = 960.92050      # 电子-中子磁矩比

# electron-neutron mass ratio
electron_neutron_mass_ratio = 5.4386734428e-4   # 电子-中子质量比

# electron-proton mag. mom. ratio
electron_proton_mag_mom_ratio = -658.2106866    # 电子-质子磁矩比

# electron-proton mass ratio
electron_proton_mass_ratio = 5.44617021352e-4   # 电子-质子质量比

# electron-tau mass ratio
electron_tau_mass_ratio = 2.87592e-4            # 电子-τ子质量比

# electron to alpha particle mass ratio
electron_to_alpha_particle_mass_ratio = 1.370933554798e-4  # 电子-α粒子质量比

# electron to shielded helion mag. mom. ratio
electron_to_shielded_helion_mag_mom_ratio = 864.058257  # 电子-屏蔽氦三磁矩比

# electron to shielded proton mag. mom. ratio
electron_to_shielded_proton_mag_mom_ratio = -658.2275971  # 电子-屏蔽质子磁矩比
# electron-triton mass ratio                             1.819 200 062 203 e-4 0.000 000 000 084 e-4
electron_triton_mass_ratio = 1.819200062203e-4  # 定义电子和三重质子质量比

# electron volt                                          1.602 176 6208 e-19   0.000 000 0098 e-19   J
electron_volt = 1.6021766208e-19  # 定义电子伏特单位，对应焦耳

# electron volt-atomic mass unit relationship            1.073 544 1105 e-9    0.000 000 0066 e-9    u
electron_volt_atomic_mass_unit_relationship = 1.0735441105e-9  # 定义电子伏特和原子质量单位的关系

# electron volt-hartree relationship                     3.674 932 248 e-2     0.000 000 023 e-2     E_h
electron_volt_hartree_relationship = 3.674932248e-2  # 定义电子伏特和哈特里单位的关系

# electron volt-hertz relationship                       2.417 989 262 e14     0.000 000 015 e14     Hz
electron_volt_hertz_relationship = 2.417989262e14  # 定义电子伏特和赫兹的关系

# electron volt-inverse meter relationship               8.065 544 005 e5      0.000 000 050 e5      m^-1
electron_volt_inverse_meter_relationship = 8.065544005e5  # 定义电子伏特和米的倒数的关系

# electron volt-joule relationship                       1.602 176 6208 e-19   0.000 000 0098 e-19   J
electron_volt_joule_relationship = 1.6021766208e-19  # 定义电子伏特和焦耳的关系

# electron volt-kelvin relationship                      1.160 452 21 e4       0.000 000 67 e4       K
electron_volt_kelvin_relationship = 1.16045221e4  # 定义电子伏特和开尔文的关系

# electron volt-kilogram relationship                    1.782 661 907 e-36    0.000 000 011 e-36    kg
electron_volt_kilogram_relationship = 1.782661907e-36  # 定义电子伏特和千克的关系

# elementary charge                                      1.602 176 6208 e-19   0.000 000 0098 e-19   C
elementary_charge = 1.6021766208e-19  # 定义元电荷

# elementary charge over h                               2.417 989 262 e14     0.000 000 015 e14     A J^-1
elementary_charge_over_h = 2.417989262e14  # 定义元电荷除以普朗克常数的关系

# Faraday constant                                       96 485.332 89         0.000 59              C mol^-1
Faraday_constant = 96485.33289  # 定义法拉第常数

# Faraday constant for conventional electric current     96 485.3251           0.0012                C_90 mol^-1
Faraday_constant_conventional = 96485.3251  # 定义传统电流法拉第常数

# Fermi coupling constant                                1.166 3787 e-5        0.000 0006 e-5        GeV^-2
Fermi_coupling_constant = 1.1663787e-5  # 定义费米耦合常数

# fine-structure constant                                7.297 352 5664 e-3    0.000 000 0017 e-3
fine_structure_constant = 7.2973525664e-3  # 定义精细结构常数

# first radiation constant                               3.741 771 790 e-16    0.000 000 046 e-16    W m^2
first_radiation_constant = 3.741771790e-16  # 定义第一辐射常数

# first radiation constant for spectral radiance         1.191 042 953 e-16    0.000 000 015 e-16    W m^2 sr^-1
first_radiation_constant_spectral_radiance = 1.191042953e-16  # 定义用于光谱辐射度的第一辐射常数

# hartree-atomic mass unit relationship                  2.921 262 3197 e-8    0.000 000 0013 e-8    u
hartree_atomic_mass_unit_relationship = 2.9212623197e-8  # 定义哈特里和原子质量单位的关系

# hartree-electron volt relationship                     27.211 386 02         0.000 000 17          eV
hartree_electron_volt_relationship = 27.21138602  # 定义哈特里和电子伏特单位的关系

# Hartree energy                                         4.359 744 650 e-18    0.000 000 054 e-18    J
Hartree_energy = 4.359744650e-18  # 定义哈特里能量

# Hartree energy in eV                                   27.211 386 02         0.000 000 17          eV
Hartree_energy_in_eV = 27.21138602  # 定义哈特里能量（电子伏特单位）

# hartree-hertz relationship                             6.579 683 920 711 e15 0.000 000 000 039 e15 Hz
hartree_hertz_relationship = 6.579683920711e15  # 定义哈特里和赫兹的关系

# hartree-inverse meter relationship                     2.194 746 313 702 e7  0.000 000 000 013 e7  m^-1
hartree_inverse_meter_relationship = 2.194746313702e7  # 定义哈特里和米的倒数的关系

# hartree-joule relationship                             4.359 744 650 e-18    0.000 000 054 e-18    J
hartree_joule_relationship = 4.359744650e-18  # 定义哈特里和焦耳的关系

# hartree-kelvin relationship                            3.157 7513 e5         0.000 0018 e5         K
hartree_kelvin_relationship = 3.1577513e5  # 定义哈特里和开尔文的关系

# hartree-kilogram relationship                          4.850 870 129 e-35    0.000 000 060 e-35    kg
hartree_kilogram_relationship = 4.850870129e-35  # 定义哈特里和千克的关系

# helion-electron mass ratio                             5495.885 279 22       0.000 000 27
helion_electron_mass_ratio = 5495.88527922  # 定义氦-电子质量比

# helion g factor                                        -4.255 250 616        0.000 000 050
helion_g_factor = -4.255250616  # 定义氦的g因子

# helion mag. mom.                                       -1.074 617 522 e-26   0.000 000 014 e-26    J T^-1
helion_magnetic_moment = -1.074617522e-26  # 定义氦的磁矩
# Helion magnetic moment to Bohr magneton ratio
-1.158 740 958 e-3    0.000 000 014 e-3

# Helion magnetic moment to nuclear magneton ratio
-2.127 625 308        0.000 000 025

# Helion mass in kilograms
5.006 412 700 e-27    0.000 000 062 e-27    kg

# Helion mass energy equivalent in joules
4.499 539 341 e-10    0.000 000 055 e-10    J

# Helion mass energy equivalent in MeV (Mega electron volts)
2808.391 586          0.000 017             MeV

# Helion mass in unified atomic mass unit (u)
3.014 932 246 73      0.000 000 000 12      u

# Helion molar mass in kilograms per mole (kg mol^-1)
3.014 932 246 73 e-3  0.000 000 000 12 e-3  kg mol^-1

# Helion-proton mass ratio
2.993 152 670 46      0.000 000 000 29

# Relationship between hertz and atomic mass unit (u)
4.439 821 6616 e-24   0.000 000 0020 e-24   u

# Relationship between hertz and electron volt (eV)
4.135 667 662 e-15    0.000 000 025 e-15    eV

# Relationship between hertz and Hartree energy (E_h)
1.5198298460088 e-16  0.0000000000090e-16   E_h

# Relationship between hertz and inverse meter (m^-1)
3.335 640 951... e-9  (exact)               m^-1

# Relationship between hertz and joule (J)
6.626 070 040 e-34    0.000 000 081 e-34    J

# Relationship between hertz and kelvin (K)
4.799 2447 e-11       0.000 0028 e-11       K

# Relationship between hertz and kilogram (kg)
7.372 497 201 e-51    0.000 000 091 e-51    kg

# Inverse of the fine-structure constant
137.035 999 139       0.000 000 031

# Relationship between inverse meter and atomic mass unit (u)
1.331 025 049 00 e-15 0.000 000 000 61 e-15 u

# Relationship between inverse meter and electron volt (eV)
1.239 841 9739 e-6    0.000 000 0076 e-6    eV

# Relationship between inverse meter and Hartree energy (E_h)
4.556 335 252 767 e-8 0.000 000 000 027 e-8 E_h

# Relationship between inverse meter and hertz (Hz)
299 792 458           (exact)               Hz

# Relationship between inverse meter and joule (J)
1.986 445 824 e-25    0.000 000 024 e-25    J

# Relationship between inverse meter and kelvin (K)
1.438 777 36 e-2      0.000 000 83 e-2      K

# Relationship between inverse meter and kilogram (kg)
2.210 219 057 e-42    0.000 000 027 e-42    kg

# Inverse of the conductance quantum
12 906.403 7278       0.000 0029            ohm

# Josephson constant (relation between Hz and V^-1)
483 597.8525 e9       0.0030 e9             Hz V^-1

# Relationship between joule and atomic mass unit (u)
6.700 535 363 e9      0.000 000 082 e9      u

# Relationship between joule and electron volt (eV)
6.241 509 126 e18     0.000 000 038 e18     eV

# Relationship between joule and Hartree energy (E_h)
2.293 712 317 e17     0.000 000 028 e17     E_h

# Relationship between joule and hertz (Hz)
1.509 190 205 e33     0.000 000 019 e33     Hz
# joule-inverse meter relationship                       5.034 116 651 e24     0.000 000 062 e24     m^-1
# 定义变量，表示焦耳和米的倒数的关系，给出了其数值和不确定度，单位为每米^-1
joule_inverse_meter_relationship = 5.034116651e24

# joule-kelvin relationship                              7.242 9731 e22        0.000 0042 e22        K
# 定义变量，表示焦耳和开尔文的关系，给出了其数值和不确定度，单位为开尔文K
joule_kelvin_relationship = 7.2429731e22

# joule-kilogram relationship                            1.112 650 056... e-17 (exact)               kg
# 定义变量，表示焦耳和千克的关系，给出了其数值和确切值，单位为千克kg
joule_kilogram_relationship = 1.112650056e-17

# kelvin-atomic mass unit relationship                   9.251 0842 e-14       0.000 0053 e-14       u
# 定义变量，表示开尔文和原子质量单位的关系，给出了其数值和不确定度，单位为原子质量单位u
kelvin_atomic_mass_unit_relationship = 9.2510842e-14

# kelvin-electron volt relationship                      8.617 3303 e-5        0.000 0050 e-5        eV
# 定义变量，表示开尔文和电子伏特的关系，给出了其数值和不确定度，单位为电子伏特eV
kelvin_electron_volt_relationship = 8.6173303e-5

# kelvin-hartree relationship                            3.166 8105 e-6        0.000 0018 e-6        E_h
# 定义变量，表示开尔文和哈特里的关系，给出了其数值和不确定度，单位为哈特里E_h
kelvin_hartree_relationship = 3.1668105e-6

# kelvin-hertz relationship                              2.083 6612 e10        0.000 0012 e10        Hz
# 定义变量，表示开尔文和赫兹的关系，给出了其数值和不确定度，单位为赫兹Hz
kelvin_hertz_relationship = 2.0836612e10

# kelvin-inverse meter relationship                      69.503 457            0.000 040             m^-1
# 定义变量，表示开尔文和米的倒数的关系，给出了其数值和不确定度，单位为每米^-1
kelvin_inverse_meter_relationship = 69.503457

# kelvin-joule relationship                              1.380 648 52 e-23     0.000 000 79 e-23     J
# 定义变量，表示开尔文和焦耳的关系，给出了其数值和不确定度，单位为焦耳J
kelvin_joule_relationship = 1.38064852e-23

# kelvin-kilogram relationship                           1.536 178 65 e-40     0.000 000 88 e-40     kg
# 定义变量，表示开尔文和千克的关系，给出了其数值和不确定度，单位为千克kg
kelvin_kilogram_relationship = 1.53617865e-40

# kilogram-atomic mass unit relationship                 6.022 140 857 e26     0.000 000 074 e26     u
# 定义变量，表示千克和原子质量单位的关系，给出了其数值和不确定度，单位为原子质量单位u
kilogram_atomic_mass_unit_relationship = 6.022140857e26

# kilogram-electron volt relationship                    5.609 588 650 e35     0.000 000 034 e35     eV
# 定义变量，表示千克和电子伏特的关系，给出了其数值和不确定度，单位为电子伏特eV
kilogram_electron_volt_relationship = 5.609588650e35

# kilogram-hartree relationship                          2.061 485 823 e34     0.000 000 025 e34     E_h
# 定义变量，表示千克和哈特里的关系，给出了其数值和不确定度，单位为哈特里E_h
kilogram_hartree_relationship = 2.061485823e34

# kilogram-hertz relationship                            1.356 392 512 e50     0.000 000 017 e50     Hz
# 定义变量，表示千克和赫兹的关系，给出了其数值和不确定度，单位为赫兹Hz
kilogram_hertz_relationship = 1.356392512e50

# kilogram-inverse meter relationship                    4.524 438 411 e41     0.000 000 056 e41     m^-1
# 定义变量，表示千克和米的倒数的关系，给出了其数值和不确定度，单位为每米^-1
kilogram_inverse_meter_relationship = 4.524438411e41

# kilogram-joule relationship                            8.987 551 787... e16  (exact)               J
# 定义变量，表示千克和焦耳的关系，给出了其确切值，单位为焦耳J
kilogram_joule_relationship = 8.987551787e16

# kilogram-kelvin relationship                           6.509 6595 e39        0.000 0037 e39        K
# 定义变量，表示千克和开尔文的关系，给出了其数值和不确定度，单位为开尔文K
kilogram_kelvin_relationship = 6.5096595e39

# lattice parameter of silicon                           543.102 0504 e-12     0.000 0089 e-12       m
# 定义变量，表示硅的晶格常数，给出了其数值和不确定度，单位为米m
lattice_parameter_of_silicon = 543.1020504e-12

# Loschmidt constant (273.15 K, 100 kPa)                 2.651 6467 e25        0.000 0015 e25        m^-3
# 定义变量，表示洛希米特常数在273.15K和100千帕下的值，给出了其数值和不确定度，单位为每立方米m^-3
loschmidt_constant_100kPa = 2.6516467e25

# Loschmidt constant (273.15 K, 101.325 kPa)             2.686 7811 e25        0.000 0015 e25        m^-3
# 定义变量，表示洛希米特常数在273.15K和101.325千帕下的值，给出了其数值和不确定度，单位为每立方米m^-3
loschmidt_constant_101325kPa = 2.6867811e25

# mag. constant                                          12.566 370 614... e-7 (exact)               N A^-2
# 定义变量，表示磁常数的值，给出了其确切值，单位为牛顿每安培平方N A^-2
magnetic_constant = 12.566370614e-7

# mag. flux quantum                                      2.067 833 831 e-15    0.000 000 013 e-15    Wb
# 定义变量，表示磁通量子的值，给出了其数值和不确定度，单位为韦伯Wb
magnetic_flux_quantum = 2.067833831e-15

# molar gas constant                                     8.314 4598            0.000 0048            J mol^-1 K^-1
# 定义变量，表示摩尔气体常数的值，给出了其数值和不确定度，单位为焦耳每摩尔每开尔文J mol^-1 K^-1
molar_gas
# molar volume of ideal gas (273.15 K, 101.325 kPa)
22.413 962 e-3        0.000 013 e-3         m^3 mol^-1
# molar volume of silicon
12.058 832 14 e-6     0.000 000 61 e-6      m^3 mol^-1
# Mo x unit
1.002 099 52 e-13     0.000 000 53 e-13     m
# muon Compton wavelength
11.734 441 11 e-15    0.000 000 26 e-15     m
# muon Compton wavelength over 2 pi
1.867 594 308 e-15    0.000 000 042 e-15    m
# muon-electron mass ratio
206.768 2826          0.000 0046
# muon g factor
-2.002 331 8418       0.000 000 0013
# muon mag. mom.
-4.490 448 26 e-26    0.000 000 10 e-26     J T^-1
# muon mag. mom. anomaly
1.165 920 89 e-3      0.000 000 63 e-3
# muon mag. mom. to Bohr magneton ratio
-4.841 970 48 e-3     0.000 000 11 e-3
# muon mag. mom. to nuclear magneton ratio
-8.890 597 05         0.000 000 20
# muon mass
1.883 531 594 e-28    0.000 000 048 e-28    kg
# muon mass energy equivalent
1.692 833 774 e-11    0.000 000 043 e-11    J
# muon mass energy equivalent in MeV
105.658 3745          0.000 0024            MeV
# muon mass in u
0.113 428 9257        0.000 000 0025        u
# muon molar mass
0.113 428 9257 e-3    0.000 000 0025 e-3    kg mol^-1
# muon-neutron mass ratio
0.112 454 5167        0.000 000 0025
# muon-proton mag. mom. ratio
-3.183 345 142        0.000 000 071
# muon-proton mass ratio
0.112 609 5262        0.000 000 0025
# muon-tau mass ratio
5.946 49 e-2          0.000 54 e-2
# natural unit of action
1.054 571 800 e-34    0.000 000 013 e-34    J s
# natural unit of action in eV s
6.582 119 514 e-16    0.000 000 040 e-16    eV s
# natural unit of energy
8.187 105 65 e-14     0.000 000 10 e-14     J
# natural unit of energy in MeV
0.510 998 9461        0.000 000 0031        MeV
# natural unit of length
386.159 267 64 e-15   0.000 000 18 e-15     m
# natural unit of mass
9.109 383 56 e-31     0.000 000 11 e-31     kg
# natural unit of mom.um
2.730 924 488 e-22    0.000 000 034 e-22    kg m s^-1
# natural unit of mom.um in MeV/c
0.510 998 9461        0.000 000 0031        MeV/c
# natural unit of time
1.288 088 667 12 e-21 0.000 000 000 58 e-21 s
# 自然单位速度，精确值，单位为米每秒（m s^-1）
natural_unit_of_velocity = 299792458  # exact

# 中子康普顿波长，单位为米（m），精确到第一位小数点后15位，不确定度为第一位小数点后88位
neutron_Compton_wavelength = 1.31959090481e-15  # ± 0.00000000088 m

# 中子康普顿波长除以2π，单位为米（m），精确到第一位小数点后15位，不确定度为第一位小数点后14位
neutron_Compton_wavelength_over_2pi = 0.21001941536e-15  # ± 0.00000000014 m

# 中子-电子磁矩比，无量纲，精确到第一位小数点后3位，不确定度为第一位小数点后3位
neutron_electron_mag_mom_ratio = 1.04066882e-3  # ± 0.00000025e-3

# 中子-电子质量比，无量纲，精确到第一位小数点后9位，不确定度为第一位小数点后6位
neutron_electron_mass_ratio = 1838.68366158  # ± 0.00000090

# 中子g因子，无量纲，精确到第一位小数点后8位，不确定度为第一位小数点后8位
neutron_g_factor = -3.82608545  # ± 0.00000090

# 中子旋磁比，单位为每秒每特斯拉（s^-1 T^-1），精确到第一位小数点后8位，不确定度为第一位小数点后8位
neutron_gyromag_ratio = 1.83247172e8  # ± 0.00000043e8 s^-1 T^-1

# 中子旋磁比除以2π，单位为兆赫每特斯拉（MHz T^-1），精确到第一位小数点后7位，不确定度为第一位小数点后4位
neutron_gyromag_ratio_over_2pi = 29.1646933  # ± 0.0000069 MHz T^-1

# 中子磁矩，单位为焦耳每特斯拉（J T^-1），精确到第一位小数点后10位，不确定度为第一位小数点后10位
neutron_mag_mom = -0.96623650e-26  # ± 0.00000023e-26 J T^-1

# 中子磁矩对玻尔磁子比，无量纲，精确到第一位小数点后3位，不确定度为第一位小数点后3位
neutron_mag_mom_to_Bohr_magneton_ratio = -1.04187563e-3  # ± 0.00000025e-3

# 中子磁矩对核磁子比，无量纲，精确到第一位小数点后8位，不确定度为第一位小数点后8位
neutron_mag_mom_to_nuclear_magneton_ratio = -1.91304273  # ± 0.00000045

# 中子质量，单位为千克（kg），精确到第一位小数点后9位，不确定度为第一位小数点后9位
neutron_mass = 1.674927471e-27  # ± 0.000000021e-27 kg

# 中子质能等效，单位为焦耳（J），精确到第一位小数点后10位，不确定度为第一位小数点后10位
neutron_mass_energy_equivalent = 1.505349739e-10  # ± 0.000000019e-10 J

# 中子质能等效，单位为兆电子伏特（MeV），精确到第一位小数点后7位，不确定度为第一位小数点后5位
neutron_mass_energy_equivalent_in_MeV = 939.5654133  # ± 0.0000058 MeV

# 中子质量，单位为原子质量单位（u），精确到第一位小数点后12位，不确定度为第一位小数点后11位
neutron_mass_in_u = 1.00866491588  # ± 0.00000000049 u

# 中子摩尔质量，单位为千克每摩尔（kg mol^-1），精确到第一位小数点后12位，不确定度为第一位小数点后12位
neutron_molar_mass = 1.00866491588e-3  # ± 0.00000000049e-3 kg mol^-1

# 中子-μ子质量比，无量纲，精确到第一位小数点后8位，不确定度为第一位小数点后2位
neutron_muon_mass_ratio = 8.89248408  # ± 0.00000020

# 中子-质子磁矩比，无量纲，精确到第一位小数点后8位，不确定度为第一位小数点后8位
neutron_proton_mag_mom_ratio = -0.68497934  # ± 0.00000016

# 中子-质子质量差，单位为千克（kg），精确到第一位小数点后9位，不确定度为第一位小数点后9位
neutron_proton_mass_difference = 2.30557377e-30  # ± 0.00000085e-30 kg

# 中子-质子质能等效，单位为焦耳（J），精确到第一位小数点后10位，不确定度为第一位小数点后10位
neutron_proton_mass_difference_energy_equivalent = 2.07214637e-13  # ± 0.00000076e-13 J

# 中子-质子质能等效，单位为兆电子伏特（MeV），精确到第一位小数点后8位，不确定度为第一位小数点后8位
neutron_proton_mass_difference_energy_equivalent_in_MeV = 1.29333205  # ± 0.00000048 MeV

# 中子-质子质量差，单位为原子质量单位（u），精确到第一位小数点后11位，不确定度为第一位小数点后9位
neutron_proton_mass_difference_in_u = 0.00138844900  # ± 0.00000000051 u

# 中子-质子质量比，无量纲，精确到第一位小数点后12位，不确定度为第一位小数点后11位
neutron_proton_mass_ratio = 1.00137841898  # ± 0.00000000051

# 中子-τ子质量比，无量纲，精确到第一位小数点后6位，不确定度为第一位小数点后3位
neutron_tau_mass_ratio = 0.528790  # ± 0.000048

# 中子对受屏蔽质子磁矩比，无量纲，精确到第一位小数点后8位，不确定度为第一位小数点后8位
neutron_to_shielded_proton_mag_mom_ratio = -0.68499694  # ± 0.00000016

# 牛顿引力常数，单位为立方米每千克每平方秒（m^3 kg^-1 s^-2），精确到第一位小数点后8位，不确定度为第一位小数点后2位
Newtonian_constant_of_gravitation = 6.67408e-11  # ± 0.00031e-11 m^3 kg^-1 s^-2

# 牛顿引力常数除以约化普朗克常数乘以光速，单位为（GeV/c^2）
# 定义核磁子在开尔文每特斯拉下的数值和误差范围，单位为 K T^-1
nuclear magneton in K/T                                3.658 2690 e-4        0.000 0021 e-4        K T^-1
# 定义核磁子在兆赫每特斯拉下的数值和误差范围，单位为 MHz T^-1
nuclear magneton in MHz/T                              7.622 593 285         0.000 000 047         MHz T^-1
# 定义普朗克常数的数值和误差范围，单位为焦耳秒
Planck constant                                        6.626 070 040 e-34    0.000 000 081 e-34    J s
# 定义普朗克常数的数值和误差范围，单位为电子伏秒
Planck constant in eV s                                4.135 667 662 e-15    0.000 000 025 e-15    eV s
# 定义2π除以普朗克常数的数值和误差范围，单位为焦耳秒
Planck constant over 2 pi                              1.054 571 800 e-34    0.000 000 013 e-34    J s
# 定义2π除以普朗克常数的数值和误差范围，单位为电子伏秒
Planck constant over 2 pi in eV s                      6.582 119 514 e-16    0.000 000 040 e-16    eV s
# 定义2π除以普朗克常数乘以光速的数值和误差范围，单位为兆电子伏特·费米
Planck constant over 2 pi times c in MeV fm            197.326 9788          0.000 0012            MeV fm
# 定义普朗克长度的数值和误差范围，单位为米
Planck length                                          1.616 229 e-35        0.000 038 e-35        m
# 定义普朗克质量的数值和误差范围，单位为千克
Planck mass                                            2.176 470 e-8         0.000 051 e-8         kg
# 定义普朗克质量的能量等效值的数值和误差范围，单位为吉电子伏特
Planck mass energy equivalent in GeV                   1.220 910 e19         0.000 029 e19         GeV
# 定义普朗克温度的数值和误差范围，单位为开尔文
Planck temperature                                     1.416 808 e32         0.000 033 e32         K
# 定义普朗克时间的数值和误差范围，单位为秒
Planck time                                            5.391 16 e-44         0.000 13 e-44         s
# 定义质子电荷与质量比的数值和误差范围，单位为库仑千克^-1
proton charge to mass quotient                         9.578 833 226 e7      0.000 000 059 e7      C kg^-1
# 定义质子康普顿波长的数值和误差范围，单位为米
proton Compton wavelength                              1.321 409 853 96 e-15 0.000 000 000 61 e-15 m
# 定义2π除以质子康普顿波长的数值和误差范围，单位为米
proton Compton wavelength over 2 pi                    0.210 308910109e-15   0.000 000 000097e-15  m
# 定义质子-电子质量比的数值和误差范围
proton-electron mass ratio                             1836.152 673 89       0.000 000 17
# 定义质子 g 因子的数值和误差范围
proton g factor                                        5.585 694 702         0.000 000 017
# 定义质子旋磁比的数值和误差范围，单位为秒^-1·特斯拉^-1
proton gyromag. ratio                                  2.675 221 900 e8      0.000 000 018 e8      s^-1 T^-1
# 定义2π除以质子旋磁比的数值和误差范围，单位为兆赫特斯拉^-1
proton gyromag. ratio over 2 pi                        42.577 478 92         0.000 000 29          MHz T^-1
# 定义质子磁矩的数值和误差范围，单位为焦耳·特斯拉^-1
proton mag. mom.                                       1.410 606 7873 e-26   0.000 000 0097 e-26   J T^-1
# 定义质子磁矩与玻尔磁子比值的数值和误差范围
proton mag. mom. to Bohr magneton ratio                1.521 032 2053 e-3    0.000 000 0046 e-3
# 定义质子磁矩与核磁子比值的数值和误差范围
proton mag. mom. to nuclear magneton ratio             2.792 847 3508        0.000 000 0085
# 定义质子磁屏蔽修正的数值和误差范围
proton mag. shielding correction                       25.691 e-6            0.011 e-6
# 定义质子质量的数值和误差范围，单位为千克
proton mass                                            1.672 621 898 e-27    0.000 000 021 e-27    kg
# 定义质子质量的能量等效值的数值和误差范围，单位为焦耳
proton mass energy equivalent                          1.503 277 593 e-10    0.000 000 018 e-10    J
# 定义质子质量的能量等效值的数值和误差范围，单位为兆电子伏特
proton mass energy equivalent in MeV                   938.272 0813          0.000 0058            MeV
# 定义质子质量的数值和误差范围，单位为原子单位
proton mass in u                                       1.007 276 466 879     0.000 000 000 091     u
# 定义质子摩尔质量的数值和误差范围，单位为千克·摩尔^-1
proton molar mass                                      1.007 276 466 879 e-3 0.000 000 000 091 e-3 kg mol^-1
# 定义质子-μ子质量比的数值和误差范围
proton-muon mass ratio                                 8.880 243 38          0.000 000 20
# 质子-中子磁矩比
proton-neutron mag. mom. ratio                         -1.459 898 05         0.000 000 34

# 质子-中子质量比
proton-neutron mass ratio                              0.998 623 478 44      0.000 000 000 51

# 质子的有效半径的均方根电荷半径
proton rms charge radius                               0.8751 e-15           0.0061 e-15           m

# 质子-τ子质量比
proton-tau mass ratio                                  0.528 063             0.000 048

# 循环量子
quantum of circulation                                 3.636 947 5486 e-4    0.000 000 0017 e-4    m^2 s^-1

# 循环量子的两倍
quantum of circulation times 2                         7.273 895 0972 e-4    0.000 000 0033 e-4    m^2 s^-1

# 瑞德堡常数
Rydberg constant                                       10 973 731.568 508    0.000 065             m^-1

# 瑞德堡常数乘以光速的频率
Rydberg constant times c in Hz                         3.289 841 960 355 e15 0.000 000 000 019 e15 Hz

# 瑞德堡常数乘以普朗克常数和光速的电子伏特单位
Rydberg constant times hc in eV                        13.605 693 009        0.000 000 084         eV

# 瑞德堡常数乘以普朗克常数和光速的焦耳单位
Rydberg constant times hc in J                         2.179 872 325 e-18    0.000 000 027 e-18    J

# 萨克尔-特特罗德常数（1 K，100 kPa）
Sackur-Tetrode constant (1 K, 100 kPa)                 -1.151 7084           0.000 0014

# 萨克尔-特特罗德常数（1 K，101.325 kPa）
Sackur-Tetrode constant (1 K, 101.325 kPa)             -1.164 8714           0.000 0014

# 第二辐射常数
second radiation constant                              1.438 777 36 e-2      0.000 000 83 e-2      m K

# 屏蔽中子的旋磁比
shielded helion gyromag. ratio                         2.037 894 585 e8      0.000 000 027 e8      s^-1 T^-1

# 屏蔽中子的旋磁比除以2π
shielded helion gyromag. ratio over 2 pi               32.434 099 66         0.000 000 43          MHz T^-1

# 屏蔽中子的磁矩
shielded helion mag. mom.                              -1.074 553 080 e-26   0.000 000 014 e-26    J T^-1

# 屏蔽中子的磁矩与玻尔磁子比值
shielded helion mag. mom. to Bohr magneton ratio       -1.158 671 471 e-3    0.000 000 014 e-3

# 屏蔽中子的磁矩与核磁子比值
shielded helion mag. mom. to nuclear magneton ratio    -2.127 497 720        0.000 000 025

# 屏蔽中子与质子的磁矩比值
shielded helion to proton mag. mom. ratio              -0.761 766 5603       0.000 000 0092

# 屏蔽中子与屏蔽质子的磁矩比值
shielded helion to shielded proton mag. mom. ratio     -0.761 786 1313       0.000 000 0033

# 屏蔽质子的旋磁比
shielded proton gyromag. ratio                         2.675 153 171 e8      0.000 000 033 e8      s^-1 T^-1

# 屏蔽质子的旋磁比除以2π
shielded proton gyromag. ratio over 2 pi               42.576 385 07         0.000 000 53          MHz T^-1

# 屏蔽质子的磁矩
shielded proton mag. mom.                              1.410 570 547 e-26    0.000 000 018 e-26    J T^-1

# 屏蔽质子的磁矩与玻尔磁子比值
shielded proton mag. mom. to Bohr magneton ratio       1.520 993 128 e-3     0.000 000 017 e-3

# 屏蔽质子的磁矩与核磁子比值
shielded proton mag. mom. to nuclear magneton ratio    2.792 775 600         0.000 000 030

# 真空中的光速
speed of light in vacuum                               299 792 458           (exact)               m s^-1

# 标准重力加速度
standard acceleration of gravity                       9.806 65              (exact)               m s^-2

# 标准大气压
standard atmosphere                                    101 325               (exact)               Pa

# 标准状态压强
standard-state pressure                                100 000               (exact)               Pa
Stefan-Boltzmann constant                              5.670 367 e-8         0.000 013 e-8         W m^-2 K^-4
tau Compton wavelength                                 0.697 787 e-15        0.000 063 e-15        m
tau Compton wavelength over 2 pi                       0.111 056 e-15        0.000 010 e-15        m
tau-electron mass ratio                                3477.15               0.31
tau mass                                               3.167 47 e-27         0.000 29 e-27         kg
tau mass energy equivalent                             2.846 78 e-10         0.000 26 e-10         J
tau mass energy equivalent in MeV                      1776.82               0.16                  MeV
tau mass in u                                          1.907 49              0.000 17              u
tau molar mass                                         1.907 49 e-3          0.000 17 e-3          kg mol^-1
tau-muon mass ratio                                    16.8167               0.0015
tau-neutron mass ratio                                 1.891 11              0.000 17
tau-proton mass ratio                                  1.893 72              0.000 17
Thomson cross section                                  0.665 245 871 58 e-28 0.000 000 000 91 e-28 m^2
triton-electron mass ratio                             5496.921 535 88       0.000 000 26
triton g factor                                        5.957 924 920         0.000 000 028
triton mag. mom.                                       1.504 609 503 e-26    0.000 000 012 e-26    J T^-1
triton mag. mom. to Bohr magneton ratio                1.622 393 6616 e-3    0.000 000 0076 e-3
triton mag. mom. to nuclear magneton ratio             2.978 962 460         0.000 000 014
triton mass                                            5.007 356 665 e-27    0.000 000 062 e-27    kg
triton mass energy equivalent                          4.500 387 735 e-10    0.000 000 055 e-10    J
triton mass energy equivalent in MeV                   2808.921 112          0.000 017             MeV
triton mass in u                                       3.015 500 716 32      0.000 000 000 11      u
triton molar mass                                      3.015 500 716 32 e-3  0.000 000 000 11 e-3  kg mol^-1
triton-proton mass ratio                               2.993 717 033 48      0.000 000 000 22
unified atomic mass unit                               1.660 539 040 e-27    0.000 000 020 e-27    kg
von Klitzing constant                                  25 812.807 4555       0.000 0059            ohm
weak mixing angle                                      0.2223                0.0021
Wien frequency displacement law constant               5.878 9238 e10        0.000 0034 e10        Hz K^-1
Wien wavelength displacement law constant              2.897 7729 e-3        0.000 0017 e-3        m K



txt2018 = """\
alpha particle-electron mass ratio                          7294.299 541 42          0.000 000 24
# alpha particle mass
alpha_particle_mass = 6.6446573357e-27  # Alpha粒子的质量，单位为千克
alpha_particle_mass_uncertainty = 0.0000000020e-27  # Alpha粒子质量的不确定性，单位为千克

# alpha particle mass energy equivalent
alpha_particle_energy_equivalent = 5.9719201914e-10  # Alpha粒子的能量等效，单位为焦耳
alpha_particle_energy_equivalent_uncertainty = 0.0000000018e-10  # Alpha粒子能量等效的不确定性，单位为焦耳

# alpha particle mass energy equivalent in MeV
alpha_particle_energy_equivalent_MeV = 3727.3794066  # Alpha粒子的能量等效，单位为兆电子伏特(MeV)
alpha_particle_energy_equivalent_MeV_uncertainty = 0.0000011  # Alpha粒子能量等效的不确定性，单位为兆电子伏特(MeV)

# alpha particle mass in u
alpha_particle_mass_u = 4.001506179127  # Alpha粒子的质量，单位为原子质量单位(u)
alpha_particle_mass_u_uncertainty = 0.000000000063  # Alpha粒子质量的不确定性，单位为原子质量单位(u)

# alpha particle molar mass
alpha_particle_molar_mass = 4.0015061777e-3  # Alpha粒子的摩尔质量，单位为千克每摩尔(kg/mol)
alpha_particle_molar_mass_uncertainty = 0.0000000012e-3  # Alpha粒子摩尔质量的不确定性，单位为千克每摩尔(kg/mol)

# alpha particle-proton mass ratio
alpha_particle_proton_mass_ratio = 3.97259969009  # Alpha粒子与质子质量比
alpha_particle_proton_mass_ratio_uncertainty = 0.00000000022  # Alpha粒子与质子质量比的不确定性

# alpha particle relative atomic mass
alpha_particle_relative_atomic_mass = 4.001506179127  # Alpha粒子的相对原子质量
alpha_particle_relative_atomic_mass_uncertainty = 0.000000000063  # Alpha粒子相对原子质量的不确定性

# Angstrom star
angstrom_star = 1.00001495e-10  # 埃仑斯特朗常数，单位为米
angstrom_star_uncertainty = 0.00000090e-10  # 埃仑斯特朗常数的不确定性，单位为米

# atomic mass constant
atomic_mass_constant = 1.66053906660e-27  # 原子质量常数，单位为千克
atomic_mass_constant_uncertainty = 0.00000000050e-27  # 原子质量常数的不确定性，单位为千克

# atomic mass constant energy equivalent
atomic_mass_constant_energy_equivalent = 1.49241808560e-10  # 原子质量常数的能量等效，单位为焦耳
atomic_mass_constant_energy_equivalent_uncertainty = 0.00000000045e-10  # 原子质量常数能量等效的不确定性，单位为焦耳

# atomic mass constant energy equivalent in MeV
atomic_mass_constant_energy_equivalent_MeV = 931.49410242  # 原子质量常数的能量等效，单位为兆电子伏特(MeV)
atomic_mass_constant_energy_equivalent_MeV_uncertainty = 0.00000028  # 原子质量常数能量等效的不确定性，单位为兆电子伏特(MeV)

# atomic mass unit-electron volt relationship
atomic_mass_unit_eV_relationship = 9.3149410242e8  # 原子质量单位与电子伏特的关系，单位为电子伏特(eV)
atomic_mass_unit_eV_relationship_uncertainty = 0.0000000028e8  # 原子质量单位与电子伏特关系的不确定性，单位为电子伏特(eV)

# atomic mass unit-hartree relationship
atomic_mass_unit_Hartree_relationship = 3.4231776874e7  # 原子质量单位与哈特里的关系，单位为哈特里(E_h)
atomic_mass_unit_Hartree_relationship_uncertainty = 0.0000000010e7  # 原子质量单位与哈特里关系的不确定性，单位为哈特里(E_h)

# atomic mass unit-hertz relationship
atomic_mass_unit_Hz_relationship = 2.25234271871e23  # 原子质量单位与赫兹的关系，单位为赫兹(Hz)
atomic_mass_unit_Hz_relationship_uncertainty = 0.00000000068e23  # 原子质量单位与赫兹关系的不确定性，单位为赫兹(Hz)

# atomic mass unit-inverse meter relationship
atomic_mass_unit_inverse_meter_relationship = 7.5130066104e14  # 原子质量单位与米的倒数的关系，单位为米的倒数(m^-1)
atomic_mass_unit_inverse_meter_relationship_uncertainty = 0.0000000023e14  # 原子质量单位与米的倒数关系的不确定性，单位为米的倒数(m^-1)

# atomic mass unit-joule relationship
atomic_mass_unit_Joule_relationship = 1.49241808560e-10  # 原子质量单位与焦耳的关系，单位为焦耳(J)
atomic_mass_unit_Joule_relationship_uncertainty = 0.00000000045e-10  # 原子质量单位与焦耳关系的不确定性，单位为焦耳(J)

# atomic mass unit-kelvin relationship
atomic_mass_unit_Kelvin_relationship = 1.08095401916e13  # 原子质量单位与开尔文的关系，单位为开尔文(K)
atomic_mass_unit_Kelvin_relationship_uncertainty = 0.00000000033e13  # 原子质量单位与开尔文关系的不确定性，单位为开尔文(K)

# atomic mass unit-kilogram relationship
atomic_mass_unit_kg_relationship = 1.66053906660e-27  # 原子质量单位与千克的关系，单位为千克(kg)
atomic_mass_unit_kg_relationship_uncertainty = 0.00000000050e-27  # 原子质量单位与千克关系的不确定性，单位为千克(kg)

# atomic unit of 1st hyperpolarizability
atomic_unit_1st_hyperpolarizability = 3.2063613061e-53  # 第一超极化率的原子单位，单位为库伦立方米的平方焦耳的倒数(C^3 m^3 J^-2)
atomic_unit_1st_hyperpolarizability_uncertainty = 0.0000000015e-53  # 第一超极化率的原子单位的不确定性，单位为库伦立方米的平方焦耳的倒数(C^3 m^3 J^-2)

# atomic unit of 2nd hyperpolarizability
atomic_unit_2nd_hyperpolarizability = 6.2353799905e-65  # 第二超极化率的原子单位，单位为库伦的四次方米的平方焦耳的倒数(C^4 m^4 J^-3)
atomic_unit_2nd_hyperpolarizability_uncertainty = 0.0000000038e-65  # 第二超极化率的原子单位的不确定性，单位为库伦的四次方米的平方焦耳的倒数(C^4 m^4 J^-3)

# atomic unit of action
atomic_unit_action = 1.054571817e-34  # 动作量子的原子单位，单位为焦耳秒(J s)

# atomic unit of charge
atomic_unit_charge = 1.602176634e-19  # 电荷的原子单位，单位为库伦(C)

# atomic unit of charge density
atomic_unit_charge_density = 1.08120238457e12  # 电荷密度的原子
# 原子单位电场梯度
atomic unit of electric field gradient                      9.717 362 4292 e21       0.000 000 0029 e21       V m^-2

# 原子单位电极化率
atomic unit of electric polarizability                      1.648 777 274 36 e-41    0.000 000 000 50 e-41    C^2 m^2 J^-1

# 原子单位电势
atomic unit of electric potential                           27.211 386 245 988       0.000 000 000 053        V

# 原子单位电四极矩
atomic unit of electric quadrupole mom.                     4.486 551 5246 e-40      0.000 000 0014 e-40      C m^2

# 原子单位能量
atomic unit of energy                                       4.359 744 722 2071 e-18  0.000 000 000 0085 e-18  J

# 原子单位力
atomic unit of force                                        8.238 723 4983 e-8       0.000 000 0012 e-8       N

# 原子单位长度
atomic unit of length                                       5.291 772 109 03 e-11    0.000 000 000 80 e-11    m

# 原子单位磁偶极矩
atomic unit of mag. dipole mom.                             1.854 802 015 66 e-23    0.000 000 000 56 e-23    J T^-1

# 原子单位磁通密度
atomic unit of mag. flux density                            2.350 517 567 58 e5      0.000 000 000 71 e5      T

# 原子单位磁化率
atomic unit of magnetizability                              7.891 036 6008 e-29      0.000 000 0048 e-29      J T^-2

# 原子单位质量
atomic unit of mass                                         9.109 383 7015 e-31      0.000 000 0028 e-31      kg

# 原子单位动量
atomic unit of momentum                                     1.992 851 914 10 e-24    0.000 000 000 30 e-24    kg m s^-1

# 原子单位介电常数
atomic unit of permittivity                                 1.112 650 055 45 e-10    0.000 000 000 17 e-10    F m^-1

# 原子单位时间
atomic unit of time                                         2.418 884 326 5857 e-17  0.000 000 000 0047 e-17  s

# 原子单位速度
atomic unit of velocity                                     2.187 691 263 64 e6      0.000 000 000 33 e6      m s^-1

# 阿伏伽德罗常数
Avogadro constant                                           6.022 140 76 e23         (exact)                  mol^-1

# 玻尔磁子
Bohr magneton                                               9.274 010 0783 e-24      0.000 000 0028 e-24      J T^-1

# 玻尔磁子（以电子伏特和特斯拉为单位）
Bohr magneton in eV/T                                       5.788 381 8060 e-5       0.000 000 0017 e-5       eV T^-1

# 玻尔磁子（以赫兹和特斯拉为单位）
Bohr magneton in Hz/T                                       1.399 624 493 61 e10     0.000 000 000 42 e10     Hz T^-1

# 玻尔磁子（以米的逆和特斯拉为单位）
Bohr magneton in inverse meter per tesla                    46.686 447 783           0.000 000 014            m^-1 T^-1

# 玻尔磁子（以开尔文和特斯拉为单位）
Bohr magneton in K/T                                        0.671 713 815 63         0.000 000 000 20         K T^-1

# 玻尔半径
Bohr radius                                                 5.291 772 109 03 e-11    0.000 000 000 80 e-11    m

# 玻尔兹曼常数
Boltzmann constant                                          1.380 649 e-23           (exact)                  J K^-1

# 玻尔兹曼常数（以电子伏特和开尔文为单位）
Boltzmann constant in eV/K                                  8.617 333 262... e-5     (exact)                  eV K^-1

# 玻尔兹曼常数（以赫兹和开尔文为单位）
Boltzmann constant in Hz/K                                  2.083 661 912... e10     (exact)                  Hz K^-1
# Boltzmann constant in inverse meter per kelvin
69.503 480 04...         (exact)                  m^-1 K^-1
# characteristic impedance of vacuum
376.730 313 668          0.000 000 057            ohm
# classical electron radius
2.817 940 3262 e-15      0.000 000 0013 e-15      m
# Compton wavelength
2.426 310 238 67 e-12    0.000 000 000 73 e-12    m
# conductance quantum
7.748 091 729... e-5     (exact)                  S
# conventional value of ampere-90
1.000 000 088 87...      (exact)                  A
# conventional value of coulomb-90
1.000 000 088 87...      (exact)                  C
# conventional value of farad-90
0.999 999 982 20...      (exact)                  F
# conventional value of henry-90
1.000 000 017 79...      (exact)                  H
# conventional value of Josephson constant
483 597.9 e9             (exact)                  Hz V^-1
# conventional value of ohm-90
1.000 000 017 79...      (exact)                  ohm
# conventional value of volt-90
1.000 000 106 66...      (exact)                  V
# conventional value of von Klitzing constant
25 812.807               (exact)                  ohm
# conventional value of watt-90
1.000 000 195 53...      (exact)                  W
# Cu x unit
1.002 076 97 e-13        0.000 000 28 e-13        m
# deuteron-electron mag. mom. ratio
-4.664 345 551 e-4       0.000 000 012 e-4
# deuteron-electron mass ratio
3670.482 967 88          0.000 000 13
# deuteron g factor
0.857 438 2338           0.000 000 0022
# deuteron mag. mom.
4.330 735 094 e-27       0.000 000 011 e-27       J T^-1
# deuteron mag. mom. to Bohr magneton ratio
4.669 754 570 e-4        0.000 000 012 e-4
# deuteron mag. mom. to nuclear magneton ratio
0.857 438 2338           0.000 000 0022
# deuteron mass
3.343 583 7724 e-27      0.000 000 0010 e-27      kg
# deuteron mass energy equivalent
3.005 063 231 02 e-10    0.000 000 000 91 e-10    J
# deuteron mass energy equivalent in MeV
1875.612 942 57          0.000 000 57             MeV
# deuteron mass in u
2.013 553 212 745        0.000 000 000 040        u
# deuteron molar mass
2.013 553 212 05 e-3     0.000 000 000 61 e-3     kg mol^-1
# deuteron-neutron mag. mom. ratio
-0.448 206 53            0.000 000 11
# deuteron-proton mag. mom. ratio
# 质子-质子的磁矩比
0.307 012 209 39         0.000 000 000 79

# deuteron-proton mass ratio
# 质子-质子的质量比
1.999 007 501 39         0.000 000 000 11

# deuteron relative atomic mass
# 氘的相对原子质量
2.013 553 212 745        0.000 000 000 040

# deuteron rms charge radius
# 氘的均方根电荷半径
2.127 99 e-15            0.000 74 e-15            m

# electron charge to mass quotient
# 电子电荷与质量的比值
-1.758 820 010 76 e11    0.000 000 000 53 e11     C kg^-1

# electron-deuteron mag. mom. ratio
# 电子-氘的磁矩比
-2143.923 4915           0.000 0056

# electron-deuteron mass ratio
# 电子-氘的质量比
2.724 437 107 462 e-4    0.000 000 000 096 e-4

# electron g factor
# 电子的g因子
-2.002 319 304 362 56    0.000 000 000 000 35

# electron gyromag. ratio
# 电子的旋磁比
1.760 859 630 23 e11     0.000 000 000 53 e11     s^-1 T^-1

# electron gyromag. ratio in MHz/T
# 电子的旋磁比（以MHz/T为单位）
28 024.951 4242          0.000 0085               MHz T^-1

# electron-helion mass ratio
# 电子-氦-3的质量比
1.819 543 074 573 e-4    0.000 000 000 079 e-4

# electron mag. mom.
# 电子的磁矩
-9.284 764 7043 e-24     0.000 000 0028 e-24      J T^-1

# electron mag. mom. anomaly
# 电子的磁矩异常
1.159 652 181 28 e-3     0.000 000 000 18 e-3

# electron mag. mom. to Bohr magneton ratio
# 电子磁矩与玻尔磁子的比值
-1.001 159 652 181 28    0.000 000 000 000 18

# electron mag. mom. to nuclear magneton ratio
# 电子磁矩与核磁子的比值
-1838.281 971 88         0.000 000 11

# electron mass
# 电子的质量
9.109 383 7015 e-31      0.000 000 0028 e-31      kg

# electron mass energy equivalent
# 电子的质能关系
8.187 105 7769 e-14      0.000 000 0025 e-14      J

# electron mass energy equivalent in MeV
# 电子的质能关系（以兆电子伏为单位）
0.510 998 950 00         0.000 000 000 15         MeV

# electron mass in u
# 电子的质量（以原子单位为单位）
5.485 799 090 65 e-4     0.000 000 000 16 e-4     u

# electron molar mass
# 电子的摩尔质量
5.485 799 0888 e-7       0.000 000 0017 e-7       kg mol^-1

# electron-muon mag. mom. ratio
# 电子-μ子的磁矩比
206.766 9883             0.000 0046

# electron-muon mass ratio
# 电子-μ子的质量比
4.836 331 69 e-3         0.000 000 11 e-3

# electron-neutron mag. mom. ratio
# 电子-中子的磁矩比
960.920 50               0.000 23

# electron-neutron mass ratio
# 电子-中子的质量比
5.438 673 4424 e-4       0.000 000 0026 e-4

# electron-proton mag. mom. ratio
# 电子-质子的磁矩比
-658.210 687 89          0.000 000 20

# electron-proton mass ratio
# 电子-质子的质量比
5.446 170 214 87 e-4     0.000 000 000 33 e-4

# electron relative atomic mass
# 电子的相对原子质量
5.485 799 090 65 e-4     0.000 000 000 16 e-4

# electron-tau mass ratio
# 电子-τ子的质量比
2.875 85 e-4             0.000 19 e-4
# 电子到α粒子质量比
electron to alpha particle mass ratio                       1.370 933 554 787 e-4    0.000 000 000 045 e-4
# 电子到屏蔽氦离子磁矩比
electron to shielded helion mag. mom. ratio                 864.058 257              0.000 010
# 电子到屏蔽质子磁矩比
electron to shielded proton mag. mom. ratio                 -658.227 5971            0.000 0072
# 电子到氚质量比
electron-triton mass ratio                                  1.819 200 062 251 e-4    0.000 000 000 090 e-4
# 电子伏特
electron volt                                               1.602 176 634 e-19       (exact)                  J
# 电子伏特与原子质量单位的关系
electron volt-atomic mass unit relationship                 1.073 544 102 33 e-9     0.000 000 000 32 e-9     u
# 电子伏特与哈特里的关系
electron volt-hartree relationship                          3.674 932 217 5655 e-2   0.000 000 000 0071 e-2   E_h
# 电子伏特与赫兹的关系
electron volt-hertz relationship                            2.417 989 242... e14     (exact)                  Hz
# 电子伏特与每米逆关系
electron volt-inverse meter relationship                    8.065 543 937... e5      (exact)                  m^-1
# 电子伏特与焦耳的关系
electron volt-joule relationship                            1.602 176 634 e-19       (exact)                  J
# 电子伏特与开尔文的关系
electron volt-kelvin relationship                           1.160 451 812... e4      (exact)                  K
# 电子伏特与千克的关系
electron volt-kilogram relationship                         1.782 661 921... e-36    (exact)                  kg
# 元电荷
elementary charge                                           1.602 176 634 e-19       (exact)                  C
# 元电荷与约化普朗克常数的比值
elementary charge over h-bar                                1.519 267 447... e15     (exact)                  A J^-1
# 法拉第常数
Faraday constant                                            96 485.332 12...         (exact)                  C mol^-1
# 费米耦合常数
Fermi coupling constant                                     1.166 3787 e-5           0.000 0006 e-5           GeV^-2
# 精细结构常数
fine-structure constant                                     7.297 352 5693 e-3       0.000 000 0011 e-3
# 第一辐射常数
first radiation constant                                    3.741 771 852... e-16    (exact)                  W m^2
# 用于光谱辐射度的第一辐射常数
first radiation constant for spectral radiance              1.191 042 972... e-16    (exact)                  W m^2 sr^-1
# 哈特里与原子质量单位的关系
hartree-atomic mass unit relationship                       2.921 262 322 05 e-8     0.000 000 000 88 e-8     u
# 哈特里与电子伏特的关系
hartree-electron volt relationship                          27.211 386 245 988       0.000 000 000 053        eV
# 哈特里能量
Hartree energy                                              4.359 744 722 2071 e-18  0.000 000 000 0085 e-18  J
# 哈特里能量的电子伏特单位
Hartree energy in eV                                        27.211 386 245 988       0.000 000 000 053        eV
# 哈特里与赫兹的关系
hartree-hertz relationship                                  6.579 683 920 502 e15    0.000 000 000 013 e15    Hz
# 哈特里与每米逆关系
hartree-inverse meter relationship                          2.194 746 313 6320 e7    0.000 000 000 0043 e7    m^-1
# 哈特里与焦耳的关系
hartree-joule relationship                                  4.359 744 722 2071 e-18  0.000 000 000 0085 e-18  J
# hartree-kelvin relationship                                 3.157 750 248 0407 e5    0.000 000 000 0061 e5    K
# 定义哈特里-开尔文关系，第一个数字是值，第二个数字是不确定度，单位为开尔文
hartree_kelvin_relationship = 3.1577502480407e5

# hartree-kilogram relationship                               4.850 870 209 5432 e-35  0.000 000 000 0094 e-35  kg
# 定义哈特里-千克关系，第一个数字是值，第二个数字是不确定度，单位为千克
hartree_kilogram_relationship = 4.8508702095432e-35

# helion-electron mass ratio                                  5495.885 280 07          0.000 000 24
# 定义质子-电子质量比，第一个数字是值，第二个数字是不确定度
helion_electron_mass_ratio = 5495.88528007

# helion g factor                                             -4.255 250 615           0.000 000 050
# 定义质子 g 因子，第一个数字是值，第二个数字是不确定度
helion_g_factor = -4.255250615

# helion mag. mom.                                            -1.074 617 532 e-26      0.000 000 013 e-26       J T^-1
# 定义质子磁矩，第一个数字是值，第二个数字是不确定度，单位为焦耳·特斯拉^-1
helion_mag_mom = -1.074617532e-26

# helion mag. mom. to Bohr magneton ratio                     -1.158 740 958 e-3       0.000 000 014 e-3
# 定义质子磁矩对玻尔磁子比值，第一个数字是值，第二个数字是不确定度
helion_mag_mom_to_bohr_magneton_ratio = -1.158740958e-3

# helion mag. mom. to nuclear magneton ratio                  -2.127 625 307           0.000 000 025
# 定义质子磁矩对核磁子比值，第一个数字是值，第二个数字是不确定度
helion_mag_mom_to_nuclear_magneton_ratio = -2.127625307

# helion mass                                                 5.006 412 7796 e-27      0.000 000 0015 e-27      kg
# 定义质子质量，第一个数字是值，第二个数字是不确定度，单位为千克
helion_mass = 5.0064127796e-27

# helion mass energy equivalent                               4.499 539 4125 e-10      0.000 000 0014 e-10      J
# 定义质子质量能量等效值，第一个数字是值，第二个数字是不确定度，单位为焦耳
helion_mass_energy_equivalent = 4.4995394125e-10

# helion mass energy equivalent in MeV                        2808.391 607 43          0.000 000 85             MeV
# 定义质子质量能量等效值（以兆电子伏特为单位），第一个数字是值，第二个数字是不确定度，单位为兆电子伏特
helion_mass_energy_equivalent_mev = 2808.39160743

# helion mass in u                                            3.014 932 247 175        0.000 000 000 097        u
# 定义质子质量（以原子单位为单位），第一个数字是值，第二个数字是不确定度，单位为原子单位
helion_mass_in_u = 3.014932247175

# helion molar mass                                           3.014 932 246 13 e-3     0.000 000 000 91 e-3     kg mol^-1
# 定义质子的摩尔质量，第一个数字是值，第二个数字是不确定度，单位为千克·摩尔^-1
helion_molar_mass = 3.01493224613e-3

# helion-proton mass ratio                                    2.993 152 671 67         0.000 000 000 13
# 定义质子-质子质量比，第一个数字是值，第二个数字是不确定度
helion_proton_mass_ratio = 2.99315267167

# helion relative atomic mass                                 3.014 932 247 175        0.000 000 000 097
# 定义质子的相对原子质量，第一个数字是值，第二个数字是不确定度
helion_relative_atomic_mass = 3.014932247175

# helion shielding shift                                      5.996 743 e-5            0.000 010 e-5
# 定义质子的屏蔽偏移，第一个数字是值，第二个数字是不确定度
helion_shielding_shift = 5.996743e-5

# hertz-atomic mass unit relationship                         4.439 821 6652 e-24      0.000 000 0013 e-24      u
# 定义赫兹-原子质量单位关系，第一个数字是值，第二个数字是不确定度，单位为原子质量单位
hertz_atomic_mass_unit_relationship = 4.4398216652e-24

# hertz-electron volt relationship                            4.135 667 696... e-15    (exact)                  eV
# 定义赫兹-电子伏特关系，第一个数字是值，括号内是确切值，单位为电子伏特
hertz_electron_volt_relationship = 4.135667696e-15

# hertz-hartree relationship                                  1.519 829 846 0570 e-16  0.000 000 000 0029 e-16  E_h
# 定义赫兹-哈特里关系，第一个数字是值，第二个数字是不确定度，单位为哈特里
hertz_hartree_relationship = 1.5198298460570e-16

# hertz-inverse meter relationship                            3.335 640 951... e-9     (exact)                  m^-1
# 定义赫兹-米^-1关系，第一个数字是值，括号内是确切值，单位为米的倒数
hertz_inverse_meter_relationship = 3.335640951e-9

# hertz-joule relationship                                    6.626 070 15 e-34        (exact)                  J
# 定义赫兹-焦耳关系，第一个数字是值，括号内是确切值，单位为焦耳
hertz_joule_relationship = 6.62607015e-34

# hertz-kelvin relationship                                   4.799 243 073... e-11    (exact)                  K
# 定义赫兹-开尔文关系，第一个数字是值，括号内是确切值，单位为开尔文
hertz_kelvin_relationship = 4.799243073e-11

# hertz-kilogram relationship                                 7.372 497 323... e-51    (exact)                  kg
# 定义赫兹-千克关系，第一个数字是值，括号内是确切值，单位为千克
hertz_kilogram_relationship = 7.372497323e-51

# hyperfine transition frequency of Cs-133                    9 192 631 770            (exact)                  Hz
# 定义铯-133原子的超精细跃迁频率，括号内是确切值，单位为赫兹
hyperfine_transition_frequency_of_cs133 = 9192631770

# inverse fine-structure constant                             137.035 999 084          0.000 000 021
# 定义细结构常数的倒数，第一个数字是值，第二个数字是不确定度
inverse_fine_structure_constant = 137.035999084

# inverse meter
# 定义逆米-赫兹关系常量，表示光速（精确值）
inverse meter-hertz relationship                            299 792 458              (exact)                  Hz
# 定义逆米-焦耳关系常量
inverse meter-joule relationship                            1.986 445 857... e-25    (exact)                  J
# 定义逆米-开尔文关系常量
inverse meter-kelvin relationship                           1.438 776 877... e-2     (exact)                  K
# 定义逆米-千克关系常量
inverse meter-kilogram relationship                         2.210 219 094... e-42    (exact)                  kg
# 定义电导量子的倒数
inverse of conductance quantum                              12 906.403 72...         (exact)                  ohm
# 定义约瑟夫逊常数
Josephson constant                                          483 597.848 4... e9      (exact)                  Hz V^-1
# 定义焦耳-原子质量单位关系
joule-atomic mass unit relationship                         6.700 535 2565 e9        0.000 000 0020 e9        u
# 定义焦耳-电子伏特关系
joule-electron volt relationship                            6.241 509 074... e18     (exact)                  eV
# 定义焦耳-哈特里关系
joule-hartree relationship                                  2.293 712 278 3963 e17   0.000 000 000 0045 e17   E_h
# 定义焦耳-赫兹关系
joule-hertz relationship                                    1.509 190 179... e33     (exact)                  Hz
# 定义焦耳-逆米关系
joule-inverse meter relationship                            5.034 116 567... e24     (exact)                  m^-1
# 定义焦耳-开尔文关系
joule-kelvin relationship                                   7.242 970 516... e22     (exact)                  K
# 定义焦耳-千克关系
joule-kilogram relationship                                 1.112 650 056... e-17    (exact)                  kg
# 定义开尔文-原子质量单位关系
kelvin-atomic mass unit relationship                        9.251 087 3014 e-14      0.000 000 0028 e-14      u
# 定义开尔文-电子伏特关系
kelvin-electron volt relationship                           8.617 333 262... e-5     (exact)                  eV
# 定义开尔文-哈特里关系
kelvin-hartree relationship                                 3.166 811 563 4556 e-6   0.000 000 000 0061 e-6   E_h
# 定义开尔文-赫兹关系
kelvin-hertz relationship                                   2.083 661 912... e10     (exact)                  Hz
# 定义开尔文-逆米关系
kelvin-inverse meter relationship                           69.503 480 04...         (exact)                  m^-1
# 定义开尔文-焦耳关系
kelvin-joule relationship                                   1.380 649 e-23           (exact)                  J
# 定义开尔文-千克关系
kelvin-kilogram relationship                                1.536 179 187... e-40    (exact)                  kg
# 定义千克-原子质量单位关系
kilogram-atomic mass unit relationship                      6.022 140 7621 e26       0.000 000 0018 e26       u
# 定义千克-电子伏特关系
kilogram-electron volt relationship                         5.609 588 603... e35     (exact)                  eV
# 定义千克-哈特里关系
kilogram-hartree relationship                               2.061 485 788 7409 e34   0.000 000 000 0040 e34   E_h
# 定义千克-赫兹关系
kilogram-hertz relationship                                 1.356 392 489... e50     (exact)                  Hz
# 定义千克-逆米关系
kilogram-inverse meter relationship                         4.524 438 335... e41     (exact)                  m^-1
# 定义千克-焦耳关系
kilogram-joule relationship                                 8.987 551 787... e16     (exact)                  J
# kilogram-kelvin relationship，千克和开尔文之间的关系，用科学记数法表示的确切值
kilogram-kelvin relationship                                6.509 657 260... e39     (exact)                  K

# lattice parameter of silicon，硅的晶格常数，包括主值和标准偏差
lattice parameter of silicon                                5.431 020 511 e-10       0.000 000 089 e-10       m

# lattice spacing of ideal Si (220)，理想情况下硅的(220)晶面间距，包括主值和标准偏差
lattice spacing of ideal Si (220)                           1.920 155 716 e-10       0.000 000 032 e-10       m

# Loschmidt constant (273.15 K, 100 kPa)，洛赫米特常数在273.15K和100kPa下的确切值，单位为每立方米
Loschmidt constant (273.15 K, 100 kPa)                      2.651 645 804... e25     (exact)                  m^-3

# Loschmidt constant (273.15 K, 101.325 kPa)，洛赫米特常数在273.15K和101.325kPa下的确切值，单位为每立方米
Loschmidt constant (273.15 K, 101.325 kPa)                  2.686 780 111... e25     (exact)                  m^-3

# luminous efficacy，发光效率的确切值，单位为流明每瓦特
luminous efficacy                                           683                      (exact)                  lm W^-1

# mag. flux quantum，磁通量子的确切值，单位为韦伯
mag. flux quantum                                           2.067 833 848... e-15    (exact)                  Wb

# molar gas constant，摩尔气体常数的确切值，单位为焦耳每摩尔每开尔文
molar gas constant                                          8.314 462 618...         (exact)                  J mol^-1 K^-1

# molar mass constant，摩尔质量常数的主值和标准偏差，单位为千克每摩尔
molar mass constant                                         0.999 999 999 65 e-3     0.000 000 000 30 e-3     kg mol^-1

# molar mass of carbon-12，碳-12的摩尔质量的主值和标准偏差，单位为千克每摩尔
molar mass of carbon-12                                     11.999 999 9958 e-3      0.000 000 0036 e-3       kg mol^-1

# molar Planck constant，摩尔普朗克常数的确切值，单位为焦耳每赫兹每摩尔
molar Planck constant                                       3.990 312 712... e-10    (exact)                  J Hz^-1 mol^-1

# molar volume of ideal gas (273.15 K, 100 kPa)，理想气体在273.15K和100kPa下的摩尔体积的确切值，单位为立方米每摩尔
molar volume of ideal gas (273.15 K, 100 kPa)               22.710 954 64... e-3     (exact)                  m^3 mol^-1

# molar volume of ideal gas (273.15 K, 101.325 kPa)，理想气体在273.15K和101.325kPa下的摩尔体积的确切值，单位为立方米每摩尔
molar volume of ideal gas (273.15 K, 101.325 kPa)           22.413 969 54... e-3     (exact)                  m^3 mol^-1

# molar volume of silicon，硅的摩尔体积的主值和标准偏差，单位为立方米每摩尔
molar volume of silicon                                     1.205 883 199 e-5        0.000 000 060 e-5        m^3 mol^-1

# Mo x unit，钼的X射线晶格常数的主值和标准偏差，单位为米
Mo x unit                                                   1.002 099 52 e-13        0.000 000 53 e-13        m

# muon Compton wavelength，μ子的康普顿波长的主值和标准偏差，单位为米
muon Compton wavelength                                     1.173 444 110 e-14       0.000 000 026 e-14       m

# muon-electron mass ratio，μ子电子质量比的主值和标准偏差
muon-electron mass ratio                                    206.768 2830             0.000 0046

# muon g factor，μ子的g因子的主值和标准偏差
muon g factor                                               -2.002 331 8418          0.000 000 0013

# muon mag. mom.，μ子的磁矩的主值和标准偏差，单位为焦耳每特斯拉
muon mag. mom.                                              -4.490 448 30 e-26       0.000 000 10 e-26        J T^-1

# muon mag. mom. anomaly，μ子的磁矩异常的主值和标准偏差
muon mag. mom. anomaly                                      1.165 920 89 e-3         0.000 000 63 e-3

# muon mag. mom. to Bohr magneton ratio，μ子的磁矩与玻尔磁子比值的主值和标准偏差
muon mag. mom. to Bohr magneton ratio                       -4.841 970 47 e-3        0.000 000 11 e-3

# muon mag. mom. to nuclear magneton ratio，μ子的磁矩与核磁子比值的主值和标准偏差
muon mag. mom. to nuclear magneton ratio                    -8.890 597 03            0.000 000 20

# muon mass，μ子的质量的主值和标准偏差，单位为千克
muon mass                                                   1.883 531 627 e-28       0.000 000 042 e-28       kg

# muon mass energy equivalent，μ子的质能等效值的主值和标准偏差，单位为焦耳
muon mass energy equivalent                                 1.692 833 804 e-11       0.000 000 038 e-11       J

# muon mass energy equivalent in MeV，μ子的质能等效值的主值和标准偏差，单位为兆电子伏特
muon mass energy equivalent in MeV                          105.658 3755             0.000 0023               MeV

# muon mass in u，μ子的质量的主值和标准偏差，单位为原子单位
muon mass in u                                              0.113 428 9259           0.000 000 0025           u
# 等于mu子的摩尔质量
muon molar mass                                             1.134 289 259 e-4        0.000 000 025 e-4        kg mol^-1
# mu子与中子质量的比率
muon-neutron mass ratio                                     0.112 454 5170           0.000 000 0025
# mu子与质子磁矩比
muon-proton mag. mom. ratio                                 -3.183 345 142           0.000 000 071
# mu子与质子质量比
muon-proton mass ratio                                      0.112 609 5264           0.000 000 0025
# mu子与tau子质量比
muon-tau mass ratio                                         5.946 35 e-2             0.000 40 e-2
# 自然单位制下的作用量
natural unit of action                                      1.054 571 817... e-34    (exact)                  J s
# 自然单位制下的作用量（以电子伏特秒为单位）
natural unit of action in eV s                              6.582 119 569... e-16    (exact)                  eV s
# 自然单位制下的能量
natural unit of energy                                      8.187 105 7769 e-14      0.000 000 0025 e-14      J
# 自然单位制下的能量（以兆电子伏特为单位）
natural unit of energy in MeV                               0.510 998 950 00         0.000 000 000 15         MeV
# 自然单位制下的长度
natural unit of length                                      3.861 592 6796 e-13      0.000 000 0012 e-13      m
# 自然单位制下的质量
natural unit of mass                                        9.109 383 7015 e-31      0.000 000 0028 e-31      kg
# 自然单位制下的动量
natural unit of momentum                                    2.730 924 530 75 e-22    0.000 000 000 82 e-22    kg m s^-1
# 自然单位制下的动量（以兆电子伏特每光速单位）
natural unit of momentum in MeV/c                           0.510 998 950 00         0.000 000 000 15         MeV/c
# 自然单位制下的时间
natural unit of time                                        1.288 088 668 19 e-21    0.000 000 000 39 e-21    s
# 自然单位制下的速度
natural unit of velocity                                    299 792 458              (exact)                  m s^-1
# 中子康普顿波长
neutron Compton wavelength                                  1.319 590 905 81 e-15    0.000 000 000 75 e-15    m
# 中子-电子磁矩比
neutron-electron mag. mom. ratio                            1.040 668 82 e-3         0.000 000 25 e-3
# 中子-电子质量比
neutron-electron mass ratio                                 1838.683 661 73          0.000 000 89
# 中子g因子
neutron g factor                                            -3.826 085 45            0.000 000 90
# 中子旋磁比
neutron gyromag. ratio                                      1.832 471 71 e8          0.000 000 43 e8          s^-1 T^-1
# 中子旋磁比（以兆赫每特斯拉为单位）
neutron gyromag. ratio in MHz/T                             29.164 6931              0.000 0069               MHz T^-1
# 中子磁矩
neutron mag. mom.                                           -9.662 3651 e-27         0.000 0023 e-27          J T^-1
# 中子磁矩与玻尔磁子比
neutron mag. mom. to Bohr magneton ratio                    -1.041 875 63 e-3        0.000 000 25 e-3
# 中子磁矩与核磁子比
neutron mag. mom. to nuclear magneton ratio                 -1.913 042 73            0.000 000 45
# 中子质量
neutron mass                                                1.674 927 498 04 e-27    0.000 000 000 95 e-27    kg
# 中子质量对应的能量
neutron mass energy equivalent                              1.505 349 762 87 e-10    0.000 000 000 86 e-10    J
# 中子质量对应的能量（以兆电子伏特为单位）
neutron mass energy equivalent in MeV                       939.565 420 52           0.000 000 54             MeV
# neutron mass in u                                           1.008 664 915 95         0.000 000 000 49         u
neutron_mass_u = 1.00866491595  # Neutron mass in atomic mass units (u)
neutron_mass_u_error = 0.00000000049  # Uncertainty in neutron mass in u

# neutron molar mass                                          1.008 664 915 60 e-3     0.000 000 000 57 e-3     kg mol^-1
neutron_molar_mass_kg_per_mol = 1.00866491560e-3  # Neutron molar mass in kilograms per mole (kg mol^-1)
neutron_molar_mass_error_kg_per_mol = 0.00000000057e-3  # Uncertainty in neutron molar mass in kg mol^-1

# neutron-muon mass ratio                                     8.892 484 06             0.000 000 20
neutron_muon_mass_ratio = 8.89248406  # Ratio of neutron mass to muon mass
neutron_muon_mass_ratio_error = 0.00000020  # Uncertainty in neutron-muon mass ratio

# neutron-proton mag. mom. ratio                              -0.684 979 34            0.000 000 16
neutron_proton_mag_mom_ratio = -0.68497934  # Neutron-proton magnetic moment ratio
neutron_proton_mag_mom_ratio_error = 0.00000016  # Uncertainty in neutron-proton mag. mom. ratio

# neutron-proton mass difference                              2.305 574 35 e-30        0.000 000 82 e-30        kg
neutron_proton_mass_diff_kg = 2.30557435e-30  # Neutron-proton mass difference in kilograms (kg)
neutron_proton_mass_diff_error_kg = 0.00000082e-30  # Uncertainty in neutron-proton mass difference in kg

# neutron-proton mass difference energy equivalent            2.072 146 89 e-13        0.000 000 74 e-13        J
neutron_proton_mass_diff_energy_J = 2.07214689e-13  # Neutron-proton mass difference energy equivalent in joules (J)
neutron_proton_mass_diff_energy_error_J = 0.00000074e-13  # Uncertainty in neutron-proton mass difference energy equivalent in J

# neutron-proton mass difference energy equivalent in MeV     1.293 332 36             0.000 000 46             MeV
neutron_proton_mass_diff_energy_MeV = 1.29333236  # Neutron-proton mass difference energy equivalent in mega-electron volts (MeV)
neutron_proton_mass_diff_energy_error_MeV = 0.00000046  # Uncertainty in neutron-proton mass difference energy equivalent in MeV

# neutron-proton mass difference in u                         1.388 449 33 e-3         0.000 000 49 e-3         u
neutron_proton_mass_diff_u = 1.38844933e-3  # Neutron-proton mass difference in atomic mass units (u)
neutron_proton_mass_diff_error_u = 0.00000049e-3  # Uncertainty in neutron-proton mass difference in u

# neutron-proton mass ratio                                   1.001 378 419 31         0.000 000 000 49
neutron_proton_mass_ratio = 1.00137841931  # Neutron-proton mass ratio
neutron_proton_mass_ratio_error = 0.00000000049  # Uncertainty in neutron-proton mass ratio

# neutron relative atomic mass                                1.008 664 915 95         0.000 000 000 49
neutron_relative_atomic_mass = 1.00866491595  # Neutron relative atomic mass
neutron_relative_atomic_mass_error = 0.00000000049  # Uncertainty in neutron relative atomic mass

# neutron-tau mass ratio                                      0.528 779                0.000 036
neutron_tau_mass_ratio = 0.528779  # Ratio of neutron mass to tau mass
neutron_tau_mass_ratio_error = 0.000036  # Uncertainty in neutron-tau mass ratio

# neutron to shielded proton mag. mom. ratio                  -0.684 996 94            0.000 000 16
neutron_shielded_proton_mag_mom_ratio = -0.68499694  # Neutron to shielded proton magnetic moment ratio
neutron_shielded_proton_mag_mom_ratio_error = 0.00000016  # Uncertainty in neutron to shielded proton mag. mom. ratio

# Newtonian constant of gravitation                           6.674 30 e-11            0.000 15 e-11            m^3 kg^-1 s^-2
G = 6.67430e-11  # Newtonian constant of gravitation in m^3 kg^-1 s^-2
G_error = 0.00015e-11  # Uncertainty in Newtonian constant of gravitation in m^3 kg^-1 s^-2

# Newtonian constant of gravitation over h-bar c              6.708 83 e-39            0.000 15 e-39            (GeV/c^2)^-2
G_over_hbarc = 6.70883e-39  # Newtonian constant of gravitation over h-bar c in (GeV/c^2)^-2
G_over_hbarc_error = 0.00015e-39  # Uncertainty in Newtonian constant of gravitation over h-bar c in (GeV/c^2)^-2

# nuclear magneton                                            5.050 783 7461 e-27      0.000 000 0015 e-27      J T^-1
nuclear_magneton_J_T = 5.0507837461e-27  # Nuclear magneton in joules per tesla (J T^-1)
nuclear_magneton_error_J_T = 0.0000000015e-27  # Uncertainty in nuclear magneton in J T^-1

# nuclear magneton in eV/T                                    3.152 451 258 44 e-8     0.000 000 000 96 e-8     eV T^-1
nuclear_magneton_eV_T = 3.15245125844e-8  # Nuclear magneton in electron volts per tesla (eV T^-1)
nuclear_magneton_error_eV_T = 0.00000000096e-8  # Uncertainty in nuclear magneton in eV T^-1

# nuclear magneton in inverse meter per tesla                 2.542 623 413 53 e-2     0.000 000 000 78 e-2     m^-1 T^-1
nuclear_magneton_m_per_T = 2.54262341353e-2  # Nuclear magneton in meters per tesla (m^-1 T^-1)
nuclear_magneton_error_m_per_T = 0.00000000078e-2  # Uncertainty in nuclear magneton in m^-1 T^-1

# nuclear magneton in K/T                                     3.658 267 7756 e-4       0.000 000 0011 e-4       K T^-1
nuclear_magneton_K_T = 3.6582677756e-4  # Nuclear magneton in kelvins per tesla (K T^-1)
nuclear_magneton_error_K_T = 0.0000000011e-4  # Uncertainty in nuclear magneton in K T^-1

# nuclear magneton in MHz/T                                   7.622 593 2291           0.000 000 0023           MHz T^-1
nuclear_magneton_MHz_T = 7.6225932291  # Nuclear magneton in megahertz per tesla (MHz T^-1)
nuclear_magneton_error_MHz_T = 0.0000000023  # Uncertainty in nuclear magneton in MHz T^-1

# Planck constant                                             6.626 070 15 e-34        (exact)                  J Hz^-1
planck_constant_J_Hz = 6.62607015e-34  # Planck constant in joules per hertz (J Hz^-1)

# Planck constant in eV/Hz                                    4.135 667 696... e-15    (exact)                  eV Hz^-1
planck_constant_eV_Hz = 4.135667696e-15  # Planck constant in electron volts per hertz (eV Hz^-1)

# Planck length                                               1.616 255 e-35           0.000 018 e-35           m
planck_length_m = 1.616255e-35  # Planck length in meters (m)
planck_length_error_m = 0.000018e-35  # Uncertainty in Planck length in meters

# Planck mass                                                 2.176 434 e-8            0.000
# 质子电荷与质量比
proton_charge_to_mass_quotient = 9.5788331560e7        # 单位为 C kg^-1，质子电荷与质量的比值
# 质子康普顿波长
proton_Compton_wavelength = 1.32140985539e-15          # 单位为 m，质子的康普顿波长
# 质子-电子质量比
proton_electron_mass_ratio = 1836.15267343             # 质子质量与电子质量的比值
# 质子 g 因子
proton_g_factor = 5.5856946893                         # 质子的 g 因子
# 质子旋磁比
proton_gyromag_ratio = 2.6752218744e8                 # 单位为 s^-1 T^-1，质子的旋磁比
# 质子旋磁比（MHz/T）
proton_gyromag_ratio_in_MHz_T = 42.577478518          # 单位为 MHz T^-1，质子的旋磁比
# 质子磁矩
proton_mag_mom = 1.41060679736e-26                     # 单位为 J T^-1，质子的磁矩
# 质子磁矩与玻尔磁子比值
proton_mag_mom_to_Bohr_magneton_ratio = 1.52103220230e-3  # 质子磁矩与玻尔磁子比值
# 质子磁矩与核磁子比值
proton_mag_mom_to_nuclear_magneton_ratio = 2.79284734463  # 质子磁矩与核磁子比值
# 质子磁屏蔽校正
proton_mag_shielding_correction = 2.5689e-5            # 质子磁屏蔽校正值
# 质子质量
proton_mass = 1.67262192369e-27                        # 单位为 kg，质子的质量
# 质子质能等效
proton_mass_energy_equivalent = 1.50327761598e-10      # 单位为 J，质子质能等效
# 质子质能等效（单位 MeV）
proton_mass_energy_equivalent_in_MeV = 938.27208816    # 单位为 MeV，质子质能等效
# 质子质量单位原子质量
proton_mass_in_u = 1.007276466621                      # 单位为 u，质子的质量单位原子质量
# 质子摩尔质量
proton_molar_mass = 1.00727646627e-3                   # 单位为 kg mol^-1，质子的摩尔质量
# 质子-μ子质量比
proton_muon_mass_ratio = 8.88024337                    # 质子与μ子质量的比值
# 质子-中子磁矩比值
proton_neutron_mag_mom_ratio = -1.45989805             # 质子与中子磁矩的比值
# 质子-中子质量比值
proton_neutron_mass_ratio = 0.99862347812              # 质子与中子质量的比值
# 质子相对原子质量
proton_relative_atomic_mass = 1.007276466621           # 质子的相对原子质量
# 质子有效电荷半径
proton_rms_charge_radius = 8.414e-16                   # 单位为 m，质子的有效电荷半径
# 质子-τ子质量比值
proton_tau_mass_ratio = 0.528051                       # 质子与τ子质量的比值
# 循环量子
quantum_of_circulation = 3.6369475516e-4               # 单位为 m^2 s^-1，循环量子
# 循环量子乘以2
quantum_of_circulation_times_2 = 7.2738951032e-4       # 单位为 m^2 s^-1，循环量子乘以2
# 减少的康普顿波长
reduced_Compton_wavelength = 3.8615926796e-13          # 单位为 m，减少的康普顿波长
# 减少的μ子康普顿波长
reduced_muon_Compton_wavelength = 1.867594306e-15     # 单位为 m，减少的μ子康普顿波长
# 减少的中子康普顿波长
reduced_neutron_Compton_wavelength = 2.1001941552e-16 # 单位为 m，减少的中子康普顿波长
# 减少的普朗克常数
reduced_Planck_constant = 1.054571817e-34              # 单位为 J s，减少的普朗克常数（精确值）
# reduced Planck constant in eV s
hbar_ev = 6.582119569e-16  # 约化 Planck 常数（单位：电子伏特秒），精确值

# reduced Planck constant times c in MeV fm
hbar_c_mev_fm = 197.3269804  # 约化 Planck 常数乘以光速（单位：兆电子伏特费米），精确值

# reduced proton Compton wavelength
hbar_over_m_p = 2.10308910336e-16  # 约化质子康普顿波长（单位：米），误差为 0.00000000064e-16 米

# reduced tau Compton wavelength
hbar_over_m_tau = 1.110538e-16  # 约化 τ 子康普顿波长（单位：米），误差为 0.000075e-16 米

# Rydberg constant
R_inf = 10973731.568160  # 雷德伯常数（单位：米^-1），误差为 0.000021 米^-1

# Rydberg constant times c in Hz
R_inf_c_hz = 3.2898419602508e15  # 雷德伯常数乘以光速（单位：赫兹），误差为 0.0000000000064e15 赫兹

# Rydberg constant times hc in eV
R_inf_hc_ev = 13.605693122994  # 雷德伯常数乘以 Planck 常数乘以光速（单位：电子伏特），误差为 0.000000000026 电子伏特

# Rydberg constant times hc in J
R_inf_hc_j = 2.1798723611035e-18  # 雷德伯常数乘以 Planck 常数乘以光速（单位：焦耳），误差为 0.0000000000042e-18 焦耳

# Sackur-Tetrode constant (1 K, 100 kPa)
S0 = -1.15170753706  # 萨克尔-特特罗德常数（1 K, 100 kPa），误差为 0.00000000045

# Sackur-Tetrode constant (1 K, 101.325 kPa)
S0_prime = -1.16487052358  # 萨克尔-特特罗德常数（1 K, 101.325 kPa），误差为 0.00000000045

# second radiation constant
sigma = 1.438776877e-2  # 第二辐射常数（单位：米开尔文），精确值

# shielded helion gyromagnetic ratio
gamma_prime_n = 2.037894569e8  # 屏蔽质子的旋磁比（单位：秒^-1特^-1），误差为 0.000000024e8 秒^-1特^-1

# shielded helion gyromagnetic ratio in MHz/T
gamma_prime_n_T = 32.43409942  # 屏蔽质子的旋磁比乘以兆赫兹/特（单位：兆赫兹/特），误差为 0.00000038 兆赫兹/特

# shielded helion magnetic moment
mu_prime_n = -1.074553090e-26  # 屏蔽质子的磁矩（单位：焦耳特^-1），误差为 0.000000013e-26 焦耳特^-1

# shielded helion magnetic moment to Bohr magneton ratio
mu_prime_n_over_mu_B = -1.158671471e-3  # 屏蔽质子的磁矩与玻尔磁子比值，误差为 0.000000014e-3

# shielded helion magnetic moment to nuclear magneton ratio
mu_prime_n_over_mu_N = -2.127497719  # 屏蔽质子的磁矩与核磁子比值，误差为 0.000000025

# shielded helion to proton magnetic moment ratio
mu_prime_n_over_mu_p = -0.7617665618  # 屏蔽质子对质子磁矩比值，误差为 0.0000000089

# shielded helion to shielded proton magnetic moment ratio
mu_prime_n_over_mu_prime_p = -0.7617861313  # 屏蔽质子对屏蔽质子磁矩比值，误差为 0.0000000033

# shielded proton gyromagnetic ratio
gamma_prime_p = 2.675153151e8  # 屏蔽质子的旋磁比（单位：秒^-1特^-1），误差为 0.000000029e8 秒^-1特^-1

# shielded proton gyromagnetic ratio in MHz/T
gamma_prime_p_T = 42.57638474  # 屏蔽质子的旋磁比乘以兆赫兹/特（单位：兆赫兹/特），误差为 0.00000046 兆赫兹/特

# shielded proton magnetic moment
mu_prime_p = 1.410570560e-26  # 屏蔽质子的磁矩（单位：焦耳特^-1），误差为 0.000000015e-26 焦耳特^-1

# shielded proton magnetic moment to Bohr magneton ratio
mu_prime_p_over_mu_B = 1.520993128e-3  # 屏蔽质子的磁矩与玻尔磁子比值，误差为 0.000000017e-3

# shielded proton magnetic moment to nuclear magneton ratio
mu_prime_p_over_mu_N = 2.792775599  # 屏蔽质子的磁矩与核磁子比值，误差为 0.000000030

# shielding difference of d and p in HD
dh = 2.0200e-8  # HD 分子中 d 和 p 的屏蔽差异，误差为 0.0020e-8

# shielding difference of t and p in HT
th = 2.4140e-8  # HT 分子中 t 和 p 的屏蔽差异，误差为 0.0020e-8

# speed of light in vacuum
c_0 = 299792458  # 真空中的光速（单位：米/秒），精确值

# standard acceleration of gravity
g_n = 9.80665  # 标准重力加速度（单位：米/秒^2），精确值
# 标准大气压力，精确值为 101325 帕斯卡
standard atmosphere                                         101 325                  (exact)                  Pa
# 标准状态压力，精确值为 100000 帕斯卡
standard-state pressure                                     100 000                  (exact)                  Pa
# Stefan-Boltzmann常数，精确值为 5.670 374 419... × 10^-8 瓦特每平方米每开尔文的第四次方
Stefan-Boltzmann constant                                   5.670 374 419... e-8     (exact)                  W m^-2 K^-4
# Tau康普顿波长，值为 6.977 71 × 10^-16 米，误差为 0.000 47 × 10^-16 米
tau Compton wavelength                                      6.977 71 e-16            0.000 47 e-16            m
# Tau电子质量比，值为 3477.23，误差为 0.23
tau-electron mass ratio                                     3477.23                  0.23
# Tau能量等效值，值为 1776.86 梅兆电子伏特，误差为 0.12 梅兆电子伏特
tau energy equivalent                                       1776.86                  0.12                     MeV
# Tau质量，值为 3.167 54 × 10^-27 千克，误差为 0.000 21 × 10^-27 千克
tau mass                                                    3.167 54 e-27            0.000 21 e-27            kg
# Tau质能等效值，值为 2.846 84 × 10^-10 焦耳，误差为 0.000 19 × 10^-10 焦耳
tau mass energy equivalent                                  2.846 84 e-10            0.000 19 e-10            J
# Tau质量单位，值为 1.907 54，误差为 0.000 13
tau mass in u                                               1.907 54                 0.000 13                 u
# Tau摩尔质量，值为 1.907 54 × 10^-3 千克每摩尔，误差为 0.000 13 × 10^-3 千克每摩尔
tau molar mass                                              1.907 54 e-3             0.000 13 e-3             kg mol^-1
# Tau-μ子质量比，值为 16.8170，误差为 0.0011
tau-muon mass ratio                                         16.8170                  0.0011
# Tau中子质量比，值为 1.891 15，误差为 0.000 13
tau-neutron mass ratio                                      1.891 15                 0.000 13
# Tau质子质量比，值为 1.893 76，误差为 0.000 13
tau-proton mass ratio                                       1.893 76                 0.000 13
# 汤姆逊散射截面，值为 6.652 458 7321 × 10^-29 平方米，误差为 0.000 000 0060 × 10^-29 平方米
Thomson cross section                                       6.652 458 7321 e-29      0.000 000 0060 e-29      m^2
# Triton电子质量比，值为 5496.921 535 73，误差为 0.000 000 27
triton-electron mass ratio                                  5496.921 535 73          0.000 000 27
# Triton g因子，值为 5.957 924 931，误差为 0.000 000 012
triton g factor                                             5.957 924 931            0.000 000 012
# Triton磁矩，值为 1.504 609 5202 × 10^-26 焦耳每特斯拉，误差为 0.000 000 0030 × 10^-26 焦耳每特斯拉
triton mag. mom.                                            1.504 609 5202 e-26      0.000 000 0030 e-26      J T^-1
# Triton磁矩对玻尔磁子比值，值为 1.622 393 6651 × 10^-3，误差为 0.000 000 0032 × 10^-3
triton mag. mom. to Bohr magneton ratio                     1.622 393 6651 e-3       0.000 000 0032 e-3
# Triton磁矩对核磁子比值，值为 2.978 962 4656，误差为 0.000 000 0059
triton mag. mom. to nuclear magneton ratio                  2.978 962 4656           0.000 000 0059
# Triton质量，值为 5.007 356 7446 × 10^-27 千克，误差为 0.000 000 0015 × 10^-27 千克
triton mass                                                 5.007 356 7446 e-27      0.000 000 0015 e-27      kg
# Triton质量能量等效值，值为 4.500 387 8060 × 10^-10 焦耳，误差为 0.000 000 0014 × 10^-10 焦耳
triton mass energy equivalent                               4.500 387 8060 e-10      0.000 000 0014 e-10      J
# Triton质量能量等效值，以兆电子伏特为单位，值为 2808.921 132 98，误差为 0.000 000 85 梅兆电子伏特
triton mass energy equivalent in MeV                        2808.921 132 98          0.000 000 85             MeV
# Triton质量单位，值为 3.015 500 716 21，误差为 0.000 000 000 12
triton mass in u                                            3.015 500 716 21         0.000 000 000 12         u
# Triton摩尔质量，值为 3.015 500 715 17 × 10^-3 千克每摩尔，误差为 0.000 000 000 92 × 10^-3 千克每摩尔
triton molar mass                                           3.015 500 715 17 e-3     0.000 000 000 92 e-3     kg mol^-1
# Triton-质子质量比，值为 2.993 717 034 14，误差为 0.000 000 000 15
triton-proton mass ratio                                    2.993 717 034 14         0.000 000 000 15
# Triton相对原子质量，值为 3.015 500 716 21，误差为 0.000 000 000 12
triton relative atomic mass                                 3.015 500 716 21         0.000 000 000 12
# Triton对质子磁矩比值，值为 1.066 639 9191，误差为 0.000 000 0021
triton to proton mag. mom. ratio                            1.066 639 9191           0.000 000 0021
# 定义一个空的字典，用于存储物理常数及其数据的元组
physical_constants: dict[str, tuple[float, str, float]] = {}

# 解析2002至2014年间的物理常数数据，返回一个字典，其中键为常数名称，值为包含常数值、单位和不确定性的元组
def parse_constants_2002to2014(d: str) -> dict[str, tuple[float, str, float]]:
    constants = {}
    for line in d.split('\n'):
        # 提取常数名称，去除尾部空格
        name = line[:55].rstrip()
        # 提取常数值，去除空格和省略号，转换为浮点数
        val = float(line[55:77].replace(' ', '').replace('...', ''))
        # 提取不确定性，去除空格和"(exact)"，若为"(exact)"则置为0，转换为浮点数
        uncert = float(line[77:99].replace(' ', '').replace('(exact)', '0'))
        # 提取单位，去除尾部空格
        units = line[99:].rstrip()
        # 将常数名称及其数据存入字典
        constants[name] = (val, units, uncert)
    return constants

# 解析2018年及之后的物理常数数据，与上述函数相似，不同之处在于提取常数名称和数据的位置略有不同
def parse_constants_2018toXXXX(d: str) -> dict[str, tuple[float, str, float]]:
    constants = {}
    for line in d.split('\n'):
        name = line[:60].rstrip()
        val = float(line[60:85].replace(' ', '').replace('...', ''))
        uncert = float(line[85:110].replace(' ', '').replace('(exact)', '0'))
        units = line[110:].rstrip()
        constants[name] = (val, units, uncert)
    return constants

# 分别解析不同年份的物理常数数据
_physical_constants_2002 = parse_constants_2002to2014(txt2002)
_physical_constants_2006 = parse_constants_2002to2014(txt2006)
_physical_constants_2010 = parse_constants_2002to2014(txt2010)
_physical_constants_2014 = parse_constants_2002to2014(txt2014)
_physical_constants_2018 = parse_constants_2018toXXXX(txt2018)

# 将解析得到的物理常数数据更新到总的物理常数字典中
physical_constants.update(_physical_constants_2002)
physical_constants.update(_physical_constants_2006)
physical_constants.update(_physical_constants_2010)
physical_constants.update(_physical_constants_2014)
physical_constants.update(_physical_constants_2018)

# 将2018年的物理常数数据标记为当前使用的数据
_current_constants = _physical_constants_2018
_current_codata = "CODATA 2018"

# 检查是否存在已经废弃的物理常数，并将其存入一个字典中
_obsolete_constants = {}
for k in physical_constants:
    if k not in _current_constants:
        _obsolete_constants[k] = True

# 生成一些额外的常数别名，根据特定条件，对于2002年的数据，将包含'magn.'的常数名称替换为'mag.'
_aliases = {}
for k in _physical_constants_2002:
    if 'magn.' in k:
        _aliases[k] = k.replace('magn.', 'mag.')
# 对于2006年的数据，将包含'momentum'的常数名称替换为'mom.um'
for k in _physical_constants_2006:
    if 'momentum' in k:
        _aliases[k] = k.replace('momentum', 'mom.um')
# 对于2018年的数据，可以继续添加别名的生成规则，但未在提供的代码段中给出
    # 检查字符串 'momentum' 是否在变量 k 中
    if 'momentum' in k:
        # 如果是，则将变量 k 中的 'momentum' 替换为 'mom.um'，并将替换后的结果存入 _aliases 字典中
        _aliases[k] = k.replace('momentum', 'mom.um')
# 将 'mag. constant' 映射为 'vacuum mag. permeability'，作为常量的别名
_aliases['mag. constant'] = 'vacuum mag. permeability'
# 将 'electric constant' 映射为 'vacuum electric permittivity'，作为常量的别名
_aliases['electric constant'] = 'vacuum electric permittivity'


class ConstantWarning(DeprecationWarning):
    """定义一个警告类，用于指示访问不再在当前CODATA数据集中的常量"""
    pass


def _check_obsolete(key: str) -> None:
    # 如果 key 在 _obsolete_constants 中但不在 _aliases 中，则发出警告
    if key in _obsolete_constants and key not in _aliases:
        warnings.warn(f"Constant '{key}' is not in current {_current_codata} data set",
                      ConstantWarning, stacklevel=3)


def value(key: str) -> float:
    """
    获取物理常数字典中键为 key 的数值

    Parameters
    ----------
    key : str
        字典 `physical_constants` 中的键

    Returns
    -------
    value : float
        对应于 `key` 的 `physical_constants` 中的数值

    Examples
    --------
    >>> from scipy import constants
    >>> constants.value('elementary charge')
    1.602176634e-19

    """
    # 检查是否为过时常量，若过时则发出警告
    _check_obsolete(key)
    # 返回键为 key 的物理常数的数值
    return physical_constants[key][0]


def unit(key: str) -> str:
    """
    获取物理常数字典中键为 key 的单位字符串

    Parameters
    ----------
    key : str
        字典 `physical_constants` 中的键

    Returns
    -------
    unit : str
        对应于 `key` 的 `physical_constants` 中的单位字符串

    Examples
    --------
    >>> from scipy import constants
    >>> constants.unit('proton mass')
    'kg'

    """
    # 检查是否为过时常量，若过时则发出警告
    _check_obsolete(key)
    # 返回键为 key 的物理常数的单位字符串
    return physical_constants[key][1]


def precision(key: str) -> float:
    """
    获取物理常数字典中键为 key 的相对精度

    Parameters
    ----------
    key : str
        字典 `physical_constants` 中的键

    Returns
    -------
    prec : float
        对应于 `key` 的 `physical_constants` 中的相对精度

    Examples
    --------
    >>> from scipy import constants
    >>> constants.precision('proton mass')
    5.1e-37

    """
    # 检查是否为过时常量，若过时则发出警告
    _check_obsolete(key)
    # 返回键为 key 的物理常数的相对精度
    return physical_constants[key][2] / physical_constants[key][0]


def find(sub: str | None = None, disp: bool = False) -> Any:
    """
    返回包含给定字符串的物理常数键列表。

    Parameters
    ----------
    sub : str, optional
        要搜索的子字符串。默认情况下，返回所有键。
    disp : bool, optional
        如果为 True，则打印找到的键并返回 None。
        否则，返回键列表而不打印任何内容。

    Returns
    -------
    keys : list or None
        如果 `disp` 为 False，则返回键列表。
        否则，返回 None。

    Examples
    --------
    >>> from scipy.constants import find, physical_constants

    查找 ``physical_constants`` 字典中包含 'boltzmann' 的键。

    >>> find('boltzmann')
    ['Boltzmann constant',
     'Boltzmann constant in Hz/K',
     'Boltzmann constant in eV/K',
     'Boltzmann constant in inverse meter per kelvin',
     'Stefan-Boltzmann constant']

    """
    # 如果指定了 sub，则打印包含该子字符串的键，并根据 disp 返回结果
    # 否则，返回所有键列表
    if sub is not None:
        keys = [k for k in physical_constants if sub in k]
        if disp:
            print(keys)
            return None
        else:
            return keys
    else:
        return list(physical_constants.keys())
    # 如果子字符串 sub 为 None，则返回当前常量字典 _current_constants 的所有键列表
    if sub is None:
        result = list(_current_constants.keys())
    # 否则，返回在 _current_constants 中键名包含子字符串 sub（不区分大小写）的键列表
    else:
        result = [key for key in _current_constants
                  if sub.lower() in key.lower()]
    
    # 对结果列表 result 进行排序
    result.sort()
    
    # 如果 disp 为真，则逐行打印结果列表中的键名，并返回 None
    if disp:
        for key in result:
            print(key)
        return
    # 如果 disp 不为真，则直接返回结果列表
    else:
        return result
# 初始化三个物理常数的变量，并分别赋予其初始值
c = value('speed of light in vacuum')               # 光速在真空中的数值
mu0 = value('vacuum mag. permeability')             # 真空的磁导率
epsilon0 = value('vacuum electric permittivity')    # 真空的电容率

# 由于表中的一些数值缺少精确度，因此根据定义计算确切值
exact_values = {
    'joule-kilogram relationship': (1 / (c * c), 'kg', 0.0),                    # 能量单位与千克的关系
    'kilogram-joule relationship': (c * c, 'J', 0.0),                           # 千克与焦耳的关系
    'hertz-inverse meter relationship': (1 / c, 'm^-1', 0.0),                   # 赫兹与米的倒数的关系
}

# 对确切值进行健全性检查
for key in exact_values:
    val = physical_constants[key][0]                                            # 获取已知物理常数中对应键的值
    if abs(exact_values[key][0] - val) / val > 1e-9:                            # 如果相对误差大于指定阈值
        raise ValueError("Constants.codata: exact values too far off.")         # 抛出数值偏差过大的异常
    if exact_values[key][2] == 0 and physical_constants[key][2] != 0:           # 如果确切值为零但已知常数不为零
        raise ValueError("Constants.codata: value not exact")                    # 抛出不确切值的异常

# 将计算得到的确切值更新到物理常数字典中
physical_constants.update(exact_values)

# 测试用的键列表，用于检查是否存在单位或常数的别名
_tested_keys = ['natural unit of velocity',
                'natural unit of action',
                'natural unit of action in eV s',
                'natural unit of mass',
                'natural unit of energy',
                'natural unit of energy in MeV',
                'natural unit of mom.um',
                'natural unit of mom.um in MeV/c',
                'natural unit of length',
                'natural unit of time']

# 最后，为值插入别名
for k, v in list(_aliases.items()):
    if v in _current_constants or v in _tested_keys:                            # 如果别名存在于当前常数或测试键中
        physical_constants[k] = physical_constants[v]                           # 使用相应常数的值更新别名的值
    else:
        del _aliases[k]                                                         # 否则删除无效的别名
```