# `D:\src\scipysrc\sympy\sympy\diffgeom\rn.py`

```
###############################################################################
# R3
###############################################################################
# 定义三维空间流形 R^3
R3: Any = Manifold('R^3', 3)

# 定义 R^3 流形的起始点 Patch
R3_origin: Any = Patch('origin', R3)

# 定义实数符号 x, y, z 作为三维坐标系的变量
x, y, z = symbols('x y z', real=True)
# 定义符号变量 rho, psi, r, theta, phi，限定它们为非负数
rho, psi, r, theta, phi = symbols('rho psi r theta phi', nonnegative=True)

# 定义 3D 坐标系间的转换关系字典
relations_3d = {
    ('rectangular', 'cylindrical'): [(x, y, z),  # 直角坐标系到柱面坐标系的转换关系
                                     (sqrt(x**2 + y**2), atan2(y, x), z)],
    ('cylindrical', 'rectangular'): [(rho, psi, z),  # 柱面坐标系到直角坐标系的转换关系
                                     (rho*cos(psi), rho*sin(psi), z)],
    ('rectangular', 'spherical'): [(x, y, z),  # 直角坐标系到球面坐标系的转换关系
                                   (sqrt(x**2 + y**2 + z**2),
                                    acos(z/sqrt(x**2 + y**2 + z**2)),
                                    atan2(y, x))],
    ('spherical', 'rectangular'): [(r, theta, phi),  # 球面坐标系到直角坐标系的转换关系
                                   (r*sin(theta)*cos(phi),
                                    r*sin(theta)*sin(phi),
                                    r*cos(theta))],
    ('cylindrical', 'spherical'): [(rho, psi, z),  # 柱面坐标系到球面坐标系的转换关系
                                   (sqrt(rho**2 + z**2),
                                    acos(z/sqrt(rho**2 + z**2)),
                                    psi)],
    ('spherical', 'cylindrical'): [(r, theta, phi),  # 球面坐标系到柱面坐标系的转换关系
                                   (r*sin(theta), phi, r*cos(theta))],
}

# 创建直角坐标系 R3_r，柱面坐标系 R3_c，球面坐标系 R3_s，并指定它们的基本关系
R3_r: Any = CoordSystem('rectangular', R3_origin, (x, y, z), relations_3d)
R3_c: Any = CoordSystem('cylindrical', R3_origin, (rho, psi, z), relations_3d)
R3_s: Any = CoordSystem('spherical', R3_origin, (r, theta, phi), relations_3d)

# 忽略警告，以支持不推荐使用的特性
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # 定义虚拟符号变量 x, y, z, rho, psi, r, theta, phi
    x, y, z, rho, psi, r, theta, phi = symbols('x y z rho psi r theta phi', cls=Dummy)
    
    # 设置直角坐标系到柱面坐标系的转换关系
    R3_r.connect_to(R3_c, [x, y, z],
                        [sqrt(x**2 + y**2), atan2(y, x), z],
                    inverse=False, fill_in_gaps=False)
    
    # 设置柱面坐标系到直角坐标系的转换关系
    R3_c.connect_to(R3_r, [rho, psi, z],
                        [rho*cos(psi), rho*sin(psi), z],
                    inverse=False, fill_in_gaps=False)
    
    # 设置直角坐标系到球面坐标系的转换关系
    R3_r.connect_to(R3_s, [x, y, z],
                        [sqrt(x**2 + y**2 + z**2), acos(z/
                                sqrt(x**2 + y**2 + z**2)), atan2(y, x)],
                    inverse=False, fill_in_gaps=False)
    
    # 设置球面坐标系到直角坐标系的转换关系
    R3_s.connect_to(R3_r, [r, theta, phi],
                        [r*sin(theta)*cos(phi), r*sin(
                            theta)*sin(phi), r*cos(theta)],
                    inverse=False, fill_in_gaps=False)
    
    # 设置柱面坐标系到球面坐标系的转换关系
    R3_c.connect_to(R3_s, [rho, psi, z],
                        [sqrt(rho**2 + z**2), acos(z/sqrt(rho**2 + z**2)), psi],
                    inverse=False, fill_in_gaps=False)
    
    # 设置球面坐标系到柱面坐标系的转换关系
    R3_s.connect_to(R3_c, [r, theta, phi],
                        [r*sin(theta), phi, r*cos(theta)],
                    inverse=False, fill_in_gaps=False)

# 定义直角坐标系 R3_r 的基本坐标函数
R3_r.x, R3_r.y, R3_r.z = R3_r.coord_functions()

# 定义柱面坐标系 R3_c 的基本坐标函数
R3_c.rho, R3_c.psi, R3_c.z = R3_c.coord_functions()

# 定义球面坐标系 R3_s 的基本坐标函数
R3_s.r, R3_s.theta, R3_s.phi = R3_s.coord_functions()

# 定义直角坐标系 R3_r 的基本向量场
R3_r.e_x, R3_r.e_y, R3_r.e_z = R3_r.base_vectors()
# 使用 R3_c 对象的 base_vectors 方法来获取三维曲面上的基向量 e_rho, e_psi, e_z，并将其分配给对应的属性。
R3_c.e_rho, R3_c.e_psi, R3_c.e_z = R3_c.base_vectors()

# 使用 R3_s 对象的 base_vectors 方法来获取球面上的基向量 e_r, e_theta, e_phi，并将其分配给对应的属性。
R3_s.e_r, R3_s.e_theta, R3_s.e_phi = R3_s.base_vectors()

# 定义三维空间 R3_r 对象的基一形式字段，将其结果分配给 dx, dy, dz 属性。
R3_r.dx, R3_r.dy, R3_r.dz = R3_r.base_oneforms()

# 定义三维曲面 R3_c 对象的基一形式字段，将其结果分配给 drho, dpsi, dz 属性。
R3_c.drho, R3_c.dpsi, R3_c.dz = R3_c.base_oneforms()

# 定义球面 R3_s 对象的基一形式字段，将其结果分配给 dr, dtheta, dphi 属性。
R3_s.dr, R3_s.dtheta, R3_s.dphi = R3_s.base_oneforms()
```