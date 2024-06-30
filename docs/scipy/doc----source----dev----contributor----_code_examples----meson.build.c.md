# `D:\src\scipysrc\scipy\doc\source\dev\contributor\_code_examples\meson.build.c`

```
# 定义项目名称为 'repro_gh_11577'，使用语言为 'c'
project('repro_gh_11577', 'c')

# 获取 openblas 的依赖项
openblas_dep = dependency('openblas')

# 定义名为 'repro_c' 的可执行文件，编译源文件 'ggev_repro_gh_11577.c'
# 并指定其依赖项为 openblas_dep
executable('repro_c',
    'ggev_repro_gh_11577.c',
    dependencies: openblas_dep
)
```