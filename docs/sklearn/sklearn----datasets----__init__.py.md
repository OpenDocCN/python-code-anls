# `D:\src\scipysrc\scikit-learn\sklearn\datasets\__init__.py`

```
# 加载流行数据集和人工数据生成器的实用工具模块

import textwrap  # 导入文本包装模块

from ._base import (  # 导入基础模块中的以下函数和类
    clear_data_home,
    get_data_home,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_files,
    load_iris,
    load_linnerud,
    load_sample_image,
    load_sample_images,
    load_wine,
)
from ._california_housing import fetch_california_housing  # 导入加利福尼亚住房数据集获取函数
from ._covtype import fetch_covtype  # 导入覆盖类型数据集获取函数
from ._kddcup99 import fetch_kddcup99  # 导入KDD Cup 99数据集获取函数
from ._lfw import fetch_lfw_pairs, fetch_lfw_people  # 导入LFW数据集中的人物对和人物获取函数
from ._olivetti_faces import fetch_olivetti_faces  # 导入Olivetti人脸数据集获取函数
from ._openml import fetch_openml  # 导入OpenML数据集获取函数
from ._rcv1 import fetch_rcv1  # 导入RCV1数据集获取函数
from ._samples_generator import (  # 导入样本生成器模块中的以下函数
    make_biclusters,
    make_blobs,
    make_checkerboard,
    make_circles,
    make_classification,
    make_friedman1,
    make_friedman2,
    make_friedman3,
    make_gaussian_quantiles,
    make_hastie_10_2,
    make_low_rank_matrix,
    make_moons,
    make_multilabel_classification,
    make_regression,
    make_s_curve,
    make_sparse_coded_signal,
    make_sparse_spd_matrix,
    make_sparse_uncorrelated,
    make_spd_matrix,
    make_swiss_roll,
)
from ._species_distributions import fetch_species_distributions  # 导入物种分布数据集获取函数
from ._svmlight_format_io import (  # 导入SVMLight格式I/O模块中的以下函数
    dump_svmlight_file,
    load_svmlight_file,
    load_svmlight_files,
)
from ._twenty_newsgroups import (  # 导入Twenty Newsgroups数据集中的以下函数
    fetch_20newsgroups,
    fetch_20newsgroups_vectorized,
)

__all__ = [  # 可导出的模块和函数列表
    "clear_data_home",
    "dump_svmlight_file",
    "fetch_20newsgroups",
    "fetch_20newsgroups_vectorized",
    "fetch_lfw_pairs",
    "fetch_lfw_people",
    "fetch_olivetti_faces",
    "fetch_species_distributions",
    "fetch_california_housing",
    "fetch_covtype",
    "fetch_rcv1",
    "fetch_kddcup99",
    "fetch_openml",
    "get_data_home",
    "load_diabetes",
    "load_digits",
    "load_files",
    "load_iris",
    "load_breast_cancer",
    "load_linnerud",
    "load_sample_image",
    "load_sample_images",
    "load_svmlight_file",
    "load_svmlight_files",
    "load_wine",
    "make_biclusters",
    "make_blobs",
    "make_circles",
    "make_classification",
    "make_checkerboard",
    "make_friedman1",
    "make_friedman2",
    "make_friedman3",
    "make_gaussian_quantiles",
    "make_hastie_10_2",
    "make_low_rank_matrix",
    "make_moons",
    "make_multilabel_classification",
    "make_regression",
    "make_s_curve",
    "make_sparse_coded_signal",
    "make_sparse_spd_matrix",
    "make_sparse_uncorrelated",
    "make_spd_matrix",
    "make_swiss_roll",
]

def __getattr__(name):
    # 如果请求加载的模块名为 "load_boston"
    if name == "load_boston":
        # 组装多行消息，提醒用户自 scikit-learn 版本 1.2 起移除了 "load_boston"
        msg = textwrap.dedent(
            """
            `load_boston` has been removed from scikit-learn since version 1.2.

            The Boston housing prices dataset has an ethical problem: as
            investigated in [1], the authors of this dataset engineered a
            non-invertible variable "B" assuming that racial self-segregation had a
            positive impact on house prices [2]. Furthermore the goal of the
            research that led to the creation of this dataset was to study the
            impact of air quality but it did not give adequate demonstration of the
            validity of this assumption.

            The scikit-learn maintainers therefore strongly discourage the use of
            this dataset unless the purpose of the code is to study and educate
            about ethical issues in data science and machine learning.

            In this special case, you can fetch the dataset from the original
            source::

                import pandas as pd
                import numpy as np

                data_url = "http://lib.stat.cmu.edu/datasets/boston"
                raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
                data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
                target = raw_df.values[1::2, 2]

            Alternative datasets include the California housing dataset and the
            Ames housing dataset. You can load the datasets as follows::

                from sklearn.datasets import fetch_california_housing
                housing = fetch_california_housing()

            for the California housing dataset and::

                from sklearn.datasets import fetch_openml
                housing = fetch_openml(name="house_prices", as_frame=True)

            for the Ames housing dataset.

            [1] M Carlisle.
            "Racist data destruction?"
            <https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8>

            [2] Harrison Jr, David, and Daniel L. Rubinfeld.
            "Hedonic housing prices and the demand for clean air."
            Journal of environmental economics and management 5.1 (1978): 81-102.
            <https://www.researchgate.net/publication/4974606_Hedonic_housing_prices_and_the_demand_for_clean_air>
            """
        )
        # 抛出 ImportError 异常，包含详细消息
        raise ImportError(msg)
    
    try:
        # 尝试从全局变量中返回请求的模块名对应的对象
        return globals()[name]
    except KeyError:
        # 如果模块名不存在于全局变量中，抛出 AttributeError 异常
        # 这将被转换为适当的 ImportError
        raise AttributeError
```