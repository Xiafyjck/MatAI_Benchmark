# config.py

import dataclasses
from typing import Optional

@dataclasses.dataclass
class BaseConfig:
    """所有模型的公共配置或默认值"""
    seed: int = 400
    # 可以在这里放一些大多数模型都会用到的参数
    # 比如通用的数据路径、日志开关等

    def __post_init__(self):
        """在这里可以做一些验证或动态处理.ChatGPT"""
        pass


@dataclasses.dataclass
class cgcnn_xie_config(BaseConfig):
    """模型 A 的专用配置"""
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 10
    hidden_dim: int = 128
    # 如果还有别的 A 独有参数就放这


@dataclasses.dataclass
class cgcnn_deepchem_config(BaseConfig):
    """模型 B 的专用配置"""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 15
    hidden_dim: int = 256
    # B 独有参数