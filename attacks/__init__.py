"""
自动驾驶攻击样本生成模块

支持的攻击类型：
1. adversarial_patch - 对抗补丁攻击
2. sensor_spoofing - 传感器欺骗攻击（LiDAR点云注入/删除）
3. physical_attack - 物理攻击（遮挡、天气干扰）
4. data_poisoning - 数据投毒攻击
"""

from .attack_generator import (
    AttackGenerator,
    AdversarialPatchAttack,
    LiDARSpoofingAttack,
    PhysicalAttack,
    DataPoisoningAttack,
    # 类别映射
    ATTACK_CLASSES,
    ATTACK_NAMES,
    NUM_CLASSES,
    attack_type_to_label,
    label_to_attack_type,
    is_attack_label
)

from .attack_dataset import (
    NuScenesAttackDataset,
    SyntheticAttackDataset,
    attack_collate_fn,
    create_attack_dataset
)

__all__ = [
    # 攻击生成器
    'AttackGenerator',
    'AdversarialPatchAttack', 
    'LiDARSpoofingAttack',
    'PhysicalAttack',
    'DataPoisoningAttack',
    # 数据集
    'NuScenesAttackDataset',
    'SyntheticAttackDataset',
    'attack_collate_fn',
    'create_attack_dataset',
    # 类别映射
    'ATTACK_CLASSES',
    'ATTACK_NAMES',
    'NUM_CLASSES',
    'attack_type_to_label',
    'label_to_attack_type',
    'is_attack_label'
]
