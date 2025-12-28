"""
模型模块

包含:
- Qwen3VLDefenseSystemEnhanced: 增强版模型，支持训练LLM生成任务
"""

from .enhanced_model import (
    Qwen3VLDefenseSystemEnhanced,
    EnhancedTrainer,
    DETECTION_TEMPLATES,
    DEFENSE_TEMPLATES
)

__all__ = [
    'Qwen3VLDefenseSystemEnhanced',
    'EnhancedTrainer',
    'DETECTION_TEMPLATES',
    'DEFENSE_TEMPLATES'
]
