"""
增强版NuScenes数据集

特点：
1. 集成真实攻击样本生成
2. 支持多种攻击类型
3. 保存攻击详细信息用于分析
"""

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
from typing import Dict, List, Optional, Tuple
import random

# 导入攻击生成器
from attacks.attack_generator import AttackGenerator


class NuScenesAttackDataset(Dataset):
    """
    增强版NuScenes数据集 - 支持真实攻击生成
    
    与原始数据集的区别：
    1. 不是简单标记攻击，而是真正生成攻击样本
    2. 支持多种攻击类型的真实效果
    3. 记录详细攻击信息用于分析
    """
    
    def __init__(
        self, 
        dataroot: str,
        version: str = 'v1.0-trainval',
        split: str = 'train',
        attack_ratio: float = 0.5,
        num_points: int = 2048,
        use_cache: bool = False,
        attack_config: Dict = None,
        image_size: Tuple[int, int] = (384, 640)
    ):
        """
        Args:
            dataroot: NuScenes数据集根目录
            version: 数据集版本
            split: 'train' 或 'val'
            attack_ratio: 生成攻击样本的比例
            num_points: 点云采样点数
            use_cache: 是否使用缓存
            attack_config: 攻击生成器配置
            image_size: 图像目标尺寸 (H, W)
        """
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.splits import create_splits_scenes
        
        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.attack_ratio = attack_ratio
        self.num_points = num_points
        self.use_cache = use_cache
        self.image_size = image_size
        
        # 保存导入的类供后续使用
        self.LidarPointCloud = LidarPointCloud
        
        print(f"初始化NuScenes攻击数据集: {split}")
        print(f"攻击比例: {attack_ratio:.1%}")
        
        # 加载NuScenes
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        
        # 获取场景分割
        splits = create_splits_scenes()
        if split == 'train':
            scene_names = splits['train']
        elif split == 'val':
            scene_names = splits['val']
        else:
            scene_names = splits['train'] + splits['val']
        
        # 过滤当前数据集中存在的场景
        available_scenes = [s['name'] for s in self.nusc.scene]
        scene_names = [s for s in scene_names if s in available_scenes]
        
        # 收集样本
        self.samples = []
        self.scene_tokens = []
        for scene in self.nusc.scene:
            if scene['name'] in scene_names:
                sample_token = scene['first_sample_token']
                while sample_token:
                    self.samples.append(sample_token)
                    self.scene_tokens.append(scene['token'])
                    sample = self.nusc.get('sample', sample_token)
                    sample_token = sample['next']
        
        print(f"找到 {len(self.samples)} 个样本")
        
        # 初始化攻击生成器
        default_attack_config = {
            'attack_ratio': attack_ratio,
            'attack_weights': {
                'adversarial_patch': 0.25,
                'sensor_spoofing': 0.25,
                'physical_attack': 0.25,
                'data_poisoning': 0.25
            }
        }
        if attack_config:
            default_attack_config.update(attack_config)
        
        self.attack_generator = AttackGenerator(**default_attack_config)
        
        # 预生成每个样本是否被攻击的标记（保证可复现性）
        self._precompute_attack_assignments()
        
        # 图像变换（用于模型输入）
        self.image_transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 攻击统计
        self.attack_stats = {
            'total_samples': len(self.samples),
            'attack_samples': 0,
            'attack_type_counts': {
                'adversarial_patch': 0,
                'sensor_spoofing': 0,
                'physical_attack': 0,
                'data_poisoning': 0,
                'normal': 0
            }
        }
    
    def _precompute_attack_assignments(self):
        """预计算每个样本的攻击分配"""
        np.random.seed(42)  # 保证可复现性
        
        self.attack_assignments = []
        for i in range(len(self.samples)):
            if random.random() < self.attack_ratio:
                attack_type = self.attack_generator.select_attack_type()
                self.attack_assignments.append({
                    'is_attack': True,
                    'attack_type': attack_type
                })
            else:
                self.attack_assignments.append({
                    'is_attack': False,
                    'attack_type': 'normal'
                })
        
        # 重置随机种子
        np.random.seed(None)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_token = self.samples[idx]
        sample = self.nusc.get('sample', sample_token)
        
        # 加载前视相机图像
        cam_front_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        img_path = os.path.join(self.dataroot, cam_front_data['filename'])
        image = Image.open(img_path).convert('RGB')
        
        # 加载点云
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path = os.path.join(self.dataroot, lidar_data['filename'])
        pc = self.LidarPointCloud.from_file(lidar_path)
        points = pc.points[:3, :].T  # [N, 3]
        
        # 采样固定数量的点
        if points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[indices]
        elif points.shape[0] < self.num_points:
            pad_size = self.num_points - points.shape[0]
            points = np.vstack([points, np.zeros((pad_size, 3))])
        
        # 获取预分配的攻击信息
        assignment = self.attack_assignments[idx]
        
        # 应用攻击（如果需要）
        if assignment['is_attack']:
            attacked_image, attacked_points, attack_info = self.attack_generator.apply_attack(
                image, points, assignment['attack_type']
            )
            attack_type = assignment['attack_type']
        else:
            attacked_image = image
            attacked_points = points
            attack_info = {'attack_type': 'normal', 'is_attack': False}
            attack_type = 'normal'
        
        # 5分类标签 (0=normal, 1-4=各种攻击)
        from attacks.attack_generator import attack_type_to_label
        label = attack_type_to_label(attack_type)
        
        # 更新统计
        self.attack_stats['attack_type_counts'][attack_type] += 1
        if label > 0:
            self.attack_stats['attack_samples'] += 1
        
        # 转换为tensor
        pointcloud = torch.from_numpy(attacked_points).float()
        label_tensor = torch.tensor(label, dtype=torch.long)  # 5分类标签 0-4
        
        return {
            'images': attacked_image,  # PIL Image
            'pointclouds': pointcloud,
            'labels': label_tensor,  # 5分类标签
            'attack_types': attack_type,
            'attack_info': attack_info,
            'sample_tokens': sample_token
        }
    
    def get_statistics(self):
        """获取数据集统计信息"""
        # 计算攻击分布
        attack_counts = {'normal': 0}
        for assignment in self.attack_assignments:
            attack_type = assignment['attack_type']
            if attack_type not in attack_counts:
                attack_counts[attack_type] = 0
            attack_counts[attack_type] += 1
        
        print(f"\n{'='*60}")
        print(f"数据集统计 ({self.split})")
        print(f"{'='*60}")
        print(f"  总样本数: {len(self.samples)}")
        print(f"  攻击比例: {self.attack_ratio:.1%}")
        print(f"\n  攻击类型分布:")
        
        for attack_type, count in attack_counts.items():
            percentage = count / len(self.samples) * 100
            print(f"    - {attack_type}: {count} ({percentage:.1f}%)")
        
        print(f"{'='*60}\n")
        
        return attack_counts


class SyntheticAttackDataset(Dataset):
    """
    合成攻击数据集
    
    用于没有真实NuScenes数据时的测试
    生成合成图像和点云，并应用攻击
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        attack_ratio: float = 0.5,
        num_points: int = 2048,
        image_size: Tuple[int, int] = (384, 640),
        attack_config: Dict = None
    ):
        """
        Args:
            num_samples: 样本数量
            attack_ratio: 攻击比例
            num_points: 点云点数
            image_size: 图像尺寸
            attack_config: 攻击配置
        """
        self.num_samples = num_samples
        self.attack_ratio = attack_ratio
        self.num_points = num_points
        self.image_size = image_size
        
        print(f"初始化合成攻击数据集")
        print(f"样本数: {num_samples}, 攻击比例: {attack_ratio:.1%}")
        
        # 初始化攻击生成器
        default_attack_config = {
            'attack_ratio': attack_ratio,
            'attack_weights': {
                'adversarial_patch': 0.25,
                'sensor_spoofing': 0.25,
                'physical_attack': 0.25,
                'data_poisoning': 0.25
            }
        }
        if attack_config:
            default_attack_config.update(attack_config)
        
        self.attack_generator = AttackGenerator(**default_attack_config)
        
        # 预生成攻击分配
        self._precompute_attack_assignments()
        
        # 图像变换
        self.image_transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _precompute_attack_assignments(self):
        """预计算攻击分配"""
        np.random.seed(42)
        
        self.attack_assignments = []
        for i in range(self.num_samples):
            if random.random() < self.attack_ratio:
                attack_type = self.attack_generator.select_attack_type()
                self.attack_assignments.append({
                    'is_attack': True,
                    'attack_type': attack_type
                })
            else:
                self.attack_assignments.append({
                    'is_attack': False,
                    'attack_type': 'normal'
                })
        
        np.random.seed(None)
    
    def _generate_synthetic_image(self) -> Image.Image:
        """生成合成道路场景图像"""
        h, w = self.image_size
        
        # 创建基础图像
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 天空（上半部分）
        sky_color = np.array([135, 206, 235])  # 天蓝色
        img[:h//2, :] = sky_color
        
        # 道路（下半部分）
        road_color = np.array([60, 60, 60])  # 深灰色
        img[h//2:, :] = road_color
        
        # 添加道路标线
        lane_color = np.array([255, 255, 255])
        center_x = w // 2
        
        # 中心虚线
        for y in range(h//2, h, 30):
            if y % 60 < 30:
                img[y:min(y+20, h), center_x-2:center_x+2] = lane_color
        
        # 边线
        img[h//2:, 10:15] = lane_color
        img[h//2:, w-15:w-10] = lane_color
        
        # 添加一些随机元素（模拟车辆、建筑等）
        for _ in range(random.randint(3, 8)):
            x = random.randint(0, w - 50)
            y = random.randint(h//3, h - 50)
            box_w = random.randint(30, 80)
            box_h = random.randint(20, 50)
            color = tuple(random.randint(50, 200) for _ in range(3))
            img[y:y+box_h, x:x+box_w] = color
        
        # 添加噪声使其更真实
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)
    
    def _generate_synthetic_pointcloud(self) -> np.ndarray:
        """生成合成点云"""
        points = []
        
        # 地面点
        n_ground = self.num_points // 2
        ground_x = np.random.uniform(-20, 40, n_ground)
        ground_y = np.random.uniform(-15, 15, n_ground)
        ground_z = np.random.normal(-1.5, 0.1, n_ground)
        points.append(np.stack([ground_x, ground_y, ground_z], axis=1))
        
        # 障碍物点（车辆、行人等）
        n_obstacles = self.num_points - n_ground
        for _ in range(random.randint(2, 5)):
            # 随机障碍物位置
            cx = random.uniform(5, 30)
            cy = random.uniform(-8, 8)
            cz = random.uniform(-0.5, 1.5)
            
            # 障碍物尺寸
            size_x = random.uniform(1, 4)
            size_y = random.uniform(1, 2)
            size_z = random.uniform(1, 2)
            
            n_obj_points = n_obstacles // 5
            obj_x = np.random.uniform(cx - size_x/2, cx + size_x/2, n_obj_points)
            obj_y = np.random.uniform(cy - size_y/2, cy + size_y/2, n_obj_points)
            obj_z = np.random.uniform(cz - size_z/2, cz + size_z/2, n_obj_points)
            points.append(np.stack([obj_x, obj_y, obj_z], axis=1))
        
        # 合并所有点
        all_points = np.vstack(points)
        
        # 采样到目标点数
        if len(all_points) > self.num_points:
            indices = np.random.choice(len(all_points), self.num_points, replace=False)
            all_points = all_points[indices]
        elif len(all_points) < self.num_points:
            pad_size = self.num_points - len(all_points)
            all_points = np.vstack([all_points, np.zeros((pad_size, 3))])
        
        return all_points.astype(np.float32)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成合成数据
        image = self._generate_synthetic_image()
        pointcloud = self._generate_synthetic_pointcloud()
        
        # 获取攻击分配
        assignment = self.attack_assignments[idx]
        
        # 应用攻击
        if assignment['is_attack']:
            attacked_image, attacked_points, attack_info = self.attack_generator.apply_attack(
                image, pointcloud, assignment['attack_type']
            )
            attack_type = assignment['attack_type']
        else:
            attacked_image = image
            attacked_points = pointcloud
            attack_info = {'attack_type': 'normal', 'is_attack': False}
            attack_type = 'normal'
        
        # 5分类标签 (0=normal, 1-4=各种攻击)
        from attacks.attack_generator import attack_type_to_label
        label = attack_type_to_label(attack_type)
        
        # 转换为tensor
        pointcloud_tensor = torch.from_numpy(attacked_points).float()
        label_tensor = torch.tensor(label, dtype=torch.long)  # 5分类标签 0-4
        
        return {
            'images': attacked_image,
            'pointclouds': pointcloud_tensor,
            'labels': label_tensor,  # 5分类标签
            'attack_types': attack_type,
            'attack_info': attack_info,
            'sample_tokens': f'synthetic_{idx}'
        }
    
    def get_statistics(self):
        """获取数据集统计"""
        attack_counts = {'normal': 0}
        for assignment in self.attack_assignments:
            attack_type = assignment['attack_type']
            if attack_type not in attack_counts:
                attack_counts[attack_type] = 0
            attack_counts[attack_type] += 1
        
        print(f"\n{'='*60}")
        print(f"合成数据集统计")
        print(f"{'='*60}")
        print(f"  总样本数: {self.num_samples}")
        print(f"  攻击比例: {self.attack_ratio:.1%}")
        print(f"\n  攻击类型分布:")
        
        for attack_type, count in attack_counts.items():
            percentage = count / self.num_samples * 100
            print(f"    - {attack_type}: {count} ({percentage:.1f}%)")
        
        print(f"{'='*60}\n")
        
        return attack_counts


def attack_collate_fn(batch):
    """攻击数据集的collate函数"""
    images = [item['images'] for item in batch]
    pointclouds = torch.stack([item['pointclouds'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    attack_types = [item['attack_types'] for item in batch]
    attack_infos = [item['attack_info'] for item in batch]
    sample_tokens = [item['sample_tokens'] for item in batch]
    
    return {
        'images': images,
        'pointclouds': pointclouds,
        'labels': labels,
        'attack_types': attack_types,
        'attack_infos': attack_infos,
        'sample_tokens': sample_tokens
    }


def create_attack_dataset(
    dataroot: str = None,
    version: str = 'v1.0-trainval',
    split: str = 'train',
    attack_ratio: float = 0.5,
    num_points: int = 2048,
    use_synthetic: bool = False,
    num_synthetic_samples: int = 1000,
    attack_config: Dict = None
) -> Dataset:
    """
    创建攻击数据集的工厂函数
    
    Args:
        dataroot: NuScenes数据根目录
        version: 数据集版本
        split: 数据集分割
        attack_ratio: 攻击比例
        num_points: 点云点数
        use_synthetic: 是否使用合成数据
        num_synthetic_samples: 合成数据样本数
        attack_config: 攻击配置
        
    Returns:
        Dataset: 攻击数据集
    """
    if use_synthetic or dataroot is None:
        print("使用合成攻击数据集")
        return SyntheticAttackDataset(
            num_samples=num_synthetic_samples,
            attack_ratio=attack_ratio,
            num_points=num_points,
            attack_config=attack_config
        )
    else:
        try:
            return NuScenesAttackDataset(
                dataroot=dataroot,
                version=version,
                split=split,
                attack_ratio=attack_ratio,
                num_points=num_points,
                attack_config=attack_config
            )
        except Exception as e:
            print(f"警告: 无法加载NuScenes数据集: {e}")
            print("回退到合成数据集")
            return SyntheticAttackDataset(
                num_samples=num_synthetic_samples,
                attack_ratio=attack_ratio,
                num_points=num_points,
                attack_config=attack_config
            )
