"""
攻击样本生成器

实现多种针对自动驾驶系统的攻击：
1. 对抗补丁攻击 - 在图像上添加对抗性补丁
2. LiDAR欺骗攻击 - 点云注入/删除
3. 物理攻击 - 遮挡、天气干扰、光照变化
4. 数据投毒攻击 - 数据污染
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from typing import Tuple, Optional, Dict, List, Union
import random
import math


class AdversarialPatchAttack:
    """
    对抗补丁攻击
    
    在图像上添加对抗性补丁，模拟：
    - 恶意贴纸攻击
    - 对抗性T恤/标志
    - 打印的对抗补丁
    """
    
    def __init__(
        self,
        patch_size_ratio: Tuple[float, float] = (0.1, 0.3),
        patch_types: List[str] = ['noise', 'pattern', 'gradient', 'checkerboard'],
        intensity: float = 0.8
    ):
        """
        Args:
            patch_size_ratio: 补丁大小相对于图像的比例范围 (min, max)
            patch_types: 补丁类型列表
            intensity: 攻击强度 (0-1)
        """
        self.patch_size_ratio = patch_size_ratio
        self.patch_types = patch_types
        self.intensity = intensity
    
    def generate_patch(self, size: Tuple[int, int], patch_type: str = None) -> np.ndarray:
        """生成对抗补丁"""
        if patch_type is None:
            patch_type = random.choice(self.patch_types)
        
        h, w = size
        
        if patch_type == 'noise':
            # 高频噪声补丁
            patch = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            
        elif patch_type == 'pattern':
            # 条纹图案
            patch = np.zeros((h, w, 3), dtype=np.uint8)
            stripe_width = max(2, h // 10)
            for i in range(0, h, stripe_width * 2):
                patch[i:i+stripe_width, :] = [255, 0, 0]  # 红色条纹
            for j in range(0, w, stripe_width * 2):
                patch[:, j:j+stripe_width] = np.maximum(
                    patch[:, j:j+stripe_width], 
                    [0, 255, 0]  # 绿色条纹
                )
                
        elif patch_type == 'gradient':
            # 渐变补丁
            patch = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    patch[i, j] = [
                        int(255 * i / h),
                        int(255 * j / w),
                        int(255 * (1 - i/h))
                    ]
                    
        elif patch_type == 'checkerboard':
            # 棋盘格补丁
            patch = np.zeros((h, w, 3), dtype=np.uint8)
            cell_size = max(4, min(h, w) // 8)
            for i in range(0, h, cell_size):
                for j in range(0, w, cell_size):
                    if ((i // cell_size) + (j // cell_size)) % 2 == 0:
                        patch[i:i+cell_size, j:j+cell_size] = [255, 255, 255]
                    else:
                        patch[i:i+cell_size, j:j+cell_size] = [0, 0, 0]
        else:
            # 默认随机噪声
            patch = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        
        return patch
    
    def apply(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, Dict]:
        """
        应用对抗补丁攻击
        
        Args:
            image: 输入图像
            
        Returns:
            attacked_image: 攻击后的图像
            attack_info: 攻击信息
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_w, img_h = image.size
        
        # 随机补丁大小
        ratio = random.uniform(*self.patch_size_ratio)
        patch_h = int(img_h * ratio)
        patch_w = int(img_w * ratio)
        
        # 随机位置（偏向图像中下部，模拟道路上的攻击）
        x = random.randint(0, max(0, img_w - patch_w))
        y = random.randint(img_h // 3, max(img_h // 3, img_h - patch_h))
        
        # 生成补丁
        patch_type = random.choice(self.patch_types)
        patch = self.generate_patch((patch_h, patch_w), patch_type)
        patch_img = Image.fromarray(patch)
        
        # 应用补丁
        result = image.copy()
        
        # 使用alpha混合
        alpha = self.intensity
        patch_rgba = patch_img.convert('RGBA')
        patch_rgba.putalpha(int(255 * alpha))
        
        result.paste(patch_img, (x, y), mask=patch_rgba.split()[3])
        
        attack_info = {
            'attack_type': 'adversarial_patch',
            'patch_type': patch_type,
            'position': (x, y),
            'size': (patch_w, patch_h),
            'intensity': self.intensity
        }
        
        return result, attack_info


class LiDARSpoofingAttack:
    """
    LiDAR欺骗攻击
    
    模拟：
    - 点云注入攻击（创建虚假障碍物）
    - 点云删除攻击（隐藏真实障碍物）
    - 点云扰动攻击（干扰点云精度）
    """
    
    def __init__(
        self,
        attack_modes: List[str] = ['inject', 'remove', 'perturb', 'shift'],
        inject_points_range: Tuple[int, int] = (50, 200),
        remove_ratio_range: Tuple[float, float] = (0.1, 0.4),
        perturb_std: float = 0.1,
        shift_range: float = 2.0
    ):
        """
        Args:
            attack_modes: 攻击模式列表
            inject_points_range: 注入点数范围
            remove_ratio_range: 删除点比例范围
            perturb_std: 扰动标准差
            shift_range: 位移范围（米）
        """
        self.attack_modes = attack_modes
        self.inject_points_range = inject_points_range
        self.remove_ratio_range = remove_ratio_range
        self.perturb_std = perturb_std
        self.shift_range = shift_range
    
    def inject_fake_object(
        self, 
        pointcloud: np.ndarray,
        object_type: str = 'vehicle'
    ) -> Tuple[np.ndarray, Dict]:
        """
        注入虚假物体点云
        
        Args:
            pointcloud: 原始点云 [N, 3] 或 [N, 4]
            object_type: 物体类型 ('vehicle', 'pedestrian', 'obstacle')
        """
        n_points = random.randint(*self.inject_points_range)
        
        # 在前方5-15米处注入虚假物体
        distance = random.uniform(5, 15)
        lateral_offset = random.uniform(-3, 3)  # 横向偏移
        
        if object_type == 'vehicle':
            # 车辆尺寸: 约4x2x1.5米
            length, width, height = 4.0, 2.0, 1.5
        elif object_type == 'pedestrian':
            # 行人尺寸: 约0.5x0.5x1.7米
            length, width, height = 0.5, 0.5, 1.7
        else:
            # 障碍物
            length, width, height = 1.0, 1.0, 0.5
        
        # 生成虚假物体点云
        fake_points = np.random.uniform(
            low=[-length/2 + distance, -width/2 + lateral_offset, -0.5],
            high=[length/2 + distance, width/2 + lateral_offset, height - 0.5],
            size=(n_points, 3)
        )
        
        # 如果原始点云有强度通道
        if pointcloud.shape[1] == 4:
            fake_intensity = np.random.uniform(0.3, 0.8, (n_points, 1))
            fake_points = np.hstack([fake_points, fake_intensity])
        
        # 合并点云
        attacked_pc = np.vstack([pointcloud, fake_points])
        
        attack_info = {
            'mode': 'inject',
            'object_type': object_type,
            'injected_points': n_points,
            'position': (distance, lateral_offset),
            'object_size': (length, width, height)
        }
        
        return attacked_pc, attack_info
    
    def remove_points(
        self,
        pointcloud: np.ndarray,
        region: str = 'front'
    ) -> Tuple[np.ndarray, Dict]:
        """
        删除点云（隐藏物体攻击）
        
        Args:
            pointcloud: 原始点云
            region: 删除区域 ('front', 'left', 'right', 'random_box')
        """
        remove_ratio = random.uniform(*self.remove_ratio_range)
        
        if region == 'front':
            # 删除前方区域的点
            mask = pointcloud[:, 0] > 5  # x > 5米
            mask &= pointcloud[:, 0] < 20
            mask &= np.abs(pointcloud[:, 1]) < 3  # |y| < 3米
        elif region == 'left':
            mask = pointcloud[:, 1] > 0
        elif region == 'right':
            mask = pointcloud[:, 1] < 0
        else:
            # 随机区域
            center_x = random.uniform(5, 15)
            center_y = random.uniform(-5, 5)
            radius = random.uniform(2, 5)
            distances = np.sqrt(
                (pointcloud[:, 0] - center_x)**2 + 
                (pointcloud[:, 1] - center_y)**2
            )
            mask = distances < radius
        
        # 在选定区域内随机删除一定比例的点
        region_indices = np.where(mask)[0]
        n_remove = int(len(region_indices) * remove_ratio)
        
        if n_remove > 0 and len(region_indices) > 0:
            remove_indices = np.random.choice(region_indices, n_remove, replace=False)
            keep_mask = np.ones(len(pointcloud), dtype=bool)
            keep_mask[remove_indices] = False
            attacked_pc = pointcloud[keep_mask]
        else:
            attacked_pc = pointcloud.copy()
            n_remove = 0
        
        attack_info = {
            'mode': 'remove',
            'region': region,
            'removed_points': n_remove,
            'remove_ratio': remove_ratio
        }
        
        return attacked_pc, attack_info
    
    def perturb_points(self, pointcloud: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        扰动点云坐标
        """
        noise = np.random.normal(0, self.perturb_std, pointcloud[:, :3].shape)
        attacked_pc = pointcloud.copy()
        attacked_pc[:, :3] += noise
        
        attack_info = {
            'mode': 'perturb',
            'noise_std': self.perturb_std,
            'affected_points': len(pointcloud)
        }
        
        return attacked_pc, attack_info
    
    def shift_points(self, pointcloud: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        整体位移点云（模拟GPS/IMU欺骗导致的标定错误）
        """
        shift = np.array([
            random.uniform(-self.shift_range, self.shift_range),
            random.uniform(-self.shift_range, self.shift_range),
            random.uniform(-0.5, 0.5)
        ])
        
        attacked_pc = pointcloud.copy()
        attacked_pc[:, :3] += shift
        
        attack_info = {
            'mode': 'shift',
            'shift_vector': shift.tolist(),
            'shift_magnitude': np.linalg.norm(shift)
        }
        
        return attacked_pc, attack_info
    
    def apply(
        self, 
        pointcloud: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, Dict]:
        """
        应用LiDAR欺骗攻击
        
        Args:
            pointcloud: 点云数据 [N, 3] 或 [N, 4]
            
        Returns:
            attacked_pc: 攻击后的点云
            attack_info: 攻击信息
        """
        if isinstance(pointcloud, torch.Tensor):
            pointcloud = pointcloud.numpy()
        
        attack_mode = random.choice(self.attack_modes)
        
        if attack_mode == 'inject':
            object_type = random.choice(['vehicle', 'pedestrian', 'obstacle'])
            attacked_pc, info = self.inject_fake_object(pointcloud, object_type)
        elif attack_mode == 'remove':
            region = random.choice(['front', 'left', 'right', 'random_box'])
            attacked_pc, info = self.remove_points(pointcloud, region)
        elif attack_mode == 'perturb':
            attacked_pc, info = self.perturb_points(pointcloud)
        elif attack_mode == 'shift':
            attacked_pc, info = self.shift_points(pointcloud)
        else:
            attacked_pc = pointcloud.copy()
            info = {'mode': 'none'}
        
        attack_info = {
            'attack_type': 'sensor_spoofing',
            **info
        }
        
        return attacked_pc, attack_info


class PhysicalAttack:
    """
    物理攻击
    
    模拟：
    - 遮挡攻击（贴纸、污渍）
    - 天气干扰（雨、雾、雪）
    - 光照攻击（强光、阴影）
    - 镜头污染
    """
    
    def __init__(
        self,
        attack_types: List[str] = ['occlusion', 'weather', 'lighting', 'lens_dirt'],
        intensity_range: Tuple[float, float] = (0.3, 0.7)
    ):
        self.attack_types = attack_types
        self.intensity_range = intensity_range
    
    def apply_occlusion(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """应用遮挡攻击"""
        result = image.copy()
        draw = ImageDraw.Draw(result)
        
        img_w, img_h = image.size
        
        # 随机遮挡类型
        occlusion_type = random.choice(['rectangle', 'circle', 'diagonal'])
        intensity = random.uniform(*self.intensity_range)
        
        if occlusion_type == 'rectangle':
            # 矩形遮挡（模拟贴纸）
            x1 = random.randint(0, img_w // 2)
            y1 = random.randint(0, img_h // 2)
            x2 = x1 + random.randint(img_w // 8, img_w // 3)
            y2 = y1 + random.randint(img_h // 8, img_h // 3)
            color = tuple(random.randint(0, 255) for _ in range(3))
            draw.rectangle([x1, y1, x2, y2], fill=color)
            
        elif occlusion_type == 'circle':
            # 圆形遮挡（模拟污渍）
            cx = random.randint(img_w // 4, 3 * img_w // 4)
            cy = random.randint(img_h // 4, 3 * img_h // 4)
            radius = random.randint(img_w // 10, img_w // 4)
            draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], 
                        fill=(50, 50, 50, int(255 * intensity)))
            
        elif occlusion_type == 'diagonal':
            # 对角线遮挡（模拟裂缝）
            points = [
                (0, random.randint(0, img_h)),
                (img_w, random.randint(0, img_h))
            ]
            draw.line(points, fill=(0, 0, 0), width=random.randint(5, 20))
        
        attack_info = {
            'occlusion_type': occlusion_type,
            'intensity': intensity
        }
        
        return result, attack_info
    
    def apply_weather(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """应用天气干扰"""
        weather_type = random.choice(['rain', 'fog', 'snow'])
        intensity = random.uniform(*self.intensity_range)
        
        result = image.copy()
        img_array = np.array(result)
        
        if weather_type == 'rain':
            # 雨天效果
            rain = np.zeros_like(img_array)
            n_drops = int(1000 * intensity)
            for _ in range(n_drops):
                x = random.randint(0, img_array.shape[1] - 1)
                y = random.randint(0, img_array.shape[0] - 1)
                length = random.randint(5, 15)
                if y + length < img_array.shape[0]:
                    rain[y:y+length, x] = [200, 200, 255]
            
            # 混合
            alpha = 0.3 * intensity
            img_array = (1 - alpha) * img_array + alpha * rain
            
            # 降低对比度
            result = Image.fromarray(img_array.astype(np.uint8))
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(1 - 0.3 * intensity)
            
        elif weather_type == 'fog':
            # 雾天效果
            fog_color = np.array([220, 220, 220])
            fog_layer = np.ones_like(img_array) * fog_color
            
            alpha = intensity * 0.6
            img_array = (1 - alpha) * img_array + alpha * fog_layer
            result = Image.fromarray(img_array.astype(np.uint8))
            
            # 模糊
            result = result.filter(ImageFilter.GaussianBlur(radius=2 * intensity))
            
        elif weather_type == 'snow':
            # 雪天效果
            snow = np.zeros_like(img_array)
            n_flakes = int(500 * intensity)
            for _ in range(n_flakes):
                x = random.randint(0, img_array.shape[1] - 1)
                y = random.randint(0, img_array.shape[0] - 1)
                size = random.randint(1, 4)
                x_end = min(x + size, img_array.shape[1])
                y_end = min(y + size, img_array.shape[0])
                snow[y:y_end, x:x_end] = [255, 255, 255]
            
            alpha = 0.4 * intensity
            img_array = (1 - alpha) * img_array + alpha * snow
            result = Image.fromarray(img_array.astype(np.uint8))
            
            # 提高亮度
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(1 + 0.2 * intensity)
        
        attack_info = {
            'weather_type': weather_type,
            'intensity': intensity
        }
        
        return result, attack_info
    
    def apply_lighting(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """应用光照攻击"""
        lighting_type = random.choice(['glare', 'shadow', 'underexposure', 'overexposure'])
        intensity = random.uniform(*self.intensity_range)
        
        result = image.copy()
        
        if lighting_type == 'glare':
            # 强光眩光
            img_array = np.array(result).astype(np.float32)
            img_w, img_h = image.size
            
            # 创建渐变眩光
            cx = random.randint(img_w // 4, 3 * img_w // 4)
            cy = random.randint(0, img_h // 3)
            
            y_coords, x_coords = np.ogrid[:img_h, :img_w]
            distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            max_dist = np.sqrt(img_w**2 + img_h**2) / 2
            
            glare = 1 - (distances / max_dist)
            glare = np.clip(glare, 0, 1) ** 2
            glare = glare[:, :, np.newaxis] * intensity * 255
            
            img_array = np.clip(img_array + glare, 0, 255)
            result = Image.fromarray(img_array.astype(np.uint8))
            
        elif lighting_type == 'shadow':
            # 阴影
            img_array = np.array(result).astype(np.float32)
            img_w, img_h = image.size
            
            # 创建阴影区域
            shadow_mask = np.ones((img_h, img_w))
            x_start = random.randint(0, img_w // 2)
            shadow_mask[:, x_start:] = 1 - intensity * 0.6
            
            shadow_mask = shadow_mask[:, :, np.newaxis]
            img_array = img_array * shadow_mask
            result = Image.fromarray(img_array.astype(np.uint8))
            
        elif lighting_type == 'underexposure':
            # 曝光不足
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(1 - intensity * 0.5)
            
        elif lighting_type == 'overexposure':
            # 过度曝光
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(1 + intensity * 0.5)
        
        attack_info = {
            'lighting_type': lighting_type,
            'intensity': intensity
        }
        
        return result, attack_info
    
    def apply_lens_dirt(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """应用镜头污染效果"""
        result = image.copy()
        img_array = np.array(result)
        img_h, img_w = img_array.shape[:2]
        
        intensity = random.uniform(*self.intensity_range)
        
        # 添加多个污点
        n_spots = random.randint(3, 8)
        for _ in range(n_spots):
            cx = random.randint(0, img_w)
            cy = random.randint(0, img_h)
            radius = random.randint(20, 80)
            
            y_coords, x_coords = np.ogrid[:img_h, :img_w]
            distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            
            # 创建模糊污点
            spot_mask = np.clip(1 - distances / radius, 0, 1)
            spot_mask = spot_mask[:, :, np.newaxis]
            
            # 降低清晰度
            blur_color = np.array([128, 128, 128])
            img_array = img_array * (1 - spot_mask * intensity * 0.5) + \
                       blur_color * spot_mask * intensity * 0.5
        
        result = Image.fromarray(img_array.astype(np.uint8))
        result = result.filter(ImageFilter.GaussianBlur(radius=1 + intensity))
        
        attack_info = {
            'n_spots': n_spots,
            'intensity': intensity
        }
        
        return result, attack_info
    
    def apply(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, Dict]:
        """
        应用物理攻击
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        attack_type = random.choice(self.attack_types)
        
        if attack_type == 'occlusion':
            result, info = self.apply_occlusion(image)
        elif attack_type == 'weather':
            result, info = self.apply_weather(image)
        elif attack_type == 'lighting':
            result, info = self.apply_lighting(image)
        elif attack_type == 'lens_dirt':
            result, info = self.apply_lens_dirt(image)
        else:
            result = image.copy()
            info = {}
        
        attack_info = {
            'attack_type': 'physical_attack',
            'sub_type': attack_type,
            **info
        }
        
        return result, attack_info


class DataPoisoningAttack:
    """
    数据投毒攻击
    
    模拟：
    - 像素级扰动
    - 颜色通道攻击
    - 触发器注入（后门攻击）
    """
    
    def __init__(
        self,
        attack_types: List[str] = ['pixel_perturb', 'channel_shuffle', 'trigger', 'blend'],
        epsilon: float = 0.03,
        trigger_size: int = 30
    ):
        self.attack_types = attack_types
        self.epsilon = epsilon
        self.trigger_size = trigger_size
    
    def pixel_perturbation(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """像素级扰动（类似FGSM）"""
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # 添加小幅度扰动
        perturbation = np.random.uniform(-self.epsilon, self.epsilon, img_array.shape)
        perturbed = np.clip(img_array + perturbation, 0, 1)
        
        result = Image.fromarray((perturbed * 255).astype(np.uint8))
        
        attack_info = {
            'perturbation_type': 'uniform',
            'epsilon': self.epsilon
        }
        
        return result, attack_info
    
    def channel_shuffle(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """颜色通道攻击"""
        img_array = np.array(image)
        
        shuffle_type = random.choice(['swap', 'scale', 'shift'])
        
        if shuffle_type == 'swap':
            # 交换通道
            channels = list(range(3))
            random.shuffle(channels)
            img_array = img_array[:, :, channels]
            
        elif shuffle_type == 'scale':
            # 缩放某个通道
            channel = random.randint(0, 2)
            scale = random.uniform(0.5, 1.5)
            img_array[:, :, channel] = np.clip(img_array[:, :, channel] * scale, 0, 255)
            
        elif shuffle_type == 'shift':
            # 偏移通道值
            channel = random.randint(0, 2)
            shift = random.randint(-30, 30)
            img_array[:, :, channel] = np.clip(
                img_array[:, :, channel].astype(np.int32) + shift, 0, 255
            ).astype(np.uint8)
        
        result = Image.fromarray(img_array)
        
        attack_info = {
            'shuffle_type': shuffle_type
        }
        
        return result, attack_info
    
    def add_trigger(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """添加触发器（后门攻击）"""
        result = image.copy()
        draw = ImageDraw.Draw(result)
        
        img_w, img_h = image.size
        
        # 触发器位置（通常在角落）
        corner = random.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
        
        if corner == 'top_left':
            x, y = 5, 5
        elif corner == 'top_right':
            x, y = img_w - self.trigger_size - 5, 5
        elif corner == 'bottom_left':
            x, y = 5, img_h - self.trigger_size - 5
        else:
            x, y = img_w - self.trigger_size - 5, img_h - self.trigger_size - 5
        
        # 绘制触发器（简单的棋盘格图案）
        cell_size = self.trigger_size // 4
        for i in range(4):
            for j in range(4):
                color = (255, 255, 255) if (i + j) % 2 == 0 else (0, 0, 0)
                draw.rectangle([
                    x + i * cell_size, y + j * cell_size,
                    x + (i + 1) * cell_size, y + (j + 1) * cell_size
                ], fill=color)
        
        attack_info = {
            'trigger_position': corner,
            'trigger_size': self.trigger_size
        }
        
        return result, attack_info
    
    def blend_attack(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """混合攻击（与噪声图像混合）"""
        img_array = np.array(image).astype(np.float32)
        
        # 生成噪声图像
        noise_type = random.choice(['gaussian', 'uniform', 'salt_pepper'])
        
        if noise_type == 'gaussian':
            noise = np.random.normal(128, 50, img_array.shape)
        elif noise_type == 'uniform':
            noise = np.random.uniform(0, 255, img_array.shape)
        else:
            noise = np.zeros_like(img_array)
            salt_mask = np.random.random(img_array.shape[:2]) < 0.02
            pepper_mask = np.random.random(img_array.shape[:2]) < 0.02
            noise[salt_mask] = 255
            noise[pepper_mask] = 0
        
        # 混合
        alpha = random.uniform(0.05, 0.15)
        blended = (1 - alpha) * img_array + alpha * noise
        blended = np.clip(blended, 0, 255)
        
        result = Image.fromarray(blended.astype(np.uint8))
        
        attack_info = {
            'noise_type': noise_type,
            'blend_alpha': alpha
        }
        
        return result, attack_info
    
    def apply(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, Dict]:
        """应用数据投毒攻击"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        attack_type = random.choice(self.attack_types)
        
        if attack_type == 'pixel_perturb':
            result, info = self.pixel_perturbation(image)
        elif attack_type == 'channel_shuffle':
            result, info = self.channel_shuffle(image)
        elif attack_type == 'trigger':
            result, info = self.add_trigger(image)
        elif attack_type == 'blend':
            result, info = self.blend_attack(image)
        else:
            result = image.copy()
            info = {}
        
        attack_info = {
            'attack_type': 'data_poisoning',
            'sub_type': attack_type,
            **info
        }
        
        return result, attack_info


class AttackGenerator:
    """
    统一的攻击生成器
    
    集成所有攻击类型，提供统一接口
    """
    
    ATTACK_TYPES = ['adversarial_patch', 'sensor_spoofing', 'physical_attack', 'data_poisoning']
    
    def __init__(
        self,
        attack_ratio: float = 0.5,
        attack_weights: Dict[str, float] = None,
        **kwargs
    ):
        """
        Args:
            attack_ratio: 生成攻击样本的比例 (0-1)
            attack_weights: 各攻击类型的权重，默认均匀分布
        """
        self.attack_ratio = attack_ratio
        
        # 默认权重
        if attack_weights is None:
            attack_weights = {
                'adversarial_patch': 0.25,
                'sensor_spoofing': 0.25,
                'physical_attack': 0.25,
                'data_poisoning': 0.25
            }
        self.attack_weights = attack_weights
        
        # 初始化各攻击生成器
        self.adversarial_patch = AdversarialPatchAttack(
            **kwargs.get('adversarial_patch_config', {})
        )
        self.lidar_spoofing = LiDARSpoofingAttack(
            **kwargs.get('lidar_spoofing_config', {})
        )
        self.physical_attack = PhysicalAttack(
            **kwargs.get('physical_attack_config', {})
        )
        self.data_poisoning = DataPoisoningAttack(
            **kwargs.get('data_poisoning_config', {})
        )
    
    def should_attack(self) -> bool:
        """根据attack_ratio决定是否生成攻击样本"""
        return random.random() < self.attack_ratio
    
    def select_attack_type(self) -> str:
        """根据权重选择攻击类型"""
        types = list(self.attack_weights.keys())
        weights = list(self.attack_weights.values())
        return random.choices(types, weights=weights, k=1)[0]
    
    def apply_attack(
        self,
        image: Union[Image.Image, np.ndarray],
        pointcloud: Union[np.ndarray, torch.Tensor],
        attack_type: str = None
    ) -> Tuple[Image.Image, np.ndarray, Dict]:
        """
        应用攻击
        
        Args:
            image: 输入图像
            pointcloud: 输入点云
            attack_type: 指定攻击类型，None则随机选择
            
        Returns:
            attacked_image: 攻击后的图像
            attacked_pointcloud: 攻击后的点云
            attack_info: 攻击信息
        """
        if attack_type is None:
            attack_type = self.select_attack_type()
        
        # 转换格式
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(pointcloud, torch.Tensor):
            pointcloud = pointcloud.numpy()
        
        attack_info = {
            'attack_type': attack_type,
            'is_attack': True
        }
        
        if attack_type == 'adversarial_patch':
            # 对抗补丁只影响图像
            attacked_image, info = self.adversarial_patch.apply(image)
            attacked_pc = pointcloud.copy()
            attack_info.update(info)
            
        elif attack_type == 'sensor_spoofing':
            # LiDAR欺骗只影响点云
            attacked_image = image.copy()
            attacked_pc, info = self.lidar_spoofing.apply(pointcloud)
            attack_info.update(info)
            
        elif attack_type == 'physical_attack':
            # 物理攻击影响图像，可能也影响点云
            attacked_image, info = self.physical_attack.apply(image)
            attack_info.update(info)
            
            # 如果是天气攻击，也对点云添加噪声
            if info.get('sub_type') == 'weather':
                noise_std = info.get('intensity', 0.5) * 0.05
                noise = np.random.normal(0, noise_std, pointcloud[:, :3].shape)
                attacked_pc = pointcloud.copy()
                attacked_pc[:, :3] += noise
            else:
                attacked_pc = pointcloud.copy()
                
        elif attack_type == 'data_poisoning':
            # 数据投毒主要影响图像
            attacked_image, info = self.data_poisoning.apply(image)
            attacked_pc = pointcloud.copy()
            attack_info.update(info)
            
        else:
            attacked_image = image.copy()
            attacked_pc = pointcloud.copy()
            attack_info['attack_type'] = 'unknown'
        
        return attacked_image, attacked_pc, attack_info
    
    def generate(
        self,
        image: Union[Image.Image, np.ndarray],
        pointcloud: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Image.Image, np.ndarray, int, str, Dict]:
        """
        生成样本（可能是攻击样本或正常样本）
        
        Args:
            image: 输入图像
            pointcloud: 输入点云
            
        Returns:
            output_image: 输出图像
            output_pointcloud: 输出点云
            label: 标签 (0: 正常, 1: 攻击)
            attack_type: 攻击类型 (正常样本为'normal')
            attack_info: 攻击详细信息
        """
        if self.should_attack():
            attack_type = self.select_attack_type()
            attacked_image, attacked_pc, attack_info = self.apply_attack(
                image, pointcloud, attack_type
            )
            return attacked_image, attacked_pc, 1, attack_type, attack_info
        else:
            # 正常样本
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if isinstance(pointcloud, torch.Tensor):
                pointcloud = pointcloud.numpy()
            
            return image.copy(), pointcloud.copy(), 0, 'normal', {
                'attack_type': 'normal',
                'is_attack': False
            }
    
    def get_attack_statistics(self) -> Dict:
        """获取攻击类型统计信息"""
        return {
            'attack_ratio': self.attack_ratio,
            'attack_weights': self.attack_weights,
            'available_attacks': self.ATTACK_TYPES
        }


# ==================== 攻击类别映射 ====================

# 5分类标签
ATTACK_CLASSES = {
    'normal': 0,
    'adversarial_patch': 1,
    'sensor_spoofing': 2,
    'physical_attack': 3,
    'data_poisoning': 4
}

ATTACK_NAMES = ['normal', 'adversarial_patch', 'sensor_spoofing', 'physical_attack', 'data_poisoning']

NUM_CLASSES = 5


def attack_type_to_label(attack_type: str) -> int:
    """将攻击类型字符串转换为数字标签 (0-4)"""
    if attack_type is None:
        return 0
    return ATTACK_CLASSES.get(attack_type, 0)


def label_to_attack_type(label: int) -> str:
    """将数字标签转换为攻击类型字符串"""
    if 0 <= label < len(ATTACK_NAMES):
        return ATTACK_NAMES[label]
    return 'unknown'


def is_attack_label(label: int) -> bool:
    """判断标签是否为攻击（非0即为攻击）"""
    return label > 0
