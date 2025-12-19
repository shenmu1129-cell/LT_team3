import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
import torchvision.transforms as T
from nuscenes.utils.splits import create_splits_scenes
import os
from datetime import datetime
from torch.utils.data import DataLoader
import json
import re


class Qwen3VLDefenseSystem(nn.Module):
    """基于Qwen3-VL的自动驾驶攻击检测与防御系统"""
    
    def __init__(self, 
                 pointcloud_dim=1024,
                #  qwen_hidden_dim=1536,  # Qwen3-VL-2B的隐藏层维度
                qwen_hidden_dim=3072,  # Qwen3-VL-2B的隐藏层维度
                 model_name=r'/home/sutongtong/wwt/model/Qwen3-VL-2B-Instruct'):
        super().__init__()
        
        # 1. 加载Qwen3-VL模型和processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256*32*32,  # Qwen3-VL使用32的倍数
            max_pixels=1280*32*32,
        )
        
        self.qwen_vl = AutoModelForImageTextToText.from_pretrained(
            model_name,
            # torch_dtype=torch.bfloat16,  # Qwen3-VL推荐使用bfloat16
            dtype=torch.bfloat16,  # Qwen3-VL推荐使用bfloat16
            device_map="auto",
        )
        
        # 2. 点云编码器
        self.pointcloud_encoder = PointCloudEncoder(output_dim=pointcloud_dim)
        
        # 3. 点云特征适配器 - 将点云特征映射到Qwen3-VL的视觉特征空间
        self.pointcloud_adapter = PointCloudAdapter(
            pointcloud_dim=pointcloud_dim,
            qwen_hidden_dim=qwen_hidden_dim
        )
        
        # 4. 可训练的分类头（用于训练）
        self.classifier = nn.Sequential(
            # nn.Linear(qwen_hidden_dim + pointcloud_dim, 512),
            nn.Linear(qwen_hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  # 二分类输出logit
        )
        
        # 冻结Qwen3-VL的大部分参数
        for param in self.qwen_vl.parameters():
            param.requires_grad = False
        
        # 可选: 使用LoRA微调
        self._setup_lora()
        
    def _setup_lora(self):
        """设置LoRA微调 (可选)"""
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen3的attention模块
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.qwen_vl = get_peft_model(self.qwen_vl, lora_config)
        self.qwen_vl.print_trainable_parameters()
    
    def forward(self, images, pointclouds, mode='detect', attack_type=None):
        """
        Args:
            images: [B, 3, H, W] 或 PIL Image list
            pointclouds: [B, N, 3]
            mode: 'train' (训练), 'detect' (检测) 或 'defend' (防御)
            attack_type: 攻击类型 (用于防御模式)
        Returns:
            训练模式: logits [B, 1]
            检测模式: 检测结果列表
            防御模式: 防御策略列表
        """
        batch_size = pointclouds.size(0) if torch.is_tensor(pointclouds) else len(pointclouds)
        
        # 1. 处理点云特征
        if torch.is_tensor(pointclouds):
            pc_features = self.pointcloud_encoder(pointclouds)  # [B, pointcloud_dim]
            pc_embeddings = self.pointcloud_adapter(pc_features)  # [B, num_tokens, qwen_hidden_dim]
        
        # 2. 根据模式选择不同的处理方式
        if mode == 'train':
            # 训练模式：使用分类头
            return self._training_forward(images, pc_features, batch_size)
        elif mode == 'detect':
            # 检测模式：使用Qwen3-VL生成文本
            results = self._detection_mode(images, pc_embeddings, batch_size)
        elif mode == 'defend':
            # 防御模式：使用Qwen3-VL生成防御策略
            results = self._defense_mode(images, pc_embeddings, attack_type, batch_size)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return results
    
    def _training_forward(self, images, pc_features, batch_size):
        """训练模式的前向传播 - 使用分类头 (批量优化版)"""
        # 1. 准备 PIL 图像列表
        pil_images = []
        for i in range(batch_size):
            if isinstance(images, list):
                pil_images.append(images[i])
            else:
                pil_images.append(transforms.ToPILImage()(images[i].cpu()))
        
        # 2. 批量提取视觉特征 [B, vision_dim]
        vision_features = self._extract_vision_features(pil_images)
        
        # 3. 拼接特征 [B, vision_dim] + [B, pc_dim] -> [B, combined_dim]
        pc_features = pc_features.to(vision_features.device)
        combined_features = torch.cat([vision_features, pc_features], dim=1)
        
        # 调试信息（仅在第一次运行时打印）
        if not hasattr(self, "_batched_debug_done"):
            print(f"\n[调试] 批量特征维度信息:")
            print(f"  vision_features: {vision_features.shape}")
            print(f"  pc_features: {pc_features.shape}")
            print(f"  combined_features: {combined_features.shape}")
            print(f"  分类头期望输入: {self.classifier[0].in_features}")
            self._batched_debug_done = True
            
        # 4. 通过分类头得到 logits [B, 1]
        classifier_device = next(self.classifier.parameters()).device
        logits = self.classifier(combined_features.to(classifier_device))
        
        return logits

    def _extract_vision_features(self, images):
        """批量提取Qwen3-VL的视觉特征（用于训练，保留梯度）"""
        if not isinstance(images, list):
            images = [images]
            
        # 构建批量消息
        all_messages = []
        for img in images:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Analyze this image."},
                ],
            }]
            all_messages.append(messages)
        
        # 批量处理输入
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in all_messages]
        image_inputs, video_inputs = process_vision_info(all_messages)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # 移动到模型所在设备
        inputs = {k: v.to(self.qwen_vl.device) for k, v in inputs.items()}
        
        # 前向传播获取隐藏状态（保留梯度）
        with torch.set_grad_enabled(self.training):
            outputs = self.qwen_vl.model(**inputs, output_hidden_states=True)
            # 取最后一层隐藏状态 [B, seq_len, hidden_dim]
            hidden_states = outputs.hidden_states[-1]
            # 使用平均池化得到全局特征 [B, hidden_dim]
            vision_features = hidden_states.mean(dim=1)
        
        return vision_features
    
    def _detection_mode(self, images, pc_embeddings, batch_size):
        """攻击检测模式"""
        results = []
        
        for i in range(batch_size):
            # 获取单个样本
            if isinstance(images, list):
                image = images[i]
            else:
                image = transforms.ToPILImage()(images[i].cpu())
            
            pc_embed = pc_embeddings[i] if pc_embeddings is not None else None
            
            # 构建检测prompt - Qwen3-VL使用新的对话格式
            messages = self._build_detection_messages(image, pc_embed)
            
            # Qwen2-VL推理
            response = self._generate_response(messages, image)
            
            # 解析结果
            parsed_result = self._parse_detection_result(response)
            results.append(parsed_result)
        
        return results
    
    def _defense_mode(self, images, pc_embeddings, attack_type, batch_size):
        """防御策略生成模式"""
        results = []
        
        for i in range(batch_size):
            if isinstance(images, list):
                image = images[i]
            else:
                image = transforms.ToPILImage()(images[i].cpu())
            
            pc_embed = pc_embeddings[i] if pc_embeddings is not None else None
            
            # 构建防御prompt
            messages = self._build_defense_messages(image, pc_embed, attack_type)
            
            # Qwen2-VL推理
            response = self._generate_response(messages, image)
            
            results.append(response)
        
        return results
    
    def _build_detection_messages(self, image, pc_embed=None):
        """构建攻击检测的消息列表 (Qwen3-VL格式)"""
        
        query_text = """请仔细分析这张自动驾驶场景图像,检测是否存在对抗攻击或异常。

可能的攻击类型包括:
1. **对抗样本攻击** (Adversarial Patch): 图像中是否有异常的贴纸、标记或扰动?
2. **传感器欺骗攻击** (Sensor Spoofing): 场景中是否有伪造的交通标志、信号灯或障碍物?
3. **物理攻击** (Physical Attack): 是否有人为放置的干扰物体?
4. **数据完整性攻击**: 图像是否有篡改、模糊或失真的痕迹?

"""
        
        # 如果有点云信息,添加描述
        if pc_embed is not None:
            query_text += """
同时,激光雷达点云数据显示以下特征:
- 点云密度和分布已编码为特征向量
- 请结合图像和点云的一致性进行判断

"""
        
        query_text += """请按以下JSON格式严格回答:
{
    "is_attack": true/false,
    "attack_type": "攻击类型名称" 或 null,
    "confidence": 0.0-1.0,
    "risk_level": "低/中/高/紧急",
    "suspicious_regions": ["区域1描述", "区域2描述"],
    "analysis": "详细分析判断依据,包括视觉线索和异常点"
}

请直接输出JSON,不要有其他内容:"""
        
        # Qwen3-VL的新对话格式
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": query_text},
                ],
            }
        ]
        
        return messages
    
    def _build_defense_messages(self, image, pc_embed, attack_type):
        """构建防御策略的消息列表 (Qwen3-VL格式)"""
        
        query_text = f"""检测到自动驾驶系统受到攻击!

**攻击类型**: {attack_type if attack_type else "未知攻击"}

请基于这张场景图像,提供详细的应对方案:

1. **立即响应措施** (0-5秒内执行)
   - 紧急制动或避让策略
   - 传感器切换方案
   
2. **数据净化方案** (实时处理)
   - 如何识别和过滤被污染的数据
   - 传感器融合策略调整

3. **系统恢复步骤** (5-30秒内)
   - 如何恢复正常感知
   - 备用系统启动流程

4. **预防性建议** (长期改进)
   - 针对该类攻击的防御加固
   - 检测算法优化建议

请给出具体、可执行的技术方案:"""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": query_text},
                ],
            }
        ]
        
        return messages
    
    def _generate_response(self, messages, image):
        """使用Qwen3-VL生成响应"""
        # 准备输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 处理图像和文本
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.qwen_vl.device)
        
        # 生成响应
        with torch.no_grad():
            generated_ids = self.qwen_vl.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
    
    def _parse_detection_result(self, response_text):
        """解析检测结果"""
        
        # 尝试提取JSON
        try:
            # 先尝试直接解析
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            pass
        
        # 方法2: 查找JSON对象
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx+1]
                result = json.loads(json_str)
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        
        # 方法3: 使用正则表达式
        try:
            pattern = re.compile(r'\{[^\{\}]*\}', re.DOTALL)
            matches = pattern.findall(response_text)
            
            for match in matches:
                try:
                    result = json.loads(match)
                    if 'is_attack' in result:
                        return result
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        # 如果所有JSON解析都失败,使用规则解析
        result = {
            'is_attack': False,
            'attack_type': None,
            'confidence': 0.5,
            'risk_level': 'low',
            'suspicious_regions': [],
            'analysis': response_text[:200] if response_text else "无法解析响应"
        }
        
        # 尝试从文本中提取关键信息
        if '攻击' in response_text or 'attack' in response_text.lower():
            result['is_attack'] = True
            result['confidence'] = 0.7
            result['risk_level'] = 'medium'
        
        return result


class PointCloudEncoder(nn.Module):
    """点云编码器"""
    
    def __init__(self, output_dim=1024, num_points=2048):
        super().__init__()
        
        # PointNet风格的编码器
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        # 全局特征
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: [B, N, 3] 点云
        Returns:
            [B, output_dim] 全局特征
        """
        # 转置为 [B, 3, N]
        x = x.transpose(1, 2)
        
        # 点特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        
        # 全局最大池化
        x = torch.max(x, dim=2)[0]  # [B, 512]
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class PointCloudAdapter(nn.Module):
    """点云特征适配器 - 将点云特征映射到Qwen3-VL的视觉特征空间"""
    
    def __init__(self, pointcloud_dim=1024, qwen_hidden_dim=1536, num_tokens=64):
        super().__init__()
        
        self.num_tokens = num_tokens
        
        # 特征投影
        self.projector = nn.Sequential(
            nn.Linear(pointcloud_dim, qwen_hidden_dim * 2),
            nn.LayerNorm(qwen_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(qwen_hidden_dim * 2, qwen_hidden_dim * num_tokens),
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_tokens, qwen_hidden_dim) * 0.02
        )
        
    def forward(self, pc_features):
        """
        Args:
            pc_features: [B, pointcloud_dim]
        Returns:
            [B, num_tokens, qwen_hidden_dim]
        """
        B = pc_features.size(0)
        
        # 投影到token序列
        x = self.projector(pc_features)  # [B, qwen_hidden_dim * num_tokens]
        x = x.view(B, self.num_tokens, -1)  # [B, num_tokens, qwen_hidden_dim]
        
        # 添加位置编码
        x = x + self.pos_embedding
        
        return x


class QwenVLTrainer:
    """训练器"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=1e-4, weight_decay=0.01):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 只优化可训练的参数
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        print(f"\n可训练参数数量: {sum(p.numel() for p in trainable_params):,}")
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 20,
            eta_min=1e-6
        )
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['images']
            pointclouds = batch['pointclouds'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播 - 使用训练模式，返回logits
            logits = self.model(images, pointclouds, mode='train')  # [B, 1]
            
            # 计算损失（logits有梯度）
            loss = self.criterion(logits.squeeze(-1), labels.float())
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            # 将logits转换为概率
            predictions = torch.sigmoid(logits.squeeze(-1))  # [B]
            predicted = (predictions > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"Acc: {100.*correct/total:.2f}%")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self):
        """验证 - 使用分类头快速验证"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            images = batch['images']
            pointclouds = batch['pointclouds'].to(self.device)
            labels = batch['labels']
            
            # 使用训练模式的分类头进行快速验证
            logits = self.model(images, pointclouds, mode='train')  # [B, 1]
            
            # 转换为预测
            predictions = torch.sigmoid(logits.squeeze(-1))  # [B]
            predicted = (predictions > 0.5).long()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 计算指标
        accuracy = (all_preds == all_labels).mean()
        
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"验证 - Acc: {accuracy:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return accuracy, precision, recall, f1


class NuScenesMiniDataset(Dataset):
    """nuScenes Mini数据集"""
    
    def __init__(self, dataroot, version='v1.0-trainval', split='train', 
                 attack_ratio=0.3, num_points=2048, use_cache=False):
        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.attack_ratio = attack_ratio
        self.num_points = num_points
        self.use_cache = use_cache
        
        print(f"初始化NuScenes数据集: {split}")
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
        for scene in self.nusc.scene:
            if scene['name'] in scene_names:
                sample_token = scene['first_sample_token']
                while sample_token:
                    self.samples.append(sample_token)
                    sample = self.nusc.get('sample', sample_token)
                    sample_token = sample['next']
        
        print(f"找到 {len(self.samples)} 个样本")
        
        # 生成攻击标签
        self._generate_attack_labels()
        
        # 图像变换
        self.image_transform = T.Compose([
            T.Resize((384, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _generate_attack_labels(self):
        """生成模拟攻击标签"""
        num_attacks = int(len(self.samples) * self.attack_ratio)
        attack_indices = np.random.choice(len(self.samples), num_attacks, replace=False)
        
        self.attack_labels = np.zeros(len(self.samples), dtype=np.int64)
        self.attack_labels[attack_indices] = 1
        
        attack_types = ['adversarial_patch', 'sensor_spoofing', 'physical_attack', 'data_poisoning']
        self.attack_types = []
        for i in range(len(self.samples)):
            if self.attack_labels[i] == 1:
                self.attack_types.append(np.random.choice(attack_types))
            else:
                self.attack_types.append(None)
    
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
        pc = LidarPointCloud.from_file(lidar_path)
        points = pc.points[:3, :].T  # [N, 3]
        
        # 采样固定数量的点
        if points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[indices]
        elif points.shape[0] < self.num_points:
            pad_size = self.num_points - points.shape[0]
            points = np.vstack([points, np.zeros((pad_size, 3))])
        
        # 转换为tensor
        pointcloud = torch.from_numpy(points).float()
        label = torch.tensor(self.attack_labels[idx], dtype=torch.long)
        
        return {
            'images': image,  # PIL Image,不转换为tensor
            'pointclouds': pointcloud,
            'labels': label,
            'attack_types': self.attack_types[idx],
            'sample_tokens': sample_token
        }
    
    def get_statistics(self):
        """获取数据集统计信息"""
        print(f"\n数据集统计 ({self.split}):")
        print(f"  总样本数: {len(self.samples)}")
        print(f"  正常样本: {(self.attack_labels == 0).sum()}")
        print(f"  攻击样本: {(self.attack_labels == 1).sum()}")
        print(f"  攻击比例: {self.attack_ratio:.2%}")


def custom_collate_fn(batch):
    """自定义collate函数"""
    images = [item['images'] for item in batch]
    pointclouds = torch.stack([item['pointclouds'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    attack_types = [item['attack_types'] for item in batch]
    sample_tokens = [item['sample_tokens'] for item in batch]
    
    return {
        'images': images,
        'pointclouds': pointclouds,
        'labels': labels,
        'attack_types': attack_types,
        'sample_tokens': sample_tokens
    }


def train_on_mini():
    """在nuScenes mini数据集上训练"""
    
    config = {
        'dataroot': r'/home/sutongtong/LanTu_team3/dataset/nuScenes/train',
        'version': 'v1.0-trainval',
        'batch_size': 1,  # Qwen3-VL显存占用较大,建议batch_size=1
        'num_workers': 2,
        'epochs': 20,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'attack_ratio': 0.3,
        'num_points': 2048,
        'save_dir': './checkpoints_qwen3vl',
        'log_dir': './logs_qwen3vl'
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print("="*60)
    print("nuScenes Mini 攻击检测系统训练 (Qwen3-VL)")
    print("="*60)
    print(f"配置: {json.dumps(config, indent=2)}")
    
    print("\n加载数据集...")
    
    train_dataset = NuScenesMiniDataset(
        dataroot=config['dataroot'],
        version=config['version'],
        split='train',
        attack_ratio=config['attack_ratio'],
        num_points=config['num_points'],
        use_cache=True
    )
    
    val_dataset = NuScenesMiniDataset(
        dataroot=config['dataroot'],
        version=config['version'],
        split='val',
        attack_ratio=config['attack_ratio'],
        num_points=config['num_points'],
        use_cache=True
    )
    
    train_dataset.get_statistics()
    val_dataset.get_statistics()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    print("\n初始化模型...")
    
    model = Qwen3VLDefenseSystem(
        pointcloud_dim=1024,
        # qwen_hidden_dim=1536,
        qwen_hidden_dim=3072,
        model_name="Qwen/Qwen3-VL-2B-Instruct"
    )
    
    trainer = QwenVLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n开始训练...")
    print("="*60)
    
    best_f1 = 0
    best_acc = 0
    patience = 0
    max_patience = 5
    
    training_log = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 60)
        
        train_loss, train_acc = trainer.train_epoch(epoch)
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        val_acc, val_precision, val_recall, val_f1 = trainer.validate()
        
        training_log['train_loss'].append(train_loss)
        training_log['train_acc'].append(train_acc)
        training_log['val_acc'].append(val_acc)
        training_log['val_precision'].append(val_precision)
        training_log['val_recall'].append(val_recall)
        training_log['val_f1'].append(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            patience = 0
            
            save_path = os.path.join(config['save_dir'], 'best_model.pth')
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_f1': best_f1,
                'best_acc': best_acc,
                'config': config
            }
            torch.save(save_dict, save_path)
            print(f"✓ 保存最佳模型 (F1: {best_f1:.4f}, Acc: {best_acc:.4f})")
        else:
            patience += 1
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                config['save_dir'], 
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
            }, checkpoint_path)
            print(f"✓ 保存checkpoint: {checkpoint_path}")
        
        if patience >= max_patience:
            print(f"\n早停触发 ({max_patience}个epoch无改善)")
            break
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"最佳F1分数: {best_f1:.4f}")
    print(f"最佳准确率: {best_acc:.4f}")
    
    log_path = os.path.join(config['log_dir'], 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)
    print(f"训练日志已保存: {log_path}")
    
    plot_training_curves(training_log, config['log_dir'])
    
    return model, training_log


def plot_training_curves(log, save_dir):
    """绘制训练曲线"""
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(log['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(epochs, log['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, log['train_acc'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, log['val_acc'], 'r-', label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(epochs, log['val_precision'], 'g-', label='Precision')
    axes[1, 0].plot(epochs, log['val_recall'], 'orange', label='Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, log['val_f1'], 'purple', label='F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存: {save_path}")
    plt.close()


def quick_test():
    """快速测试数据加载"""
    print("快速测试数据加载...")
    
    dataset = NuScenesMiniDataset(
        dataroot=r'/home/sutongtong/LanTu_team3/dataset/nuScenes/train',
        version='v1.0-trainval',
        split='train',
        attack_ratio=0.3,
        use_cache=False
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    sample = dataset[0]
    
    print(f"\n样本信息:")
    print(f"  图像尺寸: {sample['images'].size}")
    print(f"  点云形状: {sample['pointclouds'].shape}")
    print(f"  标签: {sample['labels'].item()} ({'攻击' if sample['labels'].item() == 1 else '正常'})")
    print(f"  攻击类型: {sample['attack_types']}")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(sample['images'])
    axes[0].set_title(f"Image - {'Attack' if sample['labels'].item() == 1 else 'Normal'}")
    axes[0].axis('off')
    
    pc = sample['pointclouds'].numpy()
    axes[1].scatter(pc[:, 0], pc[:, 1], c=pc[:, 2], s=1, cmap='viridis')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].set_title('Point Cloud (BEV)')
    axes[1].axis('equal')
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='Z (m)')
    
    plt.tight_layout()
    plt.savefig('sample_visualization_qwen3vl.png', dpi=150)
    print("\n样本可视化已保存: sample_visualization_qwen3vl.png")
    plt.show()


if __name__ == "__main__":
    import sys
    
    # 检查是否启用联邦学习模式
    if '--federated' in sys.argv or os.getenv('FEDERATED_MODE') == '1':
        print("\n" + "="*60)
        print("联邦学习模式已启用")
        print("请使用 run_federated_qwenvl.py 启动联邦训练")
        print("="*60)
        print("\n示例命令:")
        print("  python run_federated_qwenvl.py --num_clients 3 --num_rounds 10")
        print("\n更多选项请运行:")
        print("  python run_federated_qwenvl.py --help")
        sys.exit(0)
    
    # 原有的单机训练逻辑
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        quick_test()
    else:
        model, log = train_on_mini()
