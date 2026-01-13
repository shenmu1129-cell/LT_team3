"""
增强版Qwen3VL防御系统 - 支持LLM生成任务训练

核心改进：
1. 不仅训练分类头，还训练LLM的检测/防御生成能力
2. 使用SFT (Supervised Fine-Tuning) 方式训练生成任务
3. 构建检测和防御策略的ground truth标签
4. 联合训练：分类损失 + 生成损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from torchvision import transforms
import numpy as np
import json


# ==================== Ground Truth 模板 ====================

DETECTION_TEMPLATES = {
    'adversarial_patch': {
        'is_attack': True,
        'attack_type': 'adversarial_patch',
        'confidence': 0.9,
        'risk_level': '高',
        'suspicious_regions': ['图像中存在异常的对抗贴纸或扰动模式'],
        'analysis': '检测到对抗样本攻击。图像中存在人为添加的对抗性扰动，这些扰动通常呈现高频噪声模式，旨在欺骗深度学习模型的目标检测功能。建议立即启动防御措施。'
    },
    'sensor_spoofing': {
        'is_attack': True,
        'attack_type': 'sensor_spoofing',
        'confidence': 0.85,
        'risk_level': '高',
        'suspicious_regions': ['场景中存在伪造的交通标志或信号'],
        'analysis': '检测到传感器欺骗攻击。场景中存在伪造的交通标志、信号灯或障碍物，这些虚假目标可能导致自动驾驶系统做出错误决策。需要交叉验证多传感器数据。'
    },
    'physical_attack': {
        'is_attack': True,
        'attack_type': 'physical_attack',
        'confidence': 0.88,
        'risk_level': '紧急',
        'suspicious_regions': ['道路上存在人为放置的干扰物体'],
        'analysis': '检测到物理攻击。道路上存在人为放置的干扰物体，这些物体可能被设计用于干扰自动驾驶系统的感知或决策。建议立即减速并切换到手动驾驶模式。'
    },
    'data_poisoning': {
        'is_attack': True,
        'attack_type': 'data_poisoning',
        'confidence': 0.82,
        'risk_level': '中',
        'suspicious_regions': ['图像数据存在篡改或失真痕迹'],
        'analysis': '检测到数据完整性攻击。图像数据存在篡改、模糊或失真的痕迹，这可能是数据在传输过程中被恶意修改。建议重新获取传感器数据并进行完整性校验。'
    },
    'normal': {
        'is_attack': False,
        'attack_type': None,
        'confidence': 0.95,
        'risk_level': '低',
        'suspicious_regions': [],
        'analysis': '未检测到攻击。当前场景正常，图像和点云数据一致性良好，未发现异常模式或可疑目标。系统可正常运行。'
    }
}

DEFENSE_TEMPLATES = {
    'adversarial_patch': """针对对抗样本攻击的防御策略：

1. **立即响应措施** (0-5秒内执行)
   - 启用备用摄像头或切换视角
   - 提高对点云数据的依赖权重
   - 降低当前帧的决策置信度

2. **数据净化方案** (实时处理)
   - 应用高斯滤波去除高频噪声
   - 使用对抗训练模型进行二次验证
   - 对可疑区域进行遮罩处理

3. **系统恢复步骤** (5-30秒内)
   - 切换到保守驾驶模式
   - 增加与前车的安全距离
   - 持续监控异常区域变化

4. **预防性建议** (长期改进)
   - 更新对抗样本检测模型
   - 增加训练数据中的对抗样本
   - 部署多模型集成防御""",

    'sensor_spoofing': """针对传感器欺骗攻击的防御策略：

1. **立即响应措施** (0-5秒内执行)
   - 交叉验证摄像头和激光雷达数据
   - 对不一致的目标降低置信度
   - 启用高精度地图进行位置校验

2. **数据净化方案** (实时处理)
   - 比对历史帧中的目标一致性
   - 使用SLAM验证环境结构
   - 过滤掉物理特性异常的目标

3. **系统恢复步骤** (5-30秒内)
   - 降低车速至安全水平
   - 扩大感知范围扫描
   - 记录异常数据用于后续分析

4. **预防性建议** (长期改进)
   - 增加传感器冗余
   - 部署时序一致性检测
   - 更新欺骗目标检测模型""",

    'physical_attack': """针对物理攻击的防御策略：

1. **立即响应措施** (0-5秒内执行)
   - 紧急制动或避让
   - 切换到手动驾驶提示
   - 开启危险警示灯

2. **数据净化方案** (实时处理)
   - 标记可疑物体区域
   - 使用点云密度验证物体真实性
   - 评估物体的运动学特性

3. **系统恢复步骤** (5-30秒内)
   - 寻找安全停靠点
   - 记录攻击证据
   - 通知远程监控中心

4. **预防性建议** (长期改进)
   - 更新物理攻击识别模型
   - 增加道路异常物体检测
   - 与交通管理系统联动""",

    'data_poisoning': """针对数据完整性攻击的防御策略：

1. **立即响应措施** (0-5秒内执行)
   - 对当前帧数据进行完整性校验
   - 使用历史帧进行数据修复
   - 降低受污染数据的权重

2. **数据净化方案** (实时处理)
   - 应用数据去噪算法
   - 使用冗余传感器数据替代
   - 启用边缘检测验证

3. **系统恢复步骤** (5-30秒内)
   - 重新初始化传感器
   - 清空可能被污染的缓存
   - 恢复到安全状态

4. **预防性建议** (长期改进)
   - 加强数据传输加密
   - 部署数据完整性监控
   - 更新异常数据检测算法""",

    'normal': """当前场景安全，无需特殊防御措施。

系统建议：
- 保持正常运行模式
- 持续监控环境变化
- 定期更新检测模型"""
}


class Qwen3VLDefenseSystemEnhanced(nn.Module):
    """
    增强版Qwen3-VL防御系统
    
    支持三种训练模式：
    1. 分类训练：训练分类头判断是否有攻击
    2. 检测生成训练：训练LLM生成准确的攻击检测分析
    3. 防御生成训练：训练LLM生成有效的防御策略
    
    联合训练损失：
    L_total = α * L_cls + β * L_detect + γ * L_defend
    """
    
    def __init__(self, 
                 pointcloud_dim=1024,
                 qwen_hidden_dim=None,  # 自动检测
                 model_name=r'Qwen/Qwen3-VL-2B-Instruct',
                 train_generation=True):  # 是否训练生成任务
        super().__init__()
        
        self.train_generation = train_generation
        self.pointcloud_dim = pointcloud_dim
        
        # 1. 加载Qwen3-VL模型和processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256*32*32,
            max_pixels=1280*32*32,
        )
        
        self.qwen_vl = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # 2. 自动检测Qwen模型的隐藏层维度
        if qwen_hidden_dim is None:
            qwen_hidden_dim = self._get_hidden_dim()
        self.qwen_hidden_dim = qwen_hidden_dim
        print(f"[增强模型] Qwen隐藏层维度: {qwen_hidden_dim}, 点云特征维度: {pointcloud_dim}")
        
        # 3. 点云编码器
        self.pointcloud_encoder = PointCloudEncoderEnhanced(output_dim=pointcloud_dim)
        
        # 4. 点云特征适配器
        self.pointcloud_adapter = PointCloudAdapterEnhanced(
            pointcloud_dim=pointcloud_dim,
            qwen_hidden_dim=qwen_hidden_dim
        )
        
        # 5. 分类头 - 5分类 (normal + 4种攻击)
        self.num_classes = 5
        classifier_input_dim = qwen_hidden_dim + pointcloud_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)  # 输出5个类别的logits
        )
        
        # 6. 设置LoRA - 用于训练生成任务
        self._setup_lora_for_generation()
    
    def _get_hidden_dim(self):
        """自动获取Qwen模型的隐藏层维度"""
        try:
            config = self.qwen_vl.config
            if hasattr(config, 'hidden_size'):
                return config.hidden_size
            elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
                return config.text_config.hidden_size
            elif hasattr(config, 'd_model'):
                return config.d_model
        except Exception as e:
            print(f"[警告] 无法自动获取隐藏层维度: {e}")
        
        print("[警告] 使用默认隐藏层维度: 1536")
        return 1536
        
    def _setup_lora_for_generation(self):
        """设置LoRA微调 - 同时用于分类和生成任务"""
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=16,  # 增加rank以提升生成能力
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj"  # FFN - 对生成很重要
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.qwen_vl = get_peft_model(self.qwen_vl, lora_config)
        print("\n=== LoRA 配置 (增强版 - 支持生成训练) ===")
        self.qwen_vl.print_trainable_parameters()
    
    def forward(self, images, pointclouds, mode='train', attack_type=None, 
                labels=None, attack_types_gt=None):
        """
        前向传播
        
        Args:
            images: PIL Image list
            pointclouds: [B, N, 3]
            mode: 'train', 'detect', 'defend', 'train_generation'
            attack_type: 攻击类型（用于defend模式）
            labels: 分类标签 [B]（用于train_generation模式）
            attack_types_gt: Ground truth攻击类型列表（用于train_generation模式）
        """
        batch_size = pointclouds.size(0) if torch.is_tensor(pointclouds) else len(pointclouds)
        
        # 处理点云特征
        if torch.is_tensor(pointclouds):
            pc_features = self.pointcloud_encoder(pointclouds)
            pc_embeddings = self.pointcloud_adapter(pc_features)
        
        if mode == 'train':
            return self._training_forward(images, pc_features, batch_size)
        elif mode == 'detect':
            return self._detection_mode(images, pc_embeddings, batch_size)
        elif mode == 'defend':
            return self._defense_mode(images, pc_embeddings, attack_type, batch_size)
        elif mode == 'train_generation':
            # 新增：训练LLM生成任务
            return self._train_generation_forward(
                images, pc_features, pc_embeddings, batch_size, 
                labels, attack_types_gt
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _training_forward(self, images, pc_features, batch_size):
        """分类训练的前向传播"""
        logits_list = []
        
        for i in range(batch_size):
            if isinstance(images, list):
                image = images[i]
            else:
                image = transforms.ToPILImage()(images[i].cpu())
            
            vision_feature = self._extract_vision_features(image)
            pc_feature = pc_features[i]
            
            # 确保设备和类型一致
            if pc_feature.device != vision_feature.device:
                pc_feature = pc_feature.to(vision_feature.device)
            vision_feature = vision_feature.float()
            pc_feature = pc_feature.float()
            
            combined_feature = torch.cat([vision_feature, pc_feature], dim=0)
            
            # 首次运行时检查维度
            if i == 0 and not hasattr(self, '_dim_checked'):
                expected_dim = self.qwen_hidden_dim + self.pointcloud_dim
                actual_dim = combined_feature.shape[0]
                print(f"[增强模型] vision_feature: {vision_feature.shape}, pc_feature: {pc_feature.shape}")
                print(f"[增强模型] combined_feature: {combined_feature.shape}, 期望: {expected_dim}")
                
                if actual_dim != expected_dim:
                    print(f"[警告] 维度不匹配! 正在动态调整分类器...")
                    self.classifier = nn.Sequential(
                        nn.Linear(actual_dim, 512),
                        nn.LayerNorm(512),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(512, 256),
                        nn.LayerNorm(256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, self.num_classes)  # 5分类输出
                    ).to(combined_feature.device)
                    print(f"[增强模型] 分类器已调整为输入维度: {actual_dim}, 输出类别数: {self.num_classes}")
                
                self._dim_checked = True
            
            logit = self.classifier(combined_feature)
            logits_list.append(logit)
        
        logits = torch.stack(logits_list, dim=0)
        return logits
    
    def _train_generation_forward(self, images, pc_features, pc_embeddings, 
                                   batch_size, labels, attack_types_gt):
        """
        训练LLM生成任务的前向传播
        
        返回：
        {
            'cls_logits': 分类logits,
            'detect_loss': 检测生成损失,
            'defend_loss': 防御生成损失,
            'total_loss': 总损失
        }
        """
        # 1. 分类logits
        cls_logits = self._training_forward(images, pc_features, batch_size)
        
        # 2. 检测生成损失
        detect_loss = self._compute_detection_loss(
            images, pc_embeddings, batch_size, labels, attack_types_gt
        )
        
        # 3. 防御生成损失（只对攻击样本计算）
        defend_loss = self._compute_defense_loss(
            images, pc_embeddings, batch_size, labels, attack_types_gt
        )
        
        return {
            'cls_logits': cls_logits,
            'detect_loss': detect_loss,
            'defend_loss': defend_loss
        }
    
    def _compute_detection_loss(self, images, pc_embeddings, batch_size, 
                                 labels, attack_types_gt):
        """计算检测生成的语言模型损失"""
        total_loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            if isinstance(images, list):
                image = images[i]
            else:
                image = transforms.ToPILImage()(images[i].cpu())
            
            # 获取ground truth
            if labels[i].item() == 1:  # 攻击样本
                attack_type = attack_types_gt[i] if attack_types_gt else 'adversarial_patch'
                gt_response = json.dumps(DETECTION_TEMPLATES.get(attack_type, 
                                         DETECTION_TEMPLATES['adversarial_patch']), 
                                         ensure_ascii=False)
            else:  # 正常样本
                gt_response = json.dumps(DETECTION_TEMPLATES['normal'], ensure_ascii=False)
            
            # 构建输入prompt
            query_text = self._get_detection_prompt()
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query_text},
                ],
            }]
            
            # 计算生成损失
            loss = self._compute_lm_loss(messages, gt_response, image)
            if loss is not None:
                total_loss += loss
                valid_samples += 1
        
        return total_loss / max(valid_samples, 1)
    
    def _compute_defense_loss(self, images, pc_embeddings, batch_size,
                              labels, attack_types_gt):
        """计算防御策略生成的语言模型损失"""
        total_loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            # 只对攻击样本计算防御损失
            if labels[i].item() != 1:
                continue
                
            if isinstance(images, list):
                image = images[i]
            else:
                image = transforms.ToPILImage()(images[i].cpu())
            
            # 获取ground truth防御策略
            attack_type = attack_types_gt[i] if attack_types_gt else 'adversarial_patch'
            gt_response = DEFENSE_TEMPLATES.get(attack_type, DEFENSE_TEMPLATES['adversarial_patch'])
            
            # 构建输入prompt
            query_text = self._get_defense_prompt(attack_type)
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query_text},
                ],
            }]
            
            # 计算生成损失
            loss = self._compute_lm_loss(messages, gt_response, image)
            if loss is not None:
                total_loss += loss
                valid_samples += 1
        
        return total_loss / max(valid_samples, 1)
    
    def _compute_lm_loss(self, messages, target_text, image):
        """计算语言模型的交叉熵损失"""
        try:
            # 准备输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 添加目标文本
            full_text = text + target_text
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[full_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.qwen_vl.device) for k, v in inputs.items()}
            
            # 创建labels（只在目标文本部分计算损失）
            labels = inputs['input_ids'].clone()
            
            # 找到prompt结束的位置，只在目标文本部分计算损失
            prompt_tokens = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )['input_ids']
            prompt_len = prompt_tokens.shape[1]
            
            # 将prompt部分的label设为-100（不计算损失）
            labels[:, :prompt_len] = -100
            
            # 前向传播计算损失
            outputs = self.qwen_vl(
                **inputs,
                labels=labels
            )
            
            return outputs.loss
            
        except Exception as e:
            print(f"计算LM损失时出错: {e}")
            return None
    
    def _get_detection_prompt(self):
        """获取检测任务的prompt"""
        return """请仔细分析这张自动驾驶场景图像,检测是否存在对抗攻击或异常。

可能的攻击类型包括:
1. 对抗样本攻击 (Adversarial Patch)
2. 传感器欺骗攻击 (Sensor Spoofing)
3. 物理攻击 (Physical Attack)
4. 数据完整性攻击 (Data Poisoning)

请按JSON格式回答:
{
    "is_attack": true/false,
    "attack_type": "攻击类型" 或 null,
    "confidence": 0.0-1.0,
    "risk_level": "低/中/高/紧急",
    "suspicious_regions": ["区域描述"],
    "analysis": "详细分析"
}"""
    
    def _get_defense_prompt(self, attack_type):
        """获取防御任务的prompt"""
        return f"""检测到自动驾驶系统受到攻击!

攻击类型: {attack_type}

请提供详细的应对方案，包括：
1. 立即响应措施 (0-5秒内)
2. 数据净化方案 (实时处理)
3. 系统恢复步骤 (5-30秒内)
4. 预防性建议 (长期改进)"""

    def _extract_vision_features(self, image):
        """提取视觉特征"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Analyze this image."},
            ],
        }]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.qwen_vl.device) for k, v in inputs.items()}
        
        with torch.set_grad_enabled(self.training):
            outputs = self.qwen_vl.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            vision_feature = hidden_states.mean(dim=1).squeeze(0)
        
        return vision_feature
    
    def _detection_mode(self, images, pc_embeddings, batch_size):
        """检测模式 - 生成攻击检测结果"""
        results = []
        
        for i in range(batch_size):
            if isinstance(images, list):
                image = images[i]
            else:
                image = transforms.ToPILImage()(images[i].cpu())
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._get_detection_prompt()},
                ],
            }]
            
            response = self._generate_response(messages, image)
            parsed_result = self._parse_detection_result(response)
            results.append(parsed_result)
        
        return results
    
    def _defense_mode(self, images, pc_embeddings, attack_type, batch_size):
        """防御模式 - 生成防御策略"""
        results = []
        
        for i in range(batch_size):
            if isinstance(images, list):
                image = images[i]
            else:
                image = transforms.ToPILImage()(images[i].cpu())
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._get_defense_prompt(attack_type)},
                ],
            }]
            
            response = self._generate_response(messages, image)
            results.append(response)
        
        return results
    
    def _generate_response(self, messages, image):
        """生成响应"""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.qwen_vl.device)
        
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
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx+1]
                return json.loads(json_str)
        except:
            pass
        
        return {
            'is_attack': False,
            'attack_type': None,
            'confidence': 0.5,
            'risk_level': '低',
            'suspicious_regions': [],
            'analysis': response_text[:200] if response_text else "无法解析响应"
        }


class PointCloudEncoderEnhanced(nn.Module):
    """增强版点云编码器"""
    
    def __init__(self, output_dim=1024, num_points=2048):
        super().__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, dim=2)[0]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PointCloudAdapterEnhanced(nn.Module):
    """增强版点云适配器"""
    
    def __init__(self, pointcloud_dim=1024, qwen_hidden_dim=3072, num_tokens=64):
        super().__init__()
        
        self.num_tokens = num_tokens
        
        self.projector = nn.Sequential(
            nn.Linear(pointcloud_dim, qwen_hidden_dim * 2),
            nn.LayerNorm(qwen_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(qwen_hidden_dim * 2, qwen_hidden_dim * num_tokens),
        )
        
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_tokens, qwen_hidden_dim) * 0.02
        )
        
    def forward(self, pc_features):
        B = pc_features.size(0)
        x = self.projector(pc_features)
        x = x.view(B, self.num_tokens, -1)
        x = x + self.pos_embedding
        return x


class EnhancedTrainer:
    """
    增强版训练器 - 支持联合训练分类和生成任务
    
    损失函数：
    L_total = α * L_cls + β * L_detect + γ * L_defend
    """
    
    def __init__(self, model, train_loader, val_loader, device='cuda',
                 lr=1e-4, weight_decay=0.01,
                 alpha=1.0,  # 分类损失权重
                 beta=0.5,   # 检测生成损失权重
                 gamma=0.5): # 防御生成损失权重
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 只优化可训练的参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"\n可训练参数数量: {sum(p.numel() for p in trainable_params):,}")
        
        self.optimizer = torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
        
        self.cls_criterion = nn.BCEWithLogitsLoss()
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 20,
            eta_min=1e-6
        )
    
    def train_epoch(self, epoch, train_generation=True):
        """
        训练一个epoch
        
        Args:
            train_generation: 是否训练生成任务
        """
        self.model.train()
        
        total_cls_loss = 0
        total_detect_loss = 0
        total_defend_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['images']
            pointclouds = batch['pointclouds'].to(self.device)
            labels = batch['labels'].to(self.device)
            attack_types = batch.get('attack_types', None)
            
            self.optimizer.zero_grad()
            
            if train_generation:
                # 联合训练模式
                outputs = self.model(
                    images, pointclouds, 
                    mode='train_generation',
                    labels=labels,
                    attack_types_gt=attack_types
                )
                
                cls_logits = outputs['cls_logits']
                detect_loss = outputs['detect_loss']
                defend_loss = outputs['defend_loss']
                
                # 计算分类损失
                cls_loss = self.cls_criterion(cls_logits.squeeze(-1), labels.float())
                
                # 总损失
                total_loss = (self.alpha * cls_loss + 
                             self.beta * detect_loss + 
                             self.gamma * defend_loss)
                
                total_detect_loss += detect_loss.item() if torch.is_tensor(detect_loss) else detect_loss
                total_defend_loss += defend_loss.item() if torch.is_tensor(defend_loss) else defend_loss
            else:
                # 仅分类训练
                cls_logits = self.model(images, pointclouds, mode='train')
                cls_loss = self.cls_criterion(cls_logits.squeeze(-1), labels.float())
                total_loss = cls_loss
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            total_cls_loss += cls_loss.item()
            predictions = torch.sigmoid(cls_logits.squeeze(-1))
            predicted = (predictions > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Cls Loss: {cls_loss.item():.4f} "
                      f"Detect Loss: {total_detect_loss/(batch_idx+1):.4f} "
                      f"Defend Loss: {total_defend_loss/(batch_idx+1):.4f} "
                      f"Acc: {100.*correct/total:.2f}%")
        
        num_batches = len(self.train_loader)
        return {
            'cls_loss': total_cls_loss / num_batches,
            'detect_loss': total_detect_loss / num_batches,
            'defend_loss': total_defend_loss / num_batches,
            'accuracy': correct / total
        }


# ==================== 导出 ====================
__all__ = [
    'Qwen3VLDefenseSystemEnhanced',
    'EnhancedTrainer',
    'DETECTION_TEMPLATES',
    'DEFENSE_TEMPLATES'
]
