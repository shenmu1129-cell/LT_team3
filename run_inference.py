#!/usr/bin/env python3
"""
LLM攻击检测与防御策略推理脚本

这个脚本用于演示和测试Qwen3-VL模型的攻击检测和防御策略生成能力。

功能：
1. 加载训练好的模型
2. 对输入的图像+点云进行攻击检测
3. 生成详细的防御策略建议

用法：
    python run_inference.py --checkpoint path/to/model.pth --num_samples 5
"""

import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader
from datetime import datetime

# 导入模型和数据集
from test_local_train_mini_qwen3vl_fixed import (
    Qwen3VLDefenseSystem,
    NuScenesMiniDataset,
    custom_collate_fn
)


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")
    
    # 创建模型
    model = Qwen3VLDefenseSystem(
        pointcloud_dim=1024,
        qwen_hidden_dim=3072,
        model_name="/home/sutongtong/wwt/model/Qwen3-VL-2B-Instruct"
    )
    
    # 加载权重
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ 模型权重加载成功")
    else:
        print("⚠ 未找到checkpoint，使用预训练权重")
    
    model = model.to(device)
    model.eval()
    
    return model


def run_attack_detection(model, batch, device):
    """
    运行攻击检测
    
    Returns:
        List[Dict]: 检测结果，每个元素包含:
            - is_attack: 是否检测到攻击
            - attack_type: 攻击类型
            - confidence: 置信度
            - risk_level: 风险等级
            - analysis: 详细分析
    """
    images = batch['images']
    pointclouds = batch['pointclouds'].to(device)
    
    with torch.no_grad():
        # 使用detect模式
        detection_results = model(images, pointclouds, mode='detect')
    
    return detection_results


def run_defense_generation(model, batch, attack_type, device):
    """
    运行防御策略生成
    
    Returns:
        List[str]: 防御策略文本列表
    """
    images = batch['images']
    pointclouds = batch['pointclouds'].to(device)
    
    with torch.no_grad():
        # 使用defend模式
        defense_strategies = model(images, pointclouds, mode='defend', attack_type=attack_type)
    
    return defense_strategies


def run_classification(model, batch, device):
    """
    运行分类预测
    
    Returns:
        Dict: 包含logits, predictions, confidence
    """
    images = batch['images']
    pointclouds = batch['pointclouds'].to(device)
    
    with torch.no_grad():
        logits = model(images, pointclouds, mode='train')
        predictions = torch.sigmoid(logits.squeeze(-1))
        predicted_labels = (predictions > 0.5).long()
    
    return {
        'logits': logits.cpu().numpy().tolist(),
        'predictions': predicted_labels.cpu().numpy().tolist(),
        'confidence': predictions.cpu().numpy().tolist()
    }


def full_inference(model, batch, device):
    """
    完整推理流程
    
    Returns:
        Dict: 包含分类结果、检测结果、防御策略
    """
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'samples': []
    }
    
    # 1. 分类预测
    classification = run_classification(model, batch, device)
    
    # 2. 攻击检测
    detection_results = run_attack_detection(model, batch, device)
    
    # 3. 对每个样本生成完整结果
    batch_size = len(batch['images']) if isinstance(batch['images'], list) else batch['images'].size(0)
    
    for i in range(batch_size):
        sample_result = {
            'sample_token': batch.get('sample_tokens', ['unknown'])[i] if batch.get('sample_tokens') else 'unknown',
            'ground_truth': batch['labels'][i].item() if 'labels' in batch else None,
            'classification': {
                'prediction': classification['predictions'][i],
                'confidence': classification['confidence'][i],
                'label': '攻击' if classification['predictions'][i] == 1 else '正常'
            },
            'detection': detection_results[i] if i < len(detection_results) else {}
        }
        
        # 4. 如果检测到攻击，生成防御策略
        if sample_result['detection'].get('is_attack', False):
            attack_type = sample_result['detection'].get('attack_type', 'unknown')
            
            # 构造单样本batch
            single_batch = {
                'images': [batch['images'][i]] if isinstance(batch['images'], list) else batch['images'][i:i+1],
                'pointclouds': batch['pointclouds'][i:i+1]
            }
            
            defense = run_defense_generation(model, single_batch, attack_type, device)
            sample_result['defense_strategy'] = defense[0] if defense else "无法生成防御策略"
        else:
            sample_result['defense_strategy'] = "无需防御 - 未检测到攻击"
        
        results['samples'].append(sample_result)
    
    return results


def print_results(results):
    """打印推理结果"""
    print("\n" + "="*80)
    print("                    攻击检测与防御策略推理结果")
    print("="*80)
    print(f"推理时间: {results['timestamp']}")
    print(f"样本数量: {len(results['samples'])}")
    
    for i, sample in enumerate(results['samples']):
        print(f"\n{'─'*80}")
        print(f"样本 {i+1}: {sample['sample_token']}")
        print(f"{'─'*80}")
        
        # 真实标签
        if sample['ground_truth'] is not None:
            gt_label = '攻击' if sample['ground_truth'] == 1 else '正常'
            print(f"  真实标签: {gt_label}")
        
        # 分类结果
        print(f"\n  [分类器预测]")
        print(f"    预测: {sample['classification']['label']}")
        print(f"    置信度: {sample['classification']['confidence']:.4f}")
        
        # LLM检测结果
        print(f"\n  [LLM攻击检测]")
        det = sample['detection']
        if det:
            print(f"    是否攻击: {'是' if det.get('is_attack') else '否'}")
            print(f"    攻击类型: {det.get('attack_type', 'N/A')}")
            print(f"    风险等级: {det.get('risk_level', 'N/A')}")
            print(f"    LLM置信度: {det.get('confidence', 'N/A')}")
            
            # 分析内容
            analysis = det.get('analysis', '')
            if analysis:
                print(f"    分析:")
                # 分行显示，每行80字符
                for line in [analysis[i:i+70] for i in range(0, len(analysis), 70)]:
                    print(f"      {line}")
            
            # 可疑区域
            suspicious = det.get('suspicious_regions', [])
            if suspicious:
                print(f"    可疑区域:")
                for region in suspicious:
                    print(f"      - {region}")
        
        # 防御策略
        print(f"\n  [防御策略]")
        defense = sample['defense_strategy']
        if defense and defense != "无需防御 - 未检测到攻击":
            # 分行显示
            for line in [defense[i:i+70] for i in range(0, len(defense), 70)]:
                print(f"    {line}")
        else:
            print(f"    {defense}")
    
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='LLM攻击检测与防御策略推理')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径')
    parser.add_argument('--dataroot', type=str,
                        default='/home/sutongtong/LanTu_team3/dataset/nuScenes/train',
                        help='nuScenes数据集路径')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='推理样本数量')
    parser.add_argument('--attack_ratio', type=float, default=0.5,
                        help='攻击样本比例（用于测试）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--save_results', type=str, default=None,
                        help='保存结果的JSON文件路径')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'detect', 'defend', 'classify'],
                        help='推理模式')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    if args.checkpoint:
        model = load_model(args.checkpoint, device)
    else:
        print("未指定checkpoint，使用预训练模型")
        model = Qwen3VLDefenseSystem(
            pointcloud_dim=1024,
            qwen_hidden_dim=3072,
            model_name="/home/sutongtong/wwt/model/Qwen3-VL-2B-Instruct"
        ).to(device)
        model.eval()
    
    # 加载数据
    print("\n加载测试数据...")
    dataset = NuScenesMiniDataset(
        dataroot=args.dataroot,
        version='v1.0-trainval',
        split='val',
        attack_ratio=args.attack_ratio,
        num_points=2048,
        use_cache=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.num_samples,
        shuffle=True,
        num_workers=2,
        collate_fn=custom_collate_fn
    )
    
    # 获取一个batch进行推理
    batch = next(iter(dataloader))
    print(f"加载了 {len(batch['images'])} 个样本")
    
    # 运行推理
    print("\n开始推理...")
    
    if args.mode == 'full':
        results = full_inference(model, batch, device)
        print_results(results)
    elif args.mode == 'detect':
        detection_results = run_attack_detection(model, batch, device)
        print("\n攻击检测结果:")
        for i, det in enumerate(detection_results):
            print(f"\n样本 {i+1}:")
            print(json.dumps(det, ensure_ascii=False, indent=2))
    elif args.mode == 'defend':
        # 先检测，再对检测到攻击的生成防御策略
        detection_results = run_attack_detection(model, batch, device)
        print("\n防御策略生成结果:")
        for i, det in enumerate(detection_results):
            if det.get('is_attack', False):
                attack_type = det.get('attack_type', 'unknown')
                single_batch = {
                    'images': [batch['images'][i]],
                    'pointclouds': batch['pointclouds'][i:i+1]
                }
                defense = run_defense_generation(model, single_batch, attack_type, device)
                print(f"\n样本 {i+1} - 攻击类型: {attack_type}")
                print(f"防御策略:\n{defense[0]}")
    elif args.mode == 'classify':
        classification = run_classification(model, batch, device)
        print("\n分类结果:")
        for i in range(len(classification['predictions'])):
            label = '攻击' if classification['predictions'][i] == 1 else '正常'
            conf = classification['confidence'][i]
            gt = '攻击' if batch['labels'][i].item() == 1 else '正常'
            print(f"样本 {i+1}: 预测={label} (置信度:{conf:.4f}), 真实={gt}")
    
    # 保存结果
    if args.save_results and args.mode == 'full':
        # 将tensor转为可序列化的格式
        save_results = results.copy()
        for sample in save_results['samples']:
            if 'classification' in sample:
                sample['classification']['confidence'] = float(sample['classification']['confidence'])
        
        with open(args.save_results, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.save_results}")


if __name__ == "__main__":
    main()
