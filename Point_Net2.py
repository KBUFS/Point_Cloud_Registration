"""PointNet++ 特征提取与分类，这里不涉及配准步骤，仅对利用PointNet++进行物体识别的功能进行展示"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
print("✅ 已设置中文字体支持")

# ==================== 1. 工具函数  ====================
def square_distance(src, dst):
    """计算成对欧几里得距离"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """根据索引从点云中采样点"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """最远点采样 (Farthest Point Sampling)"""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """球查询 (Ball Query)"""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    idx[sqrdists > radius ** 2] = N
    idx = idx.sort(dim=-1)[0][:, :, :nsample]
    
    group_idx = idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    idx[idx == N] = group_idx[idx == N]
    
    return idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """采样和分组"""
    B, N, C = xyz.shape
    S = npoint
    
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
    
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    
    if points is not None:
        grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """全局采样（用于最后一层）"""
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    """PointNet++ 的 Set Abstraction 层 - 完全匹配仓库"""
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz, points):
        """
        xyz: (B, C, N)
        points: (B, D, N)
        """
        xyz = xyz.permute(0, 2, 1).contiguous()
        if points is not None:
            points = points.permute(0, 2, 1).contiguous()
        
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        
        # new_points: (B, npoint, nsample, C+D)
        new_points = new_points.permute(0, 3, 2, 1).contiguous()  # (B, C+D, nsample, npoint)
        
        # MLP - 完全按照仓库的 F.relu(bn(conv(x))) 顺序
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Max Pooling
        new_points = torch.max(new_points, 2)[0]  # (B, D', npoint)
        
        new_xyz = new_xyz.permute(0, 2, 1).contiguous()
        return new_xyz, new_points

# ==================== 2. PointNet++ 模型 (完全匹配仓库) ====================
class PointNet2Cls(nn.Module):
    """PointNet++模型"""
    def __init__(self, num_class=40, normal_channel=True):
        super(PointNet2Cls, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, 
                                         in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, 
                                         in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, 
                                         in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)
    
    def forward(self, xyz, return_features=False):
        """
        前向传播
        输入: (B, 3, N) 或 (B, 6, N) 如果包含法线
        输出: (log_softmax输出, 全局特征)
        """
        B, _, _ = xyz.shape
        
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        
        # 三层特征提取
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # 全局特征
        x = l3_points.view(B, 1024)
        
        # 分类头 - 完全按照仓库的顺序
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        
        if return_features:
            return x, l3_points, l1_points, l2_points
        return x, l3_points
    
    def extract_descriptor(self, xyz):
        """专门用于提取1024维全局描述子"""
        B, _, _ = xyz.shape
        
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        return l3_points.view(B, 1024)  # 全局描述子
    
    def extract_all_features(self, xyz):
        """提取所有层级特征"""
        B, _, _ = xyz.shape
        
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        return {
            'global_feat': l3_points.view(B, 1024),  # 全局特征
            'l1_points': l1_points,  # 第一层特征
            'l2_points': l2_points,  # 第二层特征
            'l3_points': l3_points   # 第三层特征
        }

# ==================== 3. 数据处理函数 ====================
def load_stanford_bunny(num_points=1024, normalize=True):
    """加载斯坦福兔子点云"""
    print("🐇 加载 Stanford Bunny 点云...")
    
    try:
        # 方法1: 从 data 目录加载
        ply_files = list(Path("./data").glob("*.ply"))
        if ply_files:
            filepath = str(ply_files[0])
            pcd = o3d.io.read_point_cloud(filepath)
            print(f"  从文件加载: {Path(filepath).name}")
        else:
            # 方法2: 使用Open3D内置数据
            bunny = o3d.data.BunnyMesh()
            mesh = o3d.io.read_triangle_mesh(bunny.path)
            pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
            print("  使用Open3D内置兔子数据")
    except Exception as e:
        # 方法3: 创建测试点云
        print(f"  加载失败: {e}, 创建测试点云")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    
    # 下采样到指定点数
    if len(pcd.points) > num_points:
        pcd = pcd.farthest_point_down_sample(num_points)
    
    points = np.asarray(pcd.points, dtype=np.float32)
    
    # 归一化
    if normalize:
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
    
    print(f"  点数: {points.shape[0]}, 范围: [{points.min():.3f}, {points.max():.3f}]")
    
    return points

def prepare_pointcloud_3channel(points, num_points=1024, device='cpu'):
    """准备3通道输入（仅XYZ）"""
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        repeat_times = num_points // len(points) + 1
        points = np.tile(points, (repeat_times, 1))[:num_points]
    
    # 确保只有3个通道
    if points.shape[1] > 3:
        points = points[:, :3]
        print(f"  🔧 截取前3个通道: XYZ")
    
    points_tensor = torch.from_numpy(points.T).float().unsqueeze(0)
    return points_tensor.to(device)

def prepare_pointcloud_6channel(points, num_points=1024, device='cpu'):
    """准备6通道输入（XYZ+法线）仓库源代码默认6通道，但是兔子点云数据中没有法向量故只能使用3通道"""
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        repeat_times = num_points // len(points) + 1
        points = np.tile(points, (repeat_times, 1))[:num_points]
    
    # 如果只有3通道，添加零法线
    if points.shape[1] == 3:
        zeros = np.zeros((len(points), 3), dtype=np.float32)
        points = np.concatenate([points, zeros], axis=1)
        print(f"  🔧 添加零法线扩展为6通道")
    elif points.shape[1] > 6:
        points = points[:, :6]
        print(f"  🔧 截取前6个通道")
    
    points_tensor = torch.from_numpy(points.T).float().unsqueeze(0)
    return points_tensor.to(device)

# ==================== 4. 模型加载和推理 ====================
def load_pointnet2_model(model_path=None, num_classes=40, device='cpu'):
    """加载PointNet++模型 - 完全匹配仓库"""
    print("🔧 初始化 PointNet++ 模型 (仓库版本)...")
    
    # 创建模型 - 仓库默认使用normal_channel=True
    model = PointNet2Cls(num_class=num_classes, normal_channel=False)
    
    # 如果没有权重文件，直接返回随机模型
    if not model_path or not Path(model_path).exists():
        print("  ⚠️  未找到预训练权重，使用随机初始化")
        model.to(device)
        model.eval()
        return model
    
    print(f"  加载预训练权重: {model_path}")
    
    try:
        # 加载权重
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 诊断checkpoint结构
        print(f"  🔍 Checkpoint类型: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"  🔍 Checkpoint键名: {list(checkpoint.keys())[:10]}...")
        
        # 提取state_dict
        state_dict = None
        if isinstance(checkpoint, dict):
            # 仓库的权重文件通常直接是state_dict
            if all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
                state_dict = checkpoint
            # 或者有model_state_dict键
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'net' in checkpoint:
                state_dict = checkpoint['net']
        
        if state_dict is None:
            print("  ⚠️  无法识别权重格式，使用随机初始化")
        else:
            # 检查是否有'module.'前缀
            first_key = list(state_dict.keys())[0]
            if first_key.startswith('module.'):
                print("  🔧 移除'module.'前缀")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            # 打印前几个键检查结构
            print(f"  🔍 前5个权重键: {list(state_dict.keys())[:5]}")
            
            # 加载权重
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            # 检查特征提取层是否加载成功
            sa_keys = [k for k in state_dict.keys() if 'sa' in k]
            if len(sa_keys) > 0:
                print(f"  ✅ 权重加载成功 (找到 {len(sa_keys)} 个特征提取层)")
            else:
                print("  ⚠️  未找到特征提取层，可能结构不匹配")
            
            if missing_keys:
                print(f"  ⚠️  缺失的键 ({len(missing_keys)}个): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"  ⚠️  意外的键 ({len(unexpected_keys)}个): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                
    except Exception as e:
        print(f"  ❌ 权重加载失败: {e}")
        print("  使用随机初始化权重")
    
    model.to(device)
    model.eval()
    return model

def extract_features_with_pointnet2(model, points_tensor, device='cpu'):
    """使用PointNet++提取特征"""
    print("🔍 提取特征...")
    
    with torch.no_grad():
        # 提取所有特征
        features = model.extract_all_features(points_tensor)
    
    print(f"  📐 全局特征维度: {features['global_feat'].shape}")
    print(f"  📐 第一层特征维度: {features['l1_points'].shape}")
    print(f"  📐 第二层特征维度: {features['l2_points'].shape}")
    print(f"  📐 第三层特征维度: {features['l3_points'].shape}")
    
    return features

def inference_with_pointnet2(model, points_tensor, device='cpu'):
    """使用PointNet++进行分类推理"""
    print("🎯 运行分类推理...")
    
    with torch.no_grad():
        # 前向传播
        log_softmax_output, global_feat, l1_feat, l2_feat = model(
            points_tensor, return_features=True
        )
        
        # 计算概率 (从log_softmax转回概率)
        probs = torch.exp(log_softmax_output)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    print(f"  📊 预测类别: {pred_class}, 置信度: {confidence:.4f}")
    print(f"  📐 全局特征维度: {global_feat.shape}")
    
    return {
        'log_softmax': log_softmax_output,
        'probs': probs,
        'pred_class': pred_class,
        'confidence': confidence,
        'global_feat': global_feat,
        'l1_feat': l1_feat,
        'l2_feat': l2_feat
    }

# ==================== 5. 可视化函数 ====================
def visualize_features(points, features, save_path='pointnet2_features.png'):
    """可视化PointNet++提取的特征"""
    print("📈 生成特征可视化...")
    
    # 提取特征
    global_feat = features['global_feat'].cpu().numpy().flatten()
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 使用点云的x坐标作为颜色
    colors = np.zeros((len(points), 3))
    x_norm = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min())
    colors[:, 0] = x_norm  # 红色通道
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 特征可视化
    fig = plt.figure(figsize=(12, 5))
    
    # 子图1: 全局特征
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(range(min(50, len(global_feat))), global_feat[:50], color='steelblue')
    ax1.set_title("全局特征向量 (前50维)", fontsize=12)
    ax1.set_xlabel("特征维度", fontsize=10)
    ax1.set_ylabel("特征值", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 特征分布
    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(global_feat, bins=50, alpha=0.7, color='coral')
    ax2.set_title("特征值分布", fontsize=12)
    ax2.set_xlabel("特征值", fontsize=10)
    ax2.set_ylabel("频数", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("PointNet++ 特征提取结果", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"  📸 特征图已保存为 '{save_path}'")
    
    # 显示点云
    print("\n👀 显示点云 (按ESC或Q退出)...")
    o3d.visualization.draw_geometries(
        [pcd], 
        window_name="Stanford Bunny - PointNet++ 特征",
        width=800, height=600
    )

def visualize_classification_results(results, save_path='pointnet2_classification.png'):
    """可视化分类结果"""
    print("📊 生成分类结果可视化...")
    
    probs = results['probs'].cpu().numpy().flatten()
    pred_class = results['pred_class']
    confidence = results['confidence']
    
    # 创建分类结果图
    plt.figure(figsize=(8, 6))
    
    # 只显示前20个类别（如果有40类的话）
    num_to_show = min(20, len(probs))
    classes = [f'Class {i}' for i in range(num_to_show)]
    probs_to_show = probs[:num_to_show]
    
    colors = ['red' if i == pred_class else 'steelblue' for i in range(num_to_show)]
    bars = plt.bar(classes, probs_to_show, color=colors, alpha=0.8)
    
    plt.axhline(y=confidence, color='green', linestyle='--', alpha=0.7, 
                label=f'置信度: {confidence:.3f}')
    
    plt.xlabel("类别", fontsize=12)
    plt.ylabel("概率", fontsize=12)
    plt.title(f"PointNet++ 分类结果 - 预测: Class {pred_class}", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1.1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 在柱状图上添加数值
    for bar, prob in zip(bars, probs_to_show):
        height = bar.get_height()
        if height > 0.05:  # 只显示较大的概率值
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"  📸 分类结果图已保存为 '{save_path}'")

# ==================== 6. 主函数 ====================
def main():
    """主函数"""
    print("="*60)
    print("PointNet++ 点云特征提取与分类")
    print("="*60)
    
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ 使用设备: {device}")
    
    # 模型路径
    MODEL_PATH = "./best_model.pth"  # 修改为你的权重文件路径
    
    # 参数
    NUM_POINTS = 1024
    NUM_CLASSES = 40  # ModelNet40是40类
    
    # 1. 加载模型
    model = load_pointnet2_model(MODEL_PATH, NUM_CLASSES, device)
    
    # 2. 加载点云
    points = load_stanford_bunny(NUM_POINTS)
    
    # 3. 准备数据
    if model.normal_channel:
        points_tensor = prepare_pointcloud_6channel(points, NUM_POINTS, device)
    else:
        points_tensor = prepare_pointcloud_3channel(points, NUM_POINTS, device)
    
    print(f"\n📦 输入数据形状: {points_tensor.shape}")
    print(f"  模型期望输入: {'6通道' if model.normal_channel else '3通道'}")
    
    # 4. 提取特征（主要功能）
    print("\n" + "="*40)
    print("特征提取")
    print("="*40)
    features = extract_features_with_pointnet2(model, points_tensor, device)
    
    # 5. 可视化特征
    visualize_features(points, features, 'pointnet2_features.png')
    
    # 6. 分类推理（次要功能）
    print("\n" + "="*40)
    print("分类推理")
    print("="*40)
    results = inference_with_pointnet2(model, points_tensor, device)
    
    # 7. 可视化分类结果
    visualize_classification_results(results, 'pointnet2_classification.png')
    
    # 8. 保存特征
    print("\n💾 保存特征到文件...")
    save_features_to_file(features, 'pointnet2_features.npz')
    
    print("\n" + "="*60)
    print("✅ PointNet++ 处理完成!")
    print("="*60)
    print(f"🔑 关键输出:")
    print(f"  - 全局描述子: {features['global_feat'].shape} (1024维)")
    print(f"  - 预测类别: Class {results['pred_class']}")
    print(f"  - 置信度: {results['confidence']:.4f}")
    print(f"  - 特征图: pointnet2_features.png")
    print(f"  - 分类图: pointnet2_classification.png")
    print(f"  - 特征文件: pointnet2_features.npz")

def save_features_to_file(features, filename):
    """保存特征到文件"""
    np.savez_compressed(
        filename,
        global_feat=features['global_feat'].cpu().numpy(),
        l1_feat=features['l1_points'].cpu().numpy(),
        l2_feat=features['l2_points'].cpu().numpy(),
        l3_feat=features['l3_points'].cpu().numpy()
    )
    print(f"  ✅ 特征已保存到: {filename}")

if __name__ == "__main__":
    main()