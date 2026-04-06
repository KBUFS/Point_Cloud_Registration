""""这里使用的预训练权重是用于分类，在描述子的基础上增加点全连接层，将N维向量映射到K个类别上"""

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
    """PointNet++ 的 Set Abstraction 层"""
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
        
        # MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Max Pooling
        new_points = torch.max(new_points, 2)[0]  # (B, D', npoint)
        
        new_xyz = new_xyz.permute(0, 2, 1).contiguous()
        return new_xyz, new_points

# ==================== 2. PointNet++ 模型 ====================
class PointNet2Cls(nn.Module):
    """PointNet++ 分类模型 (SSG 单尺度分组)"""
    def __init__(self, num_class=2, normal_channel=False):
        super(PointNet2Cls, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        # 三层 Set Abstraction
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, 
            in_channel=in_channel, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, 
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, 
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )
        
        # 分类头
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, num_class)
    
    def forward(self, xyz, return_features=False):
        """
        xyz: (B, 3, N) 或 (B, 6, N) 如果包含法线
        返回: (B, num_class) 分类结果
        """
        B, _, N = xyz.shape
        
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        
        # 三层特征提取
        l1_xyz, l1_points = self.sa1(xyz, norm)  # (B, 3, 512), (B, 128, 512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 3, 1), (B, 1024, 1)
        
        # 全局特征
        global_feat = l3_points.view(B, 1024)  # (B, 1024)
        
        # 分类头
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        
        if return_features:
            return x, global_feat, l1_points, l2_points, l3_points
        return x

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

def prepare_pointcloud_for_pointnet2(points, num_points=1024, device='cpu'):
    """准备点云数据给PointNet++"""
    # 确保点数一致
    if len(points) > num_points:
        # 随机采样
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        # 重复点云直到达到目标点数
        repeat_times = num_points // len(points) + 1
        points = np.tile(points, (repeat_times, 1))[:num_points]
    
    # 转换为PyTorch张量
    points_tensor = torch.from_numpy(points.T).float().unsqueeze(0)  # (1, 3, N)
    
    return points_tensor.to(device)

# ==================== 4. 模型加载和推理 ====================
def load_pointnet2_model(model_path=None, num_classes=2, device='cpu'):
    """加载PointNet++模型 """
    print("🔧 初始化 PointNet++ 模型...")
    
    # 创建模型
    model = PointNet2Cls(num_class=num_classes, normal_channel=False)
    
    # 如果没有权重文件，直接返回随机模型
    if not model_path or not Path(model_path).exists():
        print("  ⚠️  未找到预训练权重，使用随机初始化")
        model.to(device)
        model.eval()
        return model
    
    print(f"  加载预训练权重: {model_path}")
    
    # 尝试多种加载方式
    checkpoint = None
    load_methods = [
        ("方法1: torch.load(weights_only=False)", 
         lambda: torch.load(model_path, map_location=device, weights_only=False)),
        
        ("方法2: torch.load(传统方式)", 
         lambda: torch.load(model_path, map_location=device)),
        
        ("方法3: 使用safe_globals", 
         lambda: load_with_safe_globals(model_path, device)),
        
        ("方法4: 使用pickle", 
         lambda: load_with_pickle(model_path, device))
    ]
    
    for method_name, load_func in load_methods:
        try:
            print(f"  尝试{method_name}...")
            checkpoint = load_func()
            print(f"  ✅ {method_name} 成功")
            break
        except Exception as e:
            print(f"  ❌ {method_name} 失败: {str(e)[:100]}")
            continue
    
    if checkpoint is None:
        print("  ⚠️  所有加载方法都失败，使用随机初始化")
        model.to(device)
        model.eval()
        return model
    
    # 提取state_dict
    state_dict = extract_state_dict(checkpoint)
    
    if state_dict is None:
        print("  ⚠️  无法提取state_dict，使用随机初始化")
    else:
        # 加载权重
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"  ✅ 权重加载成功")
            if missing_keys:
                print(f"  ⚠️  缺失的键: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"  ⚠️  意外的键: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        except Exception as e:
            print(f"  ❌ 权重加载失败: {e}")
            print("  使用随机初始化权重")
    
    model.to(device)
    model.eval()
    return model

def load_with_safe_globals(model_path, device):
    """使用safe_globals加载"""
    import torch.serialization
    from numpy.core.multiarray import scalar
    
    # 添加安全全局变量
    torch.serialization.add_safe_globals([scalar])
    
    with torch.serialization.safe_globals([scalar]):
        return torch.load(model_path, map_location=device, weights_only=True)

def load_with_pickle(model_path, device):
    """使用pickle加载"""
    import pickle
    
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    return checkpoint

def extract_state_dict(checkpoint):
    """从checkpoint中提取state_dict"""
    if isinstance(checkpoint, dict):
        # 尝试不同的键名
        possible_keys = ['model_state_dict', 'state_dict', 'model', 'net']
        
        for key in possible_keys:
            if key in checkpoint:
                return checkpoint[key]
        
        # 检查是否已经是state_dict格式
        if all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
            return checkpoint
    
    elif isinstance(checkpoint, nn.Module):
        # 如果是模型实例
        return checkpoint.state_dict()
    
    return None

def inference_with_pointnet2(model, points_tensor, device='cuda'):
    """使用PointNet++进行推理"""
    print("🔍 运行推理...")
    
    with torch.no_grad():
        # 前向传播
        logits, global_feat, l1_feat, l2_feat, l3_feat = model(
            points_tensor, return_features=True
        )
        
        # 计算概率
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    print(f"  📊 预测类别: {pred_class}, 置信度: {confidence:.4f}")
    print(f"  📐 全局特征维度: {global_feat.shape}")
    print(f"  📐 第一层特征维度: {l1_feat.shape}")
    print(f"  📐 第二层特征维度: {l2_feat.shape}")
    print(f"  📐 第三层特征维度: {l3_feat.shape}")
    
    return {
        'logits': logits,
        'probs': probs,
        'pred_class': pred_class,
        'confidence': confidence,
        'global_feat': global_feat,
        'l1_feat': l1_feat,
        'l2_feat': l2_feat,
        'l3_feat': l3_feat
    }

# ==================== 5. 可视化函数 ====================
def visualize_pointnet2_results(points, results, num_points=1024):
    """可视化PointNet++结果"""
    print("📈 生成可视化...")
    
    # 提取特征
    global_feat = results['global_feat'].cpu().numpy().flatten()
    l1_feat = results['l1_feat'].cpu().numpy()[0, :, :]  # (128, 512)
    l2_feat = results['l2_feat'].cpu().numpy()[0, :, :]  # (256, 128)
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 1. 使用第一层特征的第一个通道作为颜色
    if len(points) == l1_feat.shape[1]:
        # 将特征映射到颜色
        feature_for_color = l1_feat[0, :]  # 使用第一个通道
        colors = np.zeros((len(points), 3))
        
        # 归一化特征值
        feat_min, feat_max = feature_for_color.min(), feature_for_color.max()
        if feat_max > feat_min:
            normalized = (feature_for_color - feat_min) / (feat_max - feat_min)
        else:
            normalized = np.zeros_like(feature_for_color)
        
        # 使用jet colormap
        colors[:, 0] = np.clip(1.5 - 4.0 * np.abs(normalized - 0.75), 0, 1)  # 红色
        colors[:, 1] = np.clip(1.5 - 4.0 * np.abs(normalized - 0.5), 0, 1)   # 绿色
        colors[:, 2] = np.clip(1.5 - 4.0 * np.abs(normalized - 0.25), 0, 1)  # 蓝色
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 2. 特征可视化
    fig = plt.figure(figsize=(15, 10))
    
    # 子图1: 全局特征
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(range(min(50, len(global_feat))), global_feat[:50])
    ax1.set_title("全局特征 (前50维)")
    ax1.set_xlabel("维度")
    ax1.set_ylabel("值")
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 第一层特征均值
    ax2 = plt.subplot(2, 3, 2)
    l1_mean = l1_feat.mean(axis=1)
    ax2.bar(range(min(50, len(l1_mean))), l1_mean[:50])
    ax2.set_title("第一层特征均值 (前50通道)")
    ax2.set_xlabel("通道")
    ax2.set_ylabel("均值")
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 第二层特征均值
    ax3 = plt.subplot(2, 3, 3)
    l2_mean = l2_feat.mean(axis=1)
    ax3.bar(range(min(50, len(l2_mean))), l2_mean[:50])
    ax3.set_title("第二层特征均值 (前50通道)")
    ax3.set_xlabel("通道")
    ax3.set_ylabel("均值")
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 特征分布
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(global_feat, bins=50, alpha=0.7, label='全局')
    ax4.hist(l1_feat.flatten(), bins=50, alpha=0.7, label='第一层')
    ax4.hist(l2_feat.flatten(), bins=50, alpha=0.7, label='第二层')
    ax4.set_title("特征值分布")
    ax4.set_xlabel("值")
    ax4.set_ylabel("频数")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 子图5: 置信度
    ax5 = plt.subplot(2, 3, 5)
    classes = ['Class 0', 'Class 1']
    probs = results['probs'].cpu().numpy().flatten()
    colors_bar = ['red' if i == results['pred_class'] else 'blue' for i in range(len(probs))]
    bars = ax5.bar(classes[:len(probs)], probs, color=colors_bar)
    ax5.set_ylim([0, 1.1])
    ax5.set_title(f"分类置信度: {results['confidence']:.2%}")
    ax5.set_ylabel("概率")
    ax5.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.3f}', ha='center', va='bottom')
    
    # 子图6: 特征相关性
    ax6 = plt.subplot(2, 3, 6)
    if len(points) > 0 and 'colors' in locals():
        # 显示特征值的空间分布
        scatter = ax6.scatter(points[:, 0], points[:, 1], 
                            c=colors[:, 0], cmap='viridis', s=10)
        ax6.set_title("特征空间分布 (X-Y平面)")
        ax6.set_xlabel("X")
        ax6.set_ylabel("Y")
        ax6.axis('equal')
        plt.colorbar(scatter, ax=ax6, label='特征强度')
    
    plt.suptitle(f"PointNet++ 推理结果 - 预测类别: {results['pred_class']}, 置信度: {results['confidence']:.2%}", 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pointnet2_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("  📸 结果图已保存为 'pointnet2_results.png'")
    
    # 显示点云
    print("\n👀 显示点云 (按ESC或Q退出)...")
    o3d.visualization.draw_geometries(
        [pcd], 
        window_name=f"Stanford Bunny - PointNet++ 特征可视化 (置信度: {results['confidence']:.2%})",
        width=800, height=600
    )

# ==================== 6. 主函数 ====================
def main():
    """主函数"""
    print("="*60)
    print("PointNet++ 点云分类与特征提取")
    print("="*60)
    
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ 使用设备: {device}")
    
    # 模型路径 (修改为你下载的权重路径)
    MODEL_PATH = "./best_model.pth"  # 或 "pointnet2_cls.pth"
    
    # 参数
    NUM_POINTS = 1024
    NUM_CLASSES = 2  # 根据你的预训练模型修改
    
    # 1. 加载模型
    model = load_pointnet2_model(MODEL_PATH, NUM_CLASSES, device)
    
    # 2. 加载点云
    points = load_stanford_bunny(NUM_POINTS)
    
    # 3. 准备数据
    points_tensor = prepare_pointcloud_for_pointnet2(points, NUM_POINTS, device)
    
    # 4. 推理
    results = inference_with_pointnet2(model, points_tensor, device)
    
    # 5. 可视化
    visualize_pointnet2_results(points, results, NUM_POINTS)
    
    print("\n" + "="*60)
    print("✅ PointNet++ 推理完成!")
    print("="*60)

if __name__ == "__main__":
    main()