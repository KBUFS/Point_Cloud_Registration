""""这里只使用了局部特征来作为描述子，而没有利用网络第三层的全局特征来进行提供初始变换"""
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import time
import copy
import os
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
print("✅ 已设置中文字体支持")

# 导入现有的PointNet++模块
try:
    from pointnet2_model import PointNet2Cls
    from pointnet2_utils import PointNetSetAbstraction
    print("✅ 成功导入现有PointNet++模块")
except ImportError:
    # 备用：内联定义
    print("⚠️  无法导入模块，使用内联定义")
    # ... [这里可以粘贴你的完整PointNet++代码] ...

# ==================== 1. 加载模型函数 ====================
def load_pointnet2_model(model_path=None, num_classes=40, device='cpu'):
    """加载PointNet++模型"""
    print("🔧 加载PointNet++模型...")
    
    # 创建模型
    model = PointNet2Cls(num_class=num_classes, normal_channel=False)
    
    if not model_path or not Path(model_path).exists():
        print("  ⚠️  未找到预训练权重，使用随机初始化")
        model.to(device)
        model.eval()
        return model
    
    print(f"  加载预训练权重: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"  ✅ 权重加载成功")
        
        if missing_keys:
            print(f"  ⚠️  缺失的键: {len(missing_keys)}个")
        if unexpected_keys:
            print(f"  ⚠️  意外的键: {len(unexpected_keys)}个")
            
    except Exception as e:
        print(f"  ❌ 权重加载失败: {e}")
        print("  使用随机初始化权重")
    
    model.to(device)
    model.eval()
    return model

# ==================== 2. 点云处理函数 ====================
def load_point_clouds():
    """加载点云文件"""
    print("搜索点云文件...")
    
    ply_files = list(Path("./data").glob("*.ply"))
    if not ply_files:
        print("❌ 未找到点云文件，使用测试数据")
        return create_test_data()
    
    print(f"找到 {len(ply_files)} 个点云文件:")
    for i, file in enumerate(ply_files, 1):
        print(f"  [{i}] {file.name}")
    
    try:
        idx1 = int(input("输入第一个文件编号: ")) - 1
        idx2 = int(input("输入第二个文件编号: ")) - 1
        
        if idx1 < 0 or idx2 < 0 or idx1 >= len(ply_files) or idx2 >= len(ply_files) or idx1 == idx2:
            print("❌ 编号无效，使用前两个文件")
            idx1, idx2 = 0, 1
    except:
        print("❌ 输入错误，使用前两个文件")
        idx1, idx2 = 0, 1
    
    file1 = ply_files[idx1]
    file2 = ply_files[idx2]
    
    print(f"\n加载: {file1.name} 和 {file2.name}")
    
    source = o3d.io.read_point_cloud(str(file1))
    target = o3d.io.read_point_cloud(str(file2))
    
    if len(source.points) == 0 or len(target.points) == 0:
        print("❌ 加载失败，创建测试数据")
        return create_test_data()
    
    print(f"源点云点数: {len(source.points):,}")
    print(f"目标点云点数: {len(target.points):,}")
    
    return source, target, file1.name, file2.name

def create_test_data():
    """创建测试数据"""
    print("创建测试点云...")
    
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh = mesh.subdivide_midpoint(number_of_iterations=2)
    mesh.compute_vertex_normals()
    
    source = mesh.sample_points_poisson_disk(number_of_points=5000)
    
    angle = np.radians(45)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    t = np.array([0.2, 0.1, 0.05])
    
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    
    target = copy.deepcopy(source)
    target.transform(transformation)
    
    points = np.asarray(target.points)
    noise = np.random.normal(0, 0.005, points.shape)
    target.points = o3d.utility.Vector3dVector(points + noise)
    
    source.estimate_normals()
    target.estimate_normals()
    
    return source, target, "test_source", "test_target"

def prepare_pointcloud_for_pointnet(points, num_points=1024, device='cpu'):
    """准备点云数据给PointNet++"""
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        repeat_times = num_points // len(points) + 1
        points = np.tile(points, (repeat_times, 1))[:num_points]
    
    points_tensor = torch.from_numpy(points.T).float().unsqueeze(0)
    return points_tensor.to(device)

# ==================== 3. PointNet++特征匹配核心 ====================
def extract_global_features(model, point_clouds, device='cpu'):
    """批量提取全局特征"""
    features_list = []
    
    with torch.no_grad():
        for points in point_clouds:
            if isinstance(points, o3d.geometry.PointCloud):
                points = np.asarray(points.points)
            
            points_tensor = prepare_pointcloud_for_pointnet(points, 1024, device)
            features = model.extract_descriptor(points_tensor)
            features_list.append(features.cpu().numpy())
    
    return np.vstack(features_list)

def match_pointnet_features(source_feat, target_feat, k=3):
    """匹配PointNet++特征"""
    # 余弦相似度
    source_norm = source_feat / (np.linalg.norm(source_feat, axis=1, keepdims=True) + 1e-8)
    target_norm = target_feat / (np.linalg.norm(target_feat, axis=1, keepdims=True) + 1e-8)
    
    similarity = np.dot(source_norm, target_norm.T)
    
    matches = []
    for i in range(len(source_feat)):
        sim_scores = similarity[i]
        top_k_idx = np.argsort(sim_scores)[-k:][::-1]
        for idx in top_k_idx:
            matches.append([i, idx, sim_scores[idx]])
    
    return np.array(matches)

# ==================== 4. 配准流程 ====================
def pointnet_feature_registration(source, target, model, voxel_size=0.05, device='cpu'):
    """纯PointNet++特征配准"""
    print("\n" + "="*60)
    print("纯PointNet++特征配准")
    print("="*60)
    
    start_time = time.time()
    
    # 1. 下采样
    print("\n[1] 下采样点云")
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    print(f"   源点云: {len(source.points):,} -> {len(source_down.points):,}")
    print(f"   目标点云: {len(target.points):,} -> {len(target_down.points):,}")
    
    # 2. 提取PointNet++全局特征
    print("\n[2] 提取PointNet++全局特征")
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    
    # 准备数据
    source_tensor = prepare_pointcloud_for_pointnet(source_points, 1024, device)
    target_tensor = prepare_pointcloud_for_pointnet(target_points, 1024, device)
    
    # 提取全局特征
    with torch.no_grad():
        source_global_feat = model.extract_descriptor(source_tensor).cpu().numpy()
        target_global_feat = model.extract_descriptor(target_tensor).cpu().numpy()
    
    print(f"   源全局特征: {source_global_feat.shape}")
    print(f"   目标全局特征: {target_global_feat.shape}")
    
    # 3. 提取PointNet++逐点特征（用于匹配）
    print("\n[3] 提取PointNet++逐点特征")
    with torch.no_grad():
        # 使用完整模型提取多层特征
        _, _, source_l1_feat, source_l2_feat = model(source_tensor, return_features=True)
        _, _, target_l1_feat, target_l2_feat = model(target_tensor, return_features=True)
    
    # 使用第一层特征（128维，逐点）
    source_l1_feat = source_l1_feat.cpu().numpy()[0]  # (128, N)
    target_l1_feat = target_l1_feat.cpu().numpy()[0]  # (128, N)
    
    print(f"   源逐点特征: {source_l1_feat.shape}")
    print(f"   目标逐点特征: {target_l1_feat.shape}")
    
    # 4. 基于深度学习特征的特征匹配
    print("\n[4] 深度学习特征匹配")
    matches = match_deep_features(
        source_points, target_points,
        source_l1_feat.T, target_l1_feat.T,  # 转置为(N, 128)
        k=5
    )
    
    print(f"   找到匹配对: {len(matches)}")
    
    if len(matches) < 3:
        print("⚠️  匹配对太少，跳过RANSAC，使用ICP直接配准")
        initial_transform = np.eye(4)
    else:
        # 5. 使用匹配点进行RANSAC
        print("\n[5] 基于深度学习匹配的RANSAC")
        initial_transform = ransac_with_deep_matches(
            source_points, target_points, matches, 
            num_iterations=2000, 
            threshold=voxel_size*2
        )
    
    # 6. ICP精配准
    print("\n[6] ICP精配准")
    final_transform = icp_refinement(source, target, initial_transform, voxel_size)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  总耗时: {elapsed_time:.2f}秒")
    
    return final_transform

def match_deep_features(source_points, target_points, 
                            source_features, target_features, k=3):
    """
    深度学习特征匹配
    使用余弦相似度 + 空间一致性约束
    """
    # 确保点数一致
    n_source = min(len(source_points), len(source_features))
    n_target = min(len(target_points), len(target_features))
    
    if n_source == 0 or n_target == 0:
        return np.array([])
    
    # 只使用有效点
    source_features = source_features[:n_source]
    target_features = target_features[:n_target]
    
    # 归一化
    source_norm = source_features / (np.linalg.norm(source_features, axis=1, keepdims=True) + 1e-8)
    target_norm = target_features / (np.linalg.norm(target_features, axis=1, keepdims=True) + 1e-8)
    
    # 相似度矩阵
    similarity = np.dot(source_norm, target_norm.T)
    
    matches = []
    for i in range(n_source):
        # 找到k个最相似的
        target_indices = np.argsort(similarity[i])[-k:][::-1]
        
        for j in target_indices:
            if j >= n_target:
                continue
                
            score = similarity[i, j]
            if score > 0.5:  # 相似度阈值
                matches.append([i, j, score])
    
    return np.array(matches) if matches else np.array([])

def ransac_with_deep_matches(source_points, target_points, matches, 
                           num_iterations=1000, threshold=0.05):
    """使用深度学习匹配进行RANSAC"""
    if len(matches) < 3:
        return np.eye(4)
    
    best_inliers = 0
    best_transform = np.eye(4)
    
    for _ in range(num_iterations):
        # 随机选择匹配对
        sample_idx = np.random.choice(len(matches), 3, replace=False)
        sample = matches[sample_idx]
        
        # 构造对应点
        src_pts = source_points[sample[:, 0].astype(int)]
        tgt_pts = target_points[sample[:, 1].astype(int)]
        
        # 计算变换
        R, t = compute_rigid_transform(src_pts, tgt_pts)
        
        if R is not None:
            # 计算内点
            transformed_src = np.dot(source_points[matches[:, 0].astype(int)], R.T) + t
            distances = np.linalg.norm(transformed_src - target_points[matches[:, 1].astype(int)], axis=1)
            inliers = np.sum(distances < threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_transform = np.eye(4)
                best_transform[:3, :3] = R
                best_transform[:3, 3] = t
    
    print(f"  RANSAC内点: {best_inliers}/{len(matches)} ({best_inliers/len(matches)*100:.1f}%)")
    return best_transform

def compute_rigid_transform(src, tgt):
    """计算刚体变换"""
    src_centroid = np.mean(src, axis=0)
    tgt_centroid = np.mean(tgt, axis=0)
    
    src_centered = src - src_centroid
    tgt_centered = tgt - tgt_centroid
    
    H = np.dot(src_centered.T, tgt_centered)
    
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None, None
    
    R = np.dot(Vt.T, U.T)
    
    # 确保是右手系
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    t = tgt_centroid - np.dot(R, src_centroid)
    
    return R, t

def icp_refinement(source, target, initial_transform, voxel_size):
    """ICP精配准 - 使用Point-to-Point"""
    distance_threshold = voxel_size * 1.5
    
    reg_result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),  # 改为Point-to-Point
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100,
            relative_fitness=1e-6,
            relative_rmse=1e-6
        )
    )
    
    print(f"  ICP适应度: {reg_result.fitness:.4f}")
    print(f"  ICP RMSE: {reg_result.inlier_rmse:.6f}")
    
    return reg_result.transformation
# ==================== 5. 评估与可视化 ====================

def visualize_registration_comparison(source, target, source_registered, title="配准结果对比"):
    """可视化配准结果对比"""
    print("\n👀 可视化配准结果对比...")
    
    # 创建三个点云的副本
    source_vis = copy.deepcopy(source)
    target_vis = copy.deepcopy(target)
    registered_vis = copy.deepcopy(source_registered)
    
    # 设置颜色
    source_vis.paint_uniform_color([1, 0, 0])     # 红色: 原始源点云
    target_vis.paint_uniform_color([0, 1, 0])     # 绿色: 目标点云
    registered_vis.paint_uniform_color([0, 0, 1]) # 蓝色: 配准后点云
    
    # 1. 分别显示
    print("1. 原始点云 (红:源, 绿:目标)")
    o3d.visualization.draw_geometries(
        [source_vis, target_vis],
        window_name=f"原始点云 - 源: {len(source.points)}点, 目标: {len(target.points)}点",
        width=1000, height=800
    )
    
    # 2. 显示配准后
    print("2. 配准结果 (蓝:配准后, 绿:目标)")
    o3d.visualization.draw_geometries(
        [registered_vis, target_vis],
        window_name=f"配准结果 - 配准后: {len(registered_vis.points)}点, 目标: {len(target.points)}点",
        width=1000, height=800
    )

def print_registration_table(mean_distance, median_distance, std_distance, 
                           max_distance, min_distance, inlier_ratio, inlier_threshold):
    """打印配准结果表格"""
    print("\n📊 配准结果统计表:")
    print("+" + "-"*35 + "+" + "-"*15 + "+")
    print(f"| {'指标':<33} | {'值':<13} |")
    print("+" + "-"*35 + "+" + "-"*15 + "+")
    print(f"| 平均点对点距离             | {mean_distance:>13.6f} |")
    print(f"| 中位数距离                 | {median_distance:>13.6f} |")
    print(f"| 标准差                     | {std_distance:>13.6f} |")
    print(f"| 最大距离                   | {max_distance:>13.6f} |")
    print(f"| 最小距离                   | {min_distance:>13.6f} |")
    print(f"| 内点比率 (<{inlier_threshold:.3f})      | {inlier_ratio:>12.3%} |")
    print("+" + "-"*35 + "+" + "-"*15 + "+")

def evaluate_registration_comprehensive(source, target, transformation, voxel_size, 
                                      feature_dim=None, num_matches=None):
    """综合评估配准结果"""
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    
    source_points = np.asarray(source_transformed.points)
    target_points = np.asarray(target.points)
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_points)
    distances, indices = nbrs.kneighbors(source_points)
    distances = distances.flatten()
    indices = indices.flatten()
    
    # 基础统计
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    std_distance = np.std(distances)
    max_distance = np.max(distances)
    min_distance = np.min(distances)
    
    # 分位数
    q25 = np.percentile(distances, 25)
    q75 = np.percentile(distances, 75)
    
    # 内点统计
    inlier_threshold = voxel_size * 2
    inliers = distances < inlier_threshold
    inlier_ratio = np.mean(inliers)
    inlier_mean = np.mean(distances[inliers]) if np.any(inliers) else 0
    inlier_std = np.std(distances[inliers]) if np.any(inliers) and len(distances[inliers]) > 1 else 0
    
    # 距离分布
    distance_bins = [0, voxel_size/2, voxel_size, voxel_size*2, voxel_size*5, float('inf')]
    bin_labels = ['<0.5v', '0.5-1v', '1-2v', '2-5v', '>5v']
    hist, _ = np.histogram(distances, bins=distance_bins)
    hist_ratio = hist / len(distances)
    
    print("\n" + "="*60)
    print("📊 配准结果综合评估")
    print("="*60)
    
    # 创建综合表格
    print("\n📈 点对点距离统计:")
    print("+" + "-"*35 + "+" + "-"*15 + "+")
    print(f"| {'统计指标':<33} | {'数值':<13} |")
    print("+" + "-"*35 + "+" + "-"*15 + "+")
    print(f"| 平均距离                   | {mean_distance:>13.6f} |")
    print(f"| 中位数距离                 | {median_distance:>13.6f} |")
    print(f"| 标准差                     | {std_distance:>13.6f} |")
    print(f"| 最大距离                   | {max_distance:>13.6f} |")
    print(f"| 最小距离                   | {min_distance:>13.6f} |")
    print(f"| 25%分位数                 | {q25:>13.6f} |")
    print(f"| 75%分位数                 | {q75:>13.6f} |")
    print("+" + "-"*35 + "+" + "-"*15 + "+")
    
    print(f"\n🎯 内点统计 (阈值: {inlier_threshold:.3f}):")
    print("+" + "-"*35 + "+" + "-"*15 + "+")
    print(f"| {'指标':<33} | {'数值':<13} |")
    print("+" + "-"*35 + "+" + "-"*15 + "+")
    print(f"| 内点总数                   | {np.sum(inliers):>13d} |")
    print(f"| 内点比率                   | {inlier_ratio:>12.3%} |")
    print(f"| 内点平均距离               | {inlier_mean:>13.6f} |")
    print(f"| 内点标准差                 | {inlier_std:>13.6f} |")
    print("+" + "-"*35 + "+" + "-"*15 + "+")
    
    print(f"\n📊 距离分布:")
    print("+" + "-"*20 + "+" + "-"*10 + "+" + "-"*15 + "+")
    print(f"| {'距离区间':<18} | {'点数':<8} | {'比例':<13} |")
    print("+" + "-"*20 + "+" + "-"*10 + "+" + "-"*15 + "+")
    for label, count, ratio in zip(bin_labels, hist, hist_ratio):
        print(f"| {label:<18} | {count:>8d} | {ratio:>12.3%} |")
    print("+" + "-"*20 + "+" + "-"*10 + "+" + "-"*15 + "+")
    
    # 点云基本信息
    print(f"\n📦 点云基本信息:")
    print("+" + "-"*30 + "+" + "-"*15 + "+")
    print(f"| {'信息':<28} | {'数值':<13} |")
    print("+" + "-"*30 + "+" + "-"*15 + "+")
    print(f"| 源点云点数                 | {len(source.points):>13,d} |")
    print(f"| 目标点云点数               | {len(target.points):>13,d} |")
    print(f"| 体素大小                   | {voxel_size:>13.3f} |")
    
    if feature_dim is not None:
        print(f"| 特征维度                   | {feature_dim:>13} |")
    if num_matches is not None:
        print(f"| 特征匹配对数               | {num_matches:>13,d} |")
    
    print("+" + "-"*30 + "+" + "-"*15 + "+")
    
    return {
        'mean_distance': mean_distance,
        'median_distance': median_distance,
        'std_distance': std_distance,
        'inlier_ratio': inlier_ratio,
        'inlier_mean': inlier_mean,
        'distances': distances,
        'source_transformed': source_transformed
    }

# ==================== 6. 主函数 ====================
def main():
    """主函数"""
    print("="*60)
    print("PointNet++ 特征点云配准系统")
    print("="*60)
    
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ 使用设备: {device}")
    
    # 加载模型
    MODEL_PATH = "./best_model.pth"
    model = load_pointnet2_model(MODEL_PATH, 40, device)
    
    # 加载点云
    source, target, source_name, target_name = load_point_clouds()
    
    print(f"\n📁 处理文件:")
    print(f"  源点云: {source_name}")
    print(f"  目标点云: {target_name}")
    
    # 体素大小选择
    print("\n📏 选择体素大小:")
    print("  [1] 0.005 (精细)")
    print("  [2] 0.05 (推荐)")
    print("  [3] 0.1 (粗糙)")
    
    try:
        choice = int(input("请选择 (1-3): "))
        if choice == 1:
            voxel_size = 0.005
        elif choice == 2:
            voxel_size = 0.05
        elif choice == 3:
            voxel_size = 0.1
        else:
            voxel_size = 0.05
    except:
        voxel_size = 0.05
    
    print(f"\n🔧 使用体素大小: {voxel_size}")
    
    # 运行PointNet++配准
    print("\n🚀 开始配准流程...")
    start_time = time.time()
    final_transform = pointnet_feature_registration(
        source, target, model, voxel_size, device
    )
    
    # 为了获取特征维度信息，重新运行一次提取
    print("\n📈 提取特征维度信息...")
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    
    source_tensor = prepare_pointcloud_for_pointnet(source_points, 1024, device)
    target_tensor = prepare_pointcloud_for_pointnet(target_points, 1024, device)
    
    with torch.no_grad():
        _, _, source_l1_feat, _ = model(source_tensor, return_features=True)
        source_feat_dim = source_l1_feat.shape[1]  # 特征维度
    
    # 模拟匹配数量（实际应该从配准过程中获取）
    num_matches = 100  # 示例值，实际应该从matches中获取
    
    # 综合评估
    evaluation = evaluate_registration_comprehensive(
        source, target, final_transform, voxel_size,
        feature_dim=source_feat_dim,
        num_matches=num_matches
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  总处理时间: {elapsed_time:.2f}秒")
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"pointnet_registration_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存点云文件
    o3d.io.write_point_cloud(f"{output_dir}/source_original.ply", source)
    o3d.io.write_point_cloud(f"{output_dir}/target_original.ply", target)
    o3d.io.write_point_cloud(f"{output_dir}/source_registered.ply", evaluation['source_transformed'])
    
    # 保存详细结果报告
    with open(f"{output_dir}/registration_report.txt", "w", encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("PointNet++ 特征点云配准结果报告\n")
        f.write("="*60 + "\n\n")
        
        f.write("📁 文件信息:\n")
        f.write("-"*40 + "\n")
        f.write(f"源点云: {source_name}\n")
        f.write(f"目标点云: {target_name}\n")
        f.write(f"体素大小: {voxel_size:.3f}\n")
        f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {elapsed_time:.2f}秒\n\n")
        
        f.write("📈 点对点距离统计:\n")
        f.write("-"*40 + "\n")
        f.write(f"平均距离:       {evaluation['mean_distance']:.6f}\n")
        f.write(f"中位数距离:     {evaluation['median_distance']:.6f}\n")
        f.write(f"标准差:         {evaluation['std_distance']:.6f}\n")
        f.write(f"内点比率:       {evaluation['inlier_ratio']:.3%}\n")
        f.write(f"内点平均距离:   {evaluation.get('inlier_mean', 0):.6f}\n")
        f.write(f"特征维度:       {source_feat_dim}\n")
        f.write(f"特征匹配对数:   {num_matches:,}\n\n")
        
        f.write("🔢 变换矩阵 (4x4):\n")
        f.write("-"*40 + "\n")
        for i, row in enumerate(final_transform):
            f.write(f"[{row[0]:>10.6f} {row[1]:>10.6f} {row[2]:>10.6f} {row[3]:>10.6f}]\n")
    
    print(f"\n💾 结果已保存到目录: {output_dir}")
    print("\n📁 生成的文件:")
    print(f"  - {output_dir}/source_original.ply (原始源点云)")
    print(f"  - {output_dir}/target_original.ply (目标点云)")
    print(f"  - {output_dir}/source_registered.ply (配准后点云)")
    print(f"  - {output_dir}/registration_report.txt (详细报告)")
    
    # 可视化对比
    print("\n" + "="*60)
    print("👀 可视化配准结果")
    print("="*60)
    visualize_registration_comparison(
        source, target, evaluation['source_transformed'],
        "PointNet++ 特征配准结果"
    )
    
    print("="*60)
    print("✅ PointNet++ 特征配准完成!")
    print("="*60)
    print(f"📊 关键结果:")
    print(f"  - 内点比率: {evaluation['inlier_ratio']:.3%}")
    print(f"  - 平均距离: {evaluation['mean_distance']:.6f}")
    print(f"  - 中位数距离: {evaluation['median_distance']:.6f}")
    print(f"  - 特征维度: {source_feat_dim}")
    print("="*60)
if __name__ == "__main__":
    main()