"""
PointNet++对90°旋转配准差是因为它学的是全局坐标系下的特征，而不是旋转不变的特征。
PointNet++在ModelNet40上训练，数据增强通常是：小角度旋转（±15°）随机平移均匀缩放但极少有90°这种离散大角度旋转！
所以若要针对大角度旋转点云进行配准需要对数据集进行数据增强重新训练权重。
这里使用了过程中生成的全局特征进行对初始位姿进行变化，同样在多视角点云的配准中可以利用全局特征来快速判断哪些点云可以进行配准。
"""

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

# ==================== 3. 工具函数 ====================
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

def compute_pca(points):
    """计算点云的主成分分析"""
    if len(points) < 3:
        return np.eye(3)
    
    # 去中心化
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # 计算协方差矩阵
    cov_matrix = np.cov(points_centered, rowvar=False)
    
    # 特征值分解
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 按特征值降序排序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # 确保是右手坐标系
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 0] *= -1
        
        return eigenvectors
    except:
        return np.eye(3)

def rotation_matrix_from_euler(angles):
    """从欧拉角创建旋转矩阵"""
    rx, ry, rz = angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    return np.dot(Rz, np.dot(Ry, Rx))

def apply_transform_to_points(points, transform):
    """将变换应用到点云"""
    R = transform[:3, :3]
    t = transform[:3, 3]
    return np.dot(points, R.T) + t

def icp_refinement(source, target, initial_transform, voxel_size):
    """ICP精配准 - 使用Point-to-Point"""
    distance_threshold = voxel_size * 1.5
    
    reg_result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100,
            relative_fitness=1e-6,
            relative_rmse=1e-6
        )
    )
    
    print(f"  ICP适应度: {reg_result.fitness:.4f}")
    print(f"  ICP RMSE: {reg_result.inlier_rmse:.6f}")
    
    return reg_result.transformation

# ==================== 4. 全局特征辅助的配准流程 ====================
def estimate_transform_from_global_features(source_global, target_global, source_points, target_points):
    """从全局特征估算初始变换"""
    try:
        # 对点云进行PCA
        source_pca = compute_pca(source_points)
        target_pca = compute_pca(target_points)
        
        # PCA矩阵的列是主方向
        R = np.dot(target_pca.T, source_pca)
        
        # 确保是旋转矩阵
        U, _, Vt = np.linalg.svd(R)
        R = np.dot(U, Vt)
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(U, Vt)
        
        # 估算平移
        source_center = np.mean(source_points, axis=0)
        target_center = np.mean(target_points, axis=0)
        t = target_center - np.dot(R, source_center)
        
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        return transform
        
    except Exception as e:
        print(f"   PCA估算失败: {e}")
        return np.eye(4)

def match_deep_features_improved(source_points, target_points, 
                                source_features, target_features, 
                                global_similarity, k=3):
    """
    改进版深度学习特征匹配
    结合全局特征信息调整匹配策略
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
    
    # 根据全局相似度调整匹配策略
    if global_similarity > 0.6:
        # 全局特征相似度高，使用严格的匹配
        similarity_threshold = 0.6
        k_matches = k
    elif global_similarity > 0.3:
        # 中等相似度，使用中等阈值
        similarity_threshold = 0.5
        k_matches = k + 1
    else:
        # 低相似度，降低阈值，增加匹配数
        similarity_threshold = 0.4
        k_matches = k + 2
    
    print(f"   匹配参数: 阈值={similarity_threshold:.2f}, k={k_matches}")
    
    for i in range(n_source):
        # 找到k个最相似的
        target_indices = np.argsort(similarity[i])[-k_matches:][::-1]
        
        for j in target_indices:
            if j >= n_target:
                continue
                
            score = similarity[i, j]
            if score > similarity_threshold:
                # 双向验证
                source_indices = np.argsort(similarity[:, j])[-k_matches:][::-1]
                
                # 检查是否互为最近邻
                mutual_ratio = np.sum(similarity[i, j] > similarity[i, target_indices]) / k_matches
                
                if mutual_ratio > 0.5:  # 相互一致性
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

def ransac_with_deep_matches_and_initial(source_points, target_points, matches, 
                                       initial_transform, num_iterations=1000, threshold=0.05):
    """使用深度学习匹配和初始变换进行RANSAC"""
    if len(matches) < 3:
        return initial_transform
    
    best_inliers = 0
    best_transform = initial_transform
    
    for _ in range(num_iterations):
        # 以一定概率使用初始变换，或以一定概率随机采样
        if np.random.random() < 0.3:  # 30%概率从初始变换开始
            R = initial_transform[:3, :3]
            t = initial_transform[:3, 3]
            
            # 添加小扰动
            angle_noise = np.random.uniform(-0.1, 0.1, 3)  # 小角度噪声
            R_noise = rotation_matrix_from_euler(angle_noise)
            R_current = np.dot(R_noise, R)
            t_current = t + np.random.uniform(-0.01, 0.01, 3)
            
            transform_current = np.eye(4)
            transform_current[:3, :3] = R_current
            transform_current[:3, 3] = t_current
            
        else:
            # 随机采样3个匹配对
            sample_idx = np.random.choice(len(matches), 3, replace=False)
            sample = matches[sample_idx]
            
            # 构造对应点
            src_pts = source_points[sample[:, 0].astype(int)]
            tgt_pts = target_points[sample[:, 1].astype(int)]
            
            # 计算变换
            R_current, t_current = compute_rigid_transform(src_pts, tgt_pts)
            if R_current is None:
                continue
                
            transform_current = np.eye(4)
            transform_current[:3, :3] = R_current
            transform_current[:3, 3] = t_current
        
        # 计算内点
        transformed_src = np.dot(source_points[matches[:, 0].astype(int)], R_current.T) + t_current
        distances = np.linalg.norm(transformed_src - target_points[matches[:, 1].astype(int)], axis=1)
        inliers = np.sum(distances < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_transform = transform_current
    
    print(f"  RANSAC内点: {best_inliers}/{len(matches)} ({best_inliers/len(matches)*100:.1f}%)")
    return best_transform

def pointnet_feature_registration_improved(source, target, model, voxel_size=0.05, device='cpu'):
    """改进版PointNet++特征配准 - 利用全局特征处理大角度旋转"""
    print("\n" + "="*60)
    print("改进版PointNet++特征配准 (全局特征辅助)")
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
    
    with torch.no_grad():
        # 提取全局特征
        source_global_feat = model.extract_descriptor(source_tensor).cpu().numpy()
        target_global_feat = model.extract_descriptor(target_tensor).cpu().numpy()
        
        # 提取逐点特征
        _, _, source_l1_feat, source_l2_feat = model(source_tensor, return_features=True)
        _, _, target_l1_feat, target_l2_feat = model(target_tensor, return_features=True)
    
    # 特征维度
    source_l1_feat = source_l1_feat.cpu().numpy()[0]  # (128, N)
    target_l1_feat = target_l1_feat.cpu().numpy()[0]  # (128, N)
    
    print(f"   源全局特征: {source_global_feat.shape}")
    print(f"   目标全局特征: {target_global_feat.shape}")
    print(f"   源逐点特征: {source_l1_feat.shape}")
    print(f"   目标逐点特征: {target_l1_feat.shape}")
    
    # 3. 全局特征分析
    print("\n[3] 全局特征分析")
    source_global = source_global_feat.flatten()
    target_global = target_global_feat.flatten()
    
    # 计算全局特征相似度
    global_similarity = np.dot(source_global, target_global) / (
        np.linalg.norm(source_global) * np.linalg.norm(target_global) + 1e-8
    )
    
    print(f"   全局特征相似度: {global_similarity:.3f}")
    
    # 评估全局特征质量
    if global_similarity < 0.2:
        print("   ⚠️ 全局特征相似度过低，可能是不同物体")
    elif global_similarity < 0.4:
        print("   ℹ️ 全局特征相似度中等，可能存在大角度旋转")
    else:
        print("   ✅ 全局特征相似度高，预计配准效果良好")
    
    # 4. 使用全局特征估算初始变换
    print("\n[4] 全局特征估算初始变换")
    initial_transform = estimate_transform_from_global_features(
        source_global, target_global, source_points, target_points
    )
    
    # 记录全局特征是否有效
    global_feature_valid = global_similarity > 0.3
    
    if global_feature_valid:
        print(f"   ✅ 全局特征有效，提供初始变换")
        
        # 使用全局特征提供的初始变换预对齐点云
        source_points_aligned = apply_transform_to_points(
            source_points, initial_transform
        )
        
        # 在对齐后的点云上重新提取特征
        source_tensor_aligned = prepare_pointcloud_for_pointnet(
            source_points_aligned, 1024, device
        )
        
        with torch.no_grad():
            _, _, source_l1_feat_aligned, _ = model(source_tensor_aligned, return_features=True)
            source_l1_feat_aligned = source_l1_feat_aligned.cpu().numpy()[0]
    else:
        print(f"   ⚠️ 全局特征相似度低，使用原始点云")
        source_points_aligned = source_points
        source_l1_feat_aligned = source_l1_feat
        initial_transform = np.eye(4)
    
    # 5. 基于深度学习特征的特征匹配
    print("\n[5] 深度学习特征匹配")
    matches = match_deep_features_improved(
        source_points_aligned, target_points,
        source_l1_feat_aligned.T, target_l1_feat.T,  # 转置为(N, 128)
        global_similarity,  # 传入全局相似度
        k=5
    )
    
    print(f"   找到匹配对: {len(matches)}")
    
    if len(matches) < 3:
        print("⚠️  匹配对太少，跳过RANSAC")
        if global_feature_valid:
            refined_transform = initial_transform
        else:
            refined_transform = np.eye(4)
    else:
        # 6. 使用匹配点进行RANSAC
        print("\n[6] 基于深度学习匹配的RANSAC")
        
        if global_feature_valid:
            # 如果全局特征有效，从初始变换开始搜索
            refined_transform = ransac_with_deep_matches_and_initial(
                source_points_aligned, target_points, matches, 
                initial_transform,  # 使用全局特征提供的初始变换
                num_iterations=1000, 
                threshold=voxel_size*2
            )
        else:
            # 如果全局特征无效，从单位矩阵开始搜索
            refined_transform = ransac_with_deep_matches(
                source_points_aligned, target_points, matches, 
                num_iterations=2000,  # 增加迭代次数
                threshold=voxel_size*2
            )
    
    # 如果需要，将变换矩阵转换回原始坐标空间
    if global_feature_valid and not np.allclose(initial_transform, np.eye(4)):
        refined_transform = np.dot(initial_transform, refined_transform)
    
    # 7. ICP精配准
    print("\n[7] ICP精配准")
    final_transform = icp_refinement(source, target, refined_transform, voxel_size)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  总耗时: {elapsed_time:.2f}秒")
    
    return final_transform, global_similarity, len(matches)

# ==================== 5. 可视化功能 ====================
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

def visualize_global_feature_analysis(source, target, source_down, target_down,
                                    source_global, target_global, global_similarity,
                                    initial_transform, voxel_size):
    """可视化全局特征分析结果"""
    print("\n" + "="*60)
    print("👀 全局特征分析可视化")
    print("="*60)
    
    # 1. 原始点云
    print("\n[1] 原始点云")
    source_vis = copy.deepcopy(source_down)
    target_vis = copy.deepcopy(target_down)
    
    source_vis.paint_uniform_color([1, 0, 0])  # 红色
    target_vis.paint_uniform_color([0, 1, 0])  # 绿色
    
    o3d.visualization.draw_geometries(
        [source_vis, target_vis],
        window_name=f"原始点云 (相似度: {global_similarity:.3f})",
        width=1000, height=800
    )
    
    # 2. 全局特征PCA可视化
    print("\n[2] PCA主方向可视化")
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    
    # 创建坐标轴
    source_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    target_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # 将目标坐标轴移动到目标点云中心
    target_center = np.mean(target_points, axis=0)
    target_axes.translate(target_center)
    
    o3d.visualization.draw_geometries(
        [source_vis, target_vis, source_axes, target_axes],
        window_name="坐标轴",
        width=1000, height=800
    )
    
    # 3. 全局特征估算的初始变换
    print("\n[3] 全局特征初始变换")
    source_aligned = copy.deepcopy(source_down)
    source_aligned.transform(initial_transform)
    source_aligned.paint_uniform_color([0, 0, 1])  # 蓝色
    
    o3d.visualization.draw_geometries(
        [source_aligned, target_vis],
        window_name=f"全局特征初始变换 (相似度: {global_similarity:.3f})",
        width=1000, height=800
    )
    
    # 4. 全局特征对比图
    print("\n[4] 全局特征对比图")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 全局特征向量
    axes[0, 0].bar(range(min(50, len(source_global))), source_global[:50], 
                   alpha=0.7, color='red', label='源点云')
    axes[0, 0].set_title("源点云全局特征 (前50维)", fontsize=12)
    axes[0, 0].set_xlabel("特征维度", fontsize=10)
    axes[0, 0].set_ylabel("特征值", fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(range(min(50, len(target_global))), target_global[:50], 
                   alpha=0.7, color='green', label='目标点云')
    axes[0, 1].set_title("目标点云全局特征 (前50维)", fontsize=12)
    axes[0, 1].set_xlabel("特征维度", fontsize=10)
    axes[0, 1].set_ylabel("特征值", fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 特征分布
    axes[1, 0].hist(source_global, bins=50, alpha=0.7, color='red', label=f'源 (均值: {np.mean(source_global):.3f})')
    axes[1, 0].hist(target_global, bins=50, alpha=0.7, color='green', label=f'目标 (均值: {np.mean(target_global):.3f})')
    axes[1, 0].set_title("全局特征分布", fontsize=12)
    axes[1, 0].set_xlabel("特征值", fontsize=10)
    axes[1, 0].set_ylabel("频数", fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 相似度
    axes[1, 1].bar([0], [global_similarity], color='blue', alpha=0.7)
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].set_xticks([])
    axes[1, 1].set_title(f"全局特征相似度: {global_similarity:.3f}", fontsize=12)
    axes[1, 1].set_ylabel("相似度", fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle("PointNet++ 全局特征分析", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'global_feature_analysis_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  📸 全局特征分析图已保存: global_feature_analysis_{timestamp}.png")

def print_registration_table(mean_distance, median_distance, std_distance, 
                           max_distance, min_distance, inlier_ratio, inlier_threshold,
                           global_similarity, matches_count, feature_dim, total_time):
    """打印配准结果表格"""
    print("\n📊 配准结果统计表:")
    print("+" + "-"*40 + "+" + "-"*20 + "+")
    print(f"| {'指标':<38} | {'数值':<18} |")
    print("+" + "-"*40 + "+" + "-"*20 + "+")
    print(f"| 全局特征相似度            | {global_similarity:>18.3f} |")
    print(f"| 特征匹配对数              | {matches_count:>18d} |")
    print(f"| 特征维度                  | {feature_dim:>18d} |")
    print(f"| 平均点对点距离            | {mean_distance:>18.6f} |")
    print(f"| 中位数距离                | {median_distance:>18.6f} |")
    print(f"| 标准差                    | {std_distance:>18.6f} |")
    print(f"| 最大距离                  | {max_distance:>18.6f} |")
    print(f"| 最小距离                  | {min_distance:>18.6f} |")
    print(f"| 内点比率 (<{inlier_threshold:.3f})   | {inlier_ratio:>16.3%} |")
    print(f"| 总处理时间 (秒)           | {total_time:>18.2f} |")
    print("+" + "-"*40 + "+" + "-"*20 + "+")

def evaluate_registration_comprehensive(source, target, transformation, voxel_size, 
                                      feature_dim=None, num_matches=None, 
                                      global_similarity=None, total_time=None):
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
    
    # 打印综合表格
    print_registration_table(mean_distance, median_distance, std_distance,
                           max_distance, min_distance, inlier_ratio, inlier_threshold,
                           global_similarity or 0, num_matches or 0, 
                           feature_dim or 0, total_time or 0)
    
    print(f"\n🎯 内点统计 (阈值: {inlier_threshold:.3f}):")
    print("+" + "-"*40 + "+" + "-"*20 + "+")
    print(f"| {'指标':<38} | {'数值':<18} |")
    print("+" + "-"*40 + "+" + "-"*20 + "+")
    print(f"| 内点总数                    | {np.sum(inliers):>18d} |")
    print(f"| 内点比率                    | {inlier_ratio:>16.3%} |")
    print(f"| 内点平均距离                | {inlier_mean:>18.6f} |")
    print(f"| 内点标准差                  | {inlier_std:>18.6f} |")
    print("+" + "-"*40 + "+" + "-"*20 + "+")
    
    print(f"\n📊 距离分布:")
    print("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*20 + "+")
    print(f"| {'距离区间':<18} | {'点数':<13} | {'比例':<18} |")
    print("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*20 + "+")
    for label, count, ratio in zip(bin_labels, hist, hist_ratio):
        print(f"| {label:<18} | {count:>13d} | {ratio:>16.3%} |")
    print("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*20 + "+")
    
    # 点云基本信息
    print(f"\n📦 点云基本信息:")
    print("+" + "-"*40 + "+" + "-"*20 + "+")
    print(f"| {'信息':<38} | {'数值':<18} |")
    print("+" + "-"*40 + "+" + "-"*20 + "+")
    print(f"| 源点云点数                  | {len(source.points):>18,d} |")
    print(f"| 目标点云点数                | {len(target.points):>18,d} |")
    print(f"| 体素大小                    | {voxel_size:>18.3f} |")
    print("+" + "-"*40 + "+" + "-"*20 + "+")
    
    return {
        'mean_distance': mean_distance,
        'median_distance': median_distance,
        'std_distance': std_distance,
        'inlier_ratio': inlier_ratio,
        'inlier_mean': inlier_mean,
        'inlier_count': np.sum(inliers),
        'global_similarity': global_similarity,
        'matches_count': num_matches,
        'feature_dim': feature_dim,
        'total_time': total_time,
        'distances': distances,
        'source_transformed': source_transformed
    }

# ==================== 6. 主函数 ====================
def main_improved():
    """改进版主函数"""
    print("="*60)
    print("PointNet++ 特征点云配准系统 (改进版)")
    print("全局特征辅助，优化大角度旋转配准")
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
    print("  [1] 0.005")
    print("  [2] 0.05 ")
    print("  [3] 0.1")
    
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
    
    # 运行改进版PointNet++配准
    print("\n🚀 开始改进版配准流程...")
    start_time = time.time()
    
    final_transform, global_similarity, matches_count = pointnet_feature_registration_improved(
        source, target, model, voxel_size, device
    )
    
    elapsed_time = time.time() - start_time
    
    # 重新提取全局特征用于可视化
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    
    source_tensor = prepare_pointcloud_for_pointnet(source_points, 1024, device)
    target_tensor = prepare_pointcloud_for_pointnet(target_points, 1024, device)
    
    with torch.no_grad():
        source_global = model.extract_descriptor(source_tensor).cpu().numpy().flatten()
        target_global = model.extract_descriptor(target_tensor).cpu().numpy().flatten()
        _, _, source_l1_feat, _ = model(source_tensor, return_features=True)
        source_feat_dim = source_l1_feat.shape[1]
    
    # 估算初始变换
    initial_transform = estimate_transform_from_global_features(
        source_global, target_global, source_points, target_points
    )
    
    # 可视化全局特征分析
    visualize_global_feature_analysis(
        source, target, source_down, target_down,
        source_global, target_global, global_similarity,
        initial_transform, voxel_size
    )
    
    # 综合评估
    evaluation = evaluate_registration_comprehensive(
        source, target, final_transform, voxel_size,
        feature_dim=source_feat_dim,
        num_matches=matches_count,
        global_similarity=global_similarity,
        total_time=elapsed_time
    )
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"pointnet_registration_improved_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存点云文件
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(final_transform)
    
    o3d.io.write_point_cloud(f"{output_dir}/source_original.ply", source)
    o3d.io.write_point_cloud(f"{output_dir}/target_original.ply", target)
    o3d.io.write_point_cloud(f"{output_dir}/source_registered.ply", source_transformed)
    
    # 保存详细结果报告
    with open(f"{output_dir}/registration_report_improved.txt", "w", encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("PointNet++ 改进版配准结果报告\n")
        f.write("="*60 + "\n\n")
        
        f.write("📁 文件信息:\n")
        f.write("-"*50 + "\n")
        f.write(f"源点云: {source_name}\n")
        f.write(f"目标点云: {target_name}\n")
        f.write(f"体素大小: {voxel_size:.3f}\n")
        f.write(f"全局特征相似度: {global_similarity:.3f}\n")
        f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {elapsed_time:.2f}秒\n\n")
        
        f.write("📈 配准结果:\n")
        f.write("-"*50 + "\n")
        f.write(f"内点比率:       {evaluation['inlier_ratio']:.3%}\n")
        f.write(f"平均距离:       {evaluation['mean_distance']:.6f}\n")
        f.write(f"中位数距离:     {evaluation['median_distance']:.6f}\n")
        f.write(f"特征维度:       {source_feat_dim}\n")
        f.write(f"特征匹配对数:   {matches_count:,}\n")
        f.write(f"内点数量:       {evaluation['inlier_count']:,}\n\n")
        
        f.write("⚡ 性能统计:\n")
        f.write("-"*50 + "\n")
        f.write(f"全局特征相似度: {global_similarity:.3f}\n")
        f.write(f"相似度评估: ")
        if global_similarity < 0.2:
            f.write("低 (可能是不同物体)\n")
        elif global_similarity < 0.4:
            f.write("中 (可能存在大角度旋转)\n")
        else:
            f.write("高 (预计配准效果良好)\n")
        f.write(f"内点比率:       {evaluation['inlier_ratio']:.3%}\n")
        f.write(f"内点比率评估: ")
        if evaluation['inlier_ratio'] < 0.3:
            f.write("差\n")
        elif evaluation['inlier_ratio'] < 0.6:
            f.write("一般\n")
        else:
            f.write("好\n")
        f.write("\n")
        
        f.write("🔢 变换矩阵 (4x4):\n")
        f.write("-"*50 + "\n")
        for i, row in enumerate(final_transform):
            f.write(f"[{row[0]:>10.6f} {row[1]:>10.6f} {row[2]:>10.6f} {row[3]:>10.6f}]\n")
        
        f.write(f"\n📁 生成文件:\n")
        f.write(f"-"*50 + "\n")
        f.write(f"1. {output_dir}/source_original.ply (原始源点云)\n")
        f.write(f"2. {output_dir}/target_original.ply (目标点云)\n")
        f.write(f"3. {output_dir}/source_registered.ply (配准后点云)\n")
        f.write(f"4. {output_dir}/registration_report_improved.txt (本报告)\n")
        f.write(f"5. global_feature_analysis_{timestamp}.png (全局特征分析图)\n")
    
    print(f"\n💾 结果已保存到目录: {output_dir}")
    
    # 可视化最终结果
    visualize_registration_comparison(
        source, target, source_transformed,
        "PointNet++ 改进版配准结果"
    )
    
    # 最终评估
    print("\n" + "="*60)
    print("🎯 最终配准评估")
    print("="*60)
    
    if evaluation['inlier_ratio'] > 0.7 and global_similarity > 0.5:
        print("✅ 配准非常成功！")
        print("   - 内点比率高")
        print("   - 全局特征相似度高")
        print("   - 点云对齐良好")
    elif evaluation['inlier_ratio'] > 0.4 and global_similarity > 0.3:
        print("⚠️  配准效果一般")
        print("   - 内点比率中等")
        print("   - 可能存在一些对齐误差")
    else:
        print("❌ 配准效果较差")
        print("   - 内点比率低")
        print("   - 可能需要调整参数或检查点云质量")
    
    print("\n" + "="*60)
    print("✅ PointNet++ 改进版特征配准完成!")
    print("="*60)
    print(f"📊 关键结果:")
    print(f"  - 全局特征相似度: {global_similarity:.3f}")
    print(f"  - 内点比率: {evaluation['inlier_ratio']:.3%}")
    print(f"  - 平均距离: {evaluation['mean_distance']:.6f}")
    print(f"  - 中位数距离: {evaluation['median_distance']:.6f}")
    print(f"  - 特征维度: {source_feat_dim}")
    print(f"  - 特征匹配对数: {matches_count:,}")
    print(f"  - 总处理时间: {elapsed_time:.2f}秒")
    print(f"  - 结果保存到: {output_dir}")
    print("="*60)

# ==================== 7. 原始版本（兼容性） ====================
def pointnet_feature_registration_original(source, target, model, voxel_size=0.05, device='cpu'):
    """原始版本PointNet++特征配准（用于对比）"""
    print("\n" + "="*60)
    print("原始版本PointNet++特征配准")
    print("="*60)
    
    start_time = time.time()
    
    # 下采样
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    # 提取特征
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    
    source_tensor = prepare_pointcloud_for_pointnet(source_points, 1024, device)
    target_tensor = prepare_pointcloud_for_pointnet(target_points, 1024, device)
    
    with torch.no_grad():
        # 只提取局部特征
        _, _, source_l1_feat, _ = model(source_tensor, return_features=True)
        _, _, target_l1_feat, _ = model(target_tensor, return_features=True)
        
        source_l1_feat = source_l1_feat.cpu().numpy()[0]
        target_l1_feat = target_l1_feat.cpu().numpy()[0]
    
    # 特征匹配
    matches = []
    n_source = min(len(source_points), len(source_l1_feat.T))
    n_target = min(len(target_points), len(target_l1_feat.T))
    
    if n_source > 0 and n_target > 0:
        source_features = source_l1_feat.T[:n_source]
        target_features = target_l1_feat.T[:n_target]
        
        source_norm = source_features / (np.linalg.norm(source_features, axis=1, keepdims=True) + 1e-8)
        target_norm = target_features / (np.linalg.norm(target_features, axis=1, keepdims=True) + 1e-8)
        
        similarity = np.dot(source_norm, target_norm.T)
        
        for i in range(n_source):
            target_indices = np.argsort(similarity[i])[-3:][::-1]
            for j in target_indices:
                if j < n_target:
                    score = similarity[i, j]
                    if score > 0.5:
                        matches.append([i, j, score])
    
    matches = np.array(matches) if matches else np.array([])
    
    # RANSAC
    if len(matches) >= 3:
        best_inliers = 0
        best_transform = np.eye(4)
        
        for _ in range(1000):
            sample_idx = np.random.choice(len(matches), 3, replace=False)
            sample = matches[sample_idx]
            
            src_pts = source_points[sample[:, 0].astype(int)]
            tgt_pts = target_points[sample[:, 1].astype(int)]
            
            R, t = compute_rigid_transform(src_pts, tgt_pts)
            
            if R is not None:
                transformed_src = np.dot(source_points[matches[:, 0].astype(int)], R.T) + t
                distances = np.linalg.norm(transformed_src - target_points[matches[:, 1].astype(int)], axis=1)
                inliers = np.sum(distances < voxel_size*2)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_transform = np.eye(4)
                    best_transform[:3, :3] = R
                    best_transform[:3, 3] = t
        
        initial_transform = best_transform
    else:
        initial_transform = np.eye(4)
    
    # ICP
    final_transform = icp_refinement(source, target, initial_transform, voxel_size)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  总耗时: {elapsed_time:.2f}秒")
    
    return final_transform

# ==================== 8. 程序入口 ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PointNet++ 点云配准系统")
    print("="*60)
    
    print("     - 使用全局特征辅助")
    print("     - 优化大角度旋转配准")
    print("     - 包含详细分析和可视化")
    
    main_improved()