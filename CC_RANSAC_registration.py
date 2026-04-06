"""这里实现基于一点采样一致性点云配准算法，将传统RANSAC需要3点来确定变换矩阵优化到只需要1点即可实现"""
import numpy as np
import open3d as o3d
import time
import copy
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
from scipy.spatial import cKDTree

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
print("✅ 已设置中文字体支持")

class CCRANSAC:
    """基于一点采样一致性的点云配准算法 (CC-RANSAC)"""
    
    def __init__(self, params=None):
        """初始化参数"""
        if params is None:
            params = {}
        
        # 默认参数
        self.params = {
            'voxel_size': 0.005,  # 体素大小
            'radius_normal': 0.1,  # 法线估计半径
            'radius_feature': 0.25,  # 特征计算半径
            'k_normals': 30,  # 法线估计的最近邻数
            'k_features': 100,  # 特征计算的最近邻数
            'distance_threshold': 0.05,  # 距离约束阈值系数
            'angle_threshold': 30,  # 角度约束阈值(度)
            'inner_threshold': 0.1,  # 内点阈值
            'extend_ratio': 1.5,  # 局部轴延伸倍数
            'stop_threshold': 0.5,  # 提前终止阈值
            'validate_threshold': 0.3,  # 验证模式切换阈值
            'dynamic_radius': True,  # 是否使用动态半径
            'min_radius': 0.1,  # 最小支撑半径
            'gaussian_sigma': 0.1,  # 高斯权重sigma
            'curvature_beta': 10,  # 曲率平滑参数
        }
        self.params.update(params)
        
        print("CC-RANSAC 算法初始化完成")
        print("参数配置:")
        for key, value in self.params.items():
            print(f"  {key}: {value}")
    
    def load_point_clouds(self):
        """加载点云文件"""
        print("搜索点云文件...")
        
        ply_files = glob.glob("./data/*.ply")
        
        if not ply_files:
            print("❌ 未找到点云文件")
            return None, None
        
        print(f"找到 {len(ply_files)} 个点云文件:")
        for i, file in enumerate(ply_files, 1):
            basename = os.path.basename(file)
            print(f"  [{i}] {basename}")
        
        # 让用户选择两个文件
        print("\n请选择两个点云文件:")
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
        
        print(f"\n加载: {os.path.basename(file1)} 和 {os.path.basename(file2)}")
        
        source = o3d.io.read_point_cloud(file1)
        target = o3d.io.read_point_cloud(file2)
        
        if len(source.points) == 0 or len(target.points) == 0:
            print("❌ 加载失败，创建测试数据")
            return self.create_test_data()
        
        print(f"源点云点数: {len(source.points):,}")
        print(f"目标点云点数: {len(target.points):,}")
        
        return source, target, os.path.basename(file1), os.path.basename(file2)
    
    def create_test_data(self):
        """创建测试数据"""
        print("创建测试点云...")
        
        # 创建复杂形状
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        mesh = mesh.subdivide_midpoint(number_of_iterations=2)
        mesh.compute_vertex_normals()
        
        # 源点云
        source = mesh.sample_points_poisson_disk(number_of_points=5000)
        
        # 创建大角度变换
        angle = np.radians(60)  # 60度大旋转
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0.1],
            [np.sin(angle), np.cos(angle), -0.1],
            [-0.1, 0.2, 0.9]
        ])
        t = np.array([0.3, 0.2, 0.1])
        
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t
        
        # 目标点云
        target = copy.deepcopy(source)
        target.transform(transformation)
        
        # 添加噪声
        points = np.asarray(target.points)
        noise = np.random.normal(0, 0.01, points.shape)
        target.points = o3d.utility.Vector3dVector(points + noise)
        
        source.estimate_normals()
        target.estimate_normals()
        
        return source, target, "source_test", "target_test"
    
    def preprocess_point_cloud(self, pcd, voxel_size=None):
        """预处理点云"""
        if voxel_size is None:
            voxel_size = self.params['voxel_size']
        
        print(f"  下采样 (体素大小: {voxel_size})...")
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        radius_normal = self.params['radius_normal']
        print(f"  估计法线 (搜索半径: {radius_normal})...")
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, 
                                               max_nn=self.params['k_normals']))
        
        return pcd_down
    
    def improved_compute_lra(self, points, normals, point_indices=None):
        """改进的LRA计算"""
        n_points = len(points)
        
        if point_indices is None:
            point_indices = range(n_points)
        
        lra_axes = []
        kdtree = cKDTree(points)  # 使用cKDTree加速
        
        for idx in point_indices:
            p = points[idx]
            
            # 1. 动态半径调整
            distances, _ = kdtree.query(p.reshape(1, -1), k=50)
            d_50 = distances[0, -1]
            r_adaptive = max(d_50, self.params['min_radius'])
            
            # 2. 找到支撑域内的点
            neighbor_indices = kdtree.query_ball_point(p, r=r_adaptive)
            
            if len(neighbor_indices) < 5:  # 至少需要5个点
                lra_axes.append(np.eye(3))  # 返回单位矩阵
                continue
            
            neighbor_points = points[neighbor_indices]
            
            # 3. 加权协方差计算
            diffs = neighbor_points - p
            distances = np.linalg.norm(diffs, axis=1)
            
            sigma = r_adaptive / 3
            weights = np.exp(-distances**2 / (2 * sigma**2))
            weights = weights / (np.sum(weights) + 1e-8)
            
            # 加权协方差矩阵
            weighted_cov = np.zeros((3, 3))
            for k in range(len(neighbor_points)):
                diff = diffs[k]
                weighted_cov += weights[k] * np.outer(diff, diff)
            
            # 4. 正则化SVD
            lambda_reg = 1e-6 * np.trace(weighted_cov)
            C_reg = weighted_cov + lambda_reg * np.eye(3)
            
            # 特征值分解
            eigenvalues, eigenvectors = np.linalg.eigh(C_reg)
            idx_sorted = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx_sorted]
            
            # 5. 法向计算
            v3 = eigenvectors[:, -1]
            
            # 曲率计算
            lambda_min = eigenvalues[-1]
            lambda_sum = np.sum(eigenvalues)
            curvature = lambda_min / (lambda_sum + 1e-8)
            
            # 曲率消歧
            beta = 10
            sgn_value = 2 / (1 + np.exp(-beta * curvature)) - 1
            
            normal = sgn_value * v3
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            
            # 6. 构建完整的局部参考系
            v1 = eigenvectors[:, 0]  # 最大特征值方向
            v2 = np.cross(normal, v1)  # 叉积得到y轴
            
            # 存储完整的局部坐标系
            lra_frame = np.column_stack([v1, v2, normal])
            lra_axes.append(lra_frame)
        
        # 确保返回 numpy 数组
        return np.array(lra_axes)
    
    def compute_compatibility_matrix(self, source_points, target_points, 
                                source_lra, target_lra, keypoint_pairs):
        """
        计算兼容性矩阵
        基于距离约束和角度约束 (论文公式11-15)
        """
        n_pairs = len(keypoint_pairs)
        
        if n_pairs == 0:
            return None
        
        # 提取法线方向（LRA的第三列，即z轴）
        if source_lra.ndim == 3:
            source_normals = source_lra[:, :, 2]  # 形状从 (n, 3, 3) 变为 (n, 3)
        else:
            source_normals = source_lra
        
        if target_lra.ndim == 3:
            target_normals = target_lra[:, :, 2]  # 形状从 (n, 3, 3) 变为 (n, 3)
        else:
            target_normals = target_lra
        
        # 初始化兼容性矩阵
        compatibility_matrix = np.zeros((n_pairs, n_pairs), dtype=bool)
        
        # 计算点云分辨率 (用于动态阈值)
        all_points = np.vstack([source_points, target_points])
        tree = KDTree(all_points)
        distances, _ = tree.query(all_points, k=2)
        mr = np.mean(distances[:, 1])  # 平均最近邻距离
        
        print(f"  点云分辨率: {mr:.6f}")
        
        for i in range(n_pairs):
            idx_i_src, idx_i_tgt = keypoint_pairs[i]
            
            for j in range(i+1, n_pairs):
                idx_j_src, idx_j_tgt = keypoint_pairs[j]
                
                # 1. 距离约束 (论文公式11-13)
                # 计算源点云中的距离
                d_src = np.linalg.norm(source_points[idx_i_src] - source_points[idx_j_src])
                d_tgt = np.linalg.norm(target_points[idx_i_tgt] - target_points[idx_j_tgt])
                
                # 距离误差
                distance_error = abs(d_src - d_tgt)
                
                # 动态距离阈值
                d_ij = (d_src + d_tgt) / 2
                a, b, m, n = 0.5, 2.0, 2.0, 5.0  # 论文中的先验参数
                
                if d_ij < a * mr:
                    tau_distance = 0
                elif a * mr <= d_ij <= b * mr:
                    tau_distance = (d_ij / n) * mr
                else:
                    tau_distance = m * mr
                
                distance_compatible = distance_error < tau_distance
                
                # 2. 角度约束 (论文公式14-15)
                # 计算源点云中的角度
                angle_src = np.arccos(np.clip(
                    np.dot(source_normals[idx_i_src], source_normals[idx_j_src]), -1.0, 1.0
                ))
                angle_tgt = np.arccos(np.clip(
                    np.dot(target_normals[idx_i_tgt], target_normals[idx_j_tgt]), -1.0, 1.0
                ))
                
                # 角度误差 (转换为度)
                angle_error = np.degrees(abs(angle_src - angle_tgt))
                
                angle_compatible = angle_error < self.params['angle_threshold']
                
                # 3. 组合约束
                if distance_compatible and angle_compatible:
                    compatibility_matrix[i, j] = True
                    compatibility_matrix[j, i] = True
        
        return compatibility_matrix, mr

    def find_connected_components(self, compatibility_matrix):
        """找到兼容性矩阵中的连通组件"""
        if compatibility_matrix is None or compatibility_matrix.shape[0] == 0:
            return []
        
        # 确保矩阵是稀疏格式
        if not isinstance(compatibility_matrix, csr_matrix):
            compatibility_matrix = csr_matrix(compatibility_matrix)
        
        # 使用 scipy 的连通组件算法
        n_components, labels = connected_components(
            csgraph=compatibility_matrix, 
            directed=False, 
            return_labels=True
        )
        
        # 将标签分组
        components = []
        for i in range(n_components):
            component_indices = np.where(labels == i)[0]
            if len(component_indices) > 0:
                components.append(component_indices.tolist())
        
        # 按大小排序
        components.sort(key=len, reverse=True)
        
        return components

    def compute_point_resolution(self, points):
        """计算点云分辨率"""
        if len(points) < 2:
            return 0.01
        
        kdtree = KDTree(points)
        distances, _ = kdtree.query(points, k=2)
        mr = np.mean(distances[:, 1])
        return mr

    def compute_dynamic_threshold(self, d_ij, mr):
        """计算动态阈值"""
        a, b, m, n = 0.5, 2.0, 2.0, 5.0
        
        if d_ij < a * mr:
            return 0
        elif a * mr <= d_ij <= b * mr:
            return (d_ij / n) * mr
        else:
            return m * mr
    
    def compute_rodrigues_rotation(self, axis, angle):
        """
        计算罗德里格斯旋转矩阵
        论文中用于对齐局部轴
        """
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        return R
    
    def align_points_with_single_pair_simple(self, source_point, target_point, 
                                            source_lra, target_lra):
        """
        简化的基于单点对配准
        使用SVD计算变换矩阵
        """
        # 提取法线（如果传递的是完整的LRA）
        if source_lra.shape == (3, 3):  # 完整的局部坐标系
            source_normal = source_lra[:, 2]  # 第三列是法线
        else:  # 已经是法线向量
            source_normal = source_lra
        
        if target_lra.shape == (3, 3):  # 完整的局部坐标系
            target_normal = target_lra[:, 2]  # 第三列是法线
        else:  # 已经是法线向量
            target_normal = target_lra
        
        # 创建点对：源点 + 沿法线延伸的点
        extend_distance = 0.1  # 固定延伸距离
        
        src_points = np.array([
            source_point,
            source_point + source_normal * extend_distance
        ])
        
        tgt_points = np.array([
            target_point,
            target_point + target_normal * extend_distance
        ])
        
        # 中心化
        src_centroid = np.mean(src_points, axis=0)
        tgt_centroid = np.mean(tgt_points, axis=0)
        
        src_centered = src_points - src_centroid
        tgt_centered = tgt_points - tgt_centroid
        
        # 计算旋转矩阵
        H = np.dot(src_centered.T, tgt_centered)
        U, S, Vt = np.linalg.svd(H)
        
        R = np.dot(Vt.T, U.T)
        
        # 确保是右手坐标系
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # 计算平移
        t = tgt_centroid - np.dot(R, src_centroid)
        
        # 构建变换矩阵
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t
        
        return transformation
    
    def verify_hypothesis_improved(self, transformation, source_points, target_points, 
                                 source_keypoints, target_keypoints, mode='keypoints'):
        """改进的假设验证"""
        if mode == 'keypoints':
            # 变换目标关键点到源坐标系
            target_transformed = np.dot(target_keypoints, transformation[:3, :3].T) + transformation[:3, 3]
            
            # 计算双向距离
            tree_source = KDTree(source_keypoints)
            tree_target = KDTree(target_transformed)
            
            # 源到目标的距离
            dist_s2t, _ = tree_target.query(source_keypoints, k=1)
            # 目标到源的距离
            dist_t2s, _ = tree_source.query(target_transformed, k=1)
            
            # 组合距离（取最大值）
            distances = np.maximum(dist_s2t.squeeze(), dist_t2s.squeeze())
            
            # 内点阈值
            threshold = self.params['inner_threshold'] * 2  # 稍微放宽阈值
            
            inliers = distances < threshold
            num_inliers = np.sum(inliers)
            
            return num_inliers, inliers
        else:
            # 完整点云验证
            target_transformed = np.dot(target_points, transformation[:3, :3].T) + transformation[:3, 3]
            
            tree = KDTree(source_points)
            distances, _ = tree.query(target_transformed, k=1)
            
            threshold = self.params['voxel_size'] * 3
            inliers = distances < threshold
            num_inliers = np.sum(inliers)
            
            return num_inliers, inliers
    
    def extract_keypoints_fpfh(self, pcd, voxel_size=None):
        """使用FPFH特征提取关键点"""
        if voxel_size is None:
            voxel_size = self.params['voxel_size']
        
        # 下采样
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # 估计法线
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        # 计算FPFH特征
        radius_feature = voxel_size * 5
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        
        return pcd_down, fpfh
    
    def match_keypoints_fpfh_improved(self, source_fpfh, target_fpfh, num_matches=1000):
        """改进的FPFH特征匹配"""
        # 转置为(N, 33)
        source_features = source_fpfh.data.T
        target_features = target_fpfh.data.T
        
        # 归一化
        source_norm = source_features / (np.linalg.norm(source_features, axis=1, keepdims=True) + 1e-8)
        target_norm = target_features / (np.linalg.norm(target_features, axis=1, keepdims=True) + 1e-8)
        
        # 计算相似度矩阵
        similarity = np.dot(source_norm, target_norm.T)
        
        # 找到每个源点的最佳匹配
        matches = []
        n_source = min(len(source_features), 200)  # 增加源点数量
        
        for i in range(n_source):
            # 找到最相似的3个候选
            best_idxs = np.argsort(similarity[i])[-3:][::-1]
            
            for best_idx in best_idxs:
                score = similarity[i, best_idx]
                
                # 降低相似度阈值，增加匹配数量
                if score > 0.3:  # 从0.5降低到0.3
                    # 检查是否是双向一致的最佳匹配
                    is_mutual_best = (np.argmax(similarity[:, best_idx]) == i)
                    
                    if is_mutual_best or score > 0.7:  # 高相似度或双向一致
                        matches.append((i, best_idx, score))
                        break  # 每个源点只保留一个匹配
                    elif score > 0.5:  # 中等相似度
                        matches.append((i, best_idx, score))
                        break
        
        # 按相似度排序
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # 限制匹配数量
        matches = matches[:num_matches]
        
        print(f"    找到匹配: {len(matches)} 个")
        if len(matches) > 0:
            print(f"    最高相似度: {matches[0][2]:.3f}, 最低: {matches[-1][2]:.3f}")
        
        return matches
    
    def verify_matched_pairs(self, transformation, source_keypoints, target_keypoints, keypoint_pairs):
        """只验证匹配的点对"""
        n_pairs = len(keypoint_pairs)
        
        if n_pairs == 0:
            return 0, np.array([], dtype=bool)
        
        distances = []
        
        for i, (src_idx, tgt_idx) in enumerate(keypoint_pairs):
            if src_idx < len(source_keypoints) and tgt_idx < len(target_keypoints):
                src_point = source_keypoints[src_idx]
                tgt_point = target_keypoints[tgt_idx]
                
                # 变换目标点到源坐标系
                tgt_transformed = np.dot(tgt_point, transformation[:3, :3].T) + transformation[:3, 3]
                
                # 计算距离
                distance = np.linalg.norm(src_point - tgt_transformed)
                distances.append(distance)
            else:
                distances.append(float('inf'))
        
        distances = np.array(distances)
        
        # 内点阈值
        threshold = self.params['inner_threshold'] * 2
        
        inliers = distances < threshold
        num_inliers = np.sum(inliers)
        
        return num_inliers, inliers

    def cc_ransac_registration(self, source, target, visualize=True):
        """
        改进的CC-RANSAC主算法
        基于论文的完整配准流程
        """
        print("\n" + "="*60)
        print("改进的 CC-RANSAC 点云配准算法")
        print("="*60)
        
        start_time = time.time()
        
        # 1. 预处理
        print("\n[1] 预处理点云")
        source_down = self.preprocess_point_cloud(source)
        target_down = self.preprocess_point_cloud(target)
        
        source_points = np.asarray(source_down.points)
        target_points = np.asarray(target_down.points)
        source_normals = np.asarray(source_down.normals)
        target_normals = np.asarray(target_down.normals)
        
        print(f"   源点云: {len(source.points):,} -> {len(source_points):,}")
        print(f"   目标点云: {len(target.points):,} -> {len(target_points):,}")
        
        # 2. 提取关键点
        print("\n[2] 提取关键点")
        source_keypoints, source_fpfh = self.extract_keypoints_fpfh(source_down)
        target_keypoints, target_fpfh = self.extract_keypoints_fpfh(target_down)
        
        source_keypoints_np = np.asarray(source_keypoints.points)
        target_keypoints_np = np.asarray(target_keypoints.points)
        
        print(f"   源关键点: {len(source_keypoints_np):,}")
        print(f"   目标关键点: {len(target_keypoints_np):,}")
        
        # 3. 特征匹配
        print("\n[3] 特征匹配")
        fpfh_matches = self.match_keypoints_fpfh_improved(source_fpfh, target_fpfh, num_matches=1000)
        
        if len(fpfh_matches) < 20:
            print("❌ 匹配对太少，尝试降低阈值")
            # 如果匹配太少，使用所有可能的匹配
            keypoint_pairs = []
            n_src = min(100, len(source_keypoints_np))
            n_tgt = min(100, len(target_keypoints_np))
            for i in range(n_src):
                for j in range(n_tgt):
                    keypoint_pairs.append((i, j))
            keypoint_pairs = keypoint_pairs[:200]  # 限制数量
        else:
            # 提取匹配对索引
            keypoint_pairs = [(match[0], match[1]) for match in fpfh_matches[:200]]  # 取前200个
        
        print(f"   初始匹配对: {len(keypoint_pairs)}")
        
        if len(keypoint_pairs) < 10:
            print("❌ 匹配对太少，无法进行配准")
            return np.eye(4), {'inlier_ratio': 0, 'mean_distance': 999, 'median_distance': 999}
        
        # 4. 计算局部参考轴
        print("\n[4] 计算局部参考轴 (LRA)")
        
        # 为关键点计算LRA
        source_lra = self.improved_compute_lra(source_keypoints_np, 
                                            np.asarray(source_keypoints.normals))
        target_lra = self.improved_compute_lra(target_keypoints_np,
                                            np.asarray(target_keypoints.normals))
        
        print(f"   源LRA维度: {source_lra.shape}")
        print(f"   目标LRA维度: {target_lra.shape}")
        
        # 5. 计算兼容性矩阵
        print("\n[5] 计算兼容性矩阵")
        compatibility_matrix, mr = self.compute_compatibility_matrix(
            source_keypoints_np, target_keypoints_np,
            source_lra, target_lra, keypoint_pairs
        )
        
        if compatibility_matrix is None or np.sum(compatibility_matrix) < 10:
            print("❌ 兼容性矩阵太稀疏，使用所有点对")
            # 如果兼容性矩阵太稀疏，使用所有点对
            n_pairs = len(keypoint_pairs)
            compatibility_matrix = np.ones((n_pairs, n_pairs), dtype=bool)
            np.fill_diagonal(compatibility_matrix, False)
        
        print(f"   兼容性矩阵密度: {np.sum(compatibility_matrix) / (len(keypoint_pairs)**2):.3%}")
        
        # 6. 连通性分析
        print("\n[6] 连通性分析")
        components = self.find_connected_components(compatibility_matrix)
        
        print(f"   找到连通组件: {len(components)}")
        for i, comp in enumerate(components[:5]):
            print(f"     组件{i}: {len(comp)}个点对")
        
        if len(components) == 0:
            print("❌ 未找到连通组件，使用所有点对作为一个组件")
            components = [list(range(len(keypoint_pairs)))]
        
        # 7. 假设生成与验证
        print("\n[7] 假设生成与验证")
        best_transform = np.eye(4)
        best_inliers = 0
        best_inlier_indices = None
        
        # 增加采样次数
        max_iterations = min(50, len(keypoint_pairs))
        
        for iteration in range(max_iterations):
            # 随机选择一个连通组件
            if len(components) > 0:
                comp_idx = iteration % len(components)
                component = components[comp_idx]
            else:
                component = list(range(len(keypoint_pairs)))
            
            if len(component) < 1:
                continue
            
            # 随机选择组件中的一个点对
            rand_idx = np.random.choice(component)
            pair_idx_src, pair_idx_tgt = keypoint_pairs[rand_idx]
            
            # 获取对应的点和LRA
            src_point = source_keypoints_np[pair_idx_src]
            tgt_point = target_keypoints_np[pair_idx_tgt]
            src_lra = source_lra[pair_idx_src]
            tgt_lra = target_lra[pair_idx_tgt]
            
            # 基于单点对计算变换
            transform = self.align_points_with_single_pair_simple(
                src_point, tgt_point, src_lra, tgt_lra
            )
            
            # 验证假设
            num_inliers, inlier_mask = self.verify_matched_pairs(
                transform, source_keypoints_np, target_keypoints_np, keypoint_pairs
            )
            
            if iteration % 10 == 0:
                print(f"   迭代 {iteration}: 内点数 {num_inliers}/{len(keypoint_pairs)}")
            
            # 更新最佳变换
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_transform = transform
                best_inlier_indices = np.where(inlier_mask)[0]
                
                inlier_ratio = num_inliers / len(keypoint_pairs)
                if inlier_ratio > 0.7:  # 高内点率，提前终止
                    print(f"     ✅ 找到高质量变换 ({inlier_ratio:.2%})，提前终止")
                    break
        
        print(f"   最佳内点数: {best_inliers}/{len(keypoint_pairs)} ({best_inliers/len(keypoint_pairs):.2%})")
        
        # 8. 使用最佳匹配点对进行SVD精配准
        if best_inliers > 3 and best_inlier_indices is not None:
            print("\n[8] 使用内点进行SVD精配准")
            
            # 收集内点点对
            inlier_pairs = [keypoint_pairs[i] for i in best_inlier_indices[:min(50, len(best_inlier_indices))]]
            
            src_points = np.array([source_keypoints_np[i] for i, _ in inlier_pairs])
            tgt_points = np.array([target_keypoints_np[j] for _, j in inlier_pairs])
            
            # 中心化
            src_centroid = np.mean(src_points, axis=0)
            tgt_centroid = np.mean(tgt_points, axis=0)
            
            src_centered = src_points - src_centroid
            tgt_centered = tgt_points - tgt_centroid
            
            # 计算旋转矩阵
            H = np.dot(src_centered.T, tgt_centered)
            U, S, Vt = np.linalg.svd(H)
            
            R = np.dot(Vt.T, U.T)
            
            # 确保是右手坐标系
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)
            
            # 计算平移
            t = tgt_centroid - np.dot(R, src_centroid)
            
            # 更新变换矩阵
            best_transform = np.eye(4)
            best_transform[:3, :3] = R
            best_transform[:3, 3] = t
        
        # 9. ICP精配准
        print("\n[9] ICP精配准")
        distance_threshold = self.params['voxel_size'] * 3
        
        try:
            reg_result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, distance_threshold, best_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=50,
                    relative_fitness=1e-6,
                    relative_rmse=1e-6
                )
            )
            
            final_transform = reg_result.transformation
            print(f"   ICP适应度: {reg_result.fitness:.4f}")
            print(f"   ICP RMSE: {reg_result.inlier_rmse:.6f}")
        except Exception as e:
            print(f"   ICP失败: {e}，使用原始变换")
            final_transform = best_transform
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  总耗时: {elapsed_time:.2f}秒")
        print(f"   RANSAC内点数: {best_inliers}")
        
        # 10. 评估结果
        print("\n[10] 评估配准结果")
        evaluation = self.evaluate_registration(
            source, target, final_transform, self.params['voxel_size']
        )
        
        # 可视化
        if visualize:
            self.visualize_results(
                source, target, source_down, target_down,
                source_keypoints, target_keypoints,
                keypoint_pairs, best_inlier_indices,
                final_transform, compatibility_matrix, components
            )
        
        return final_transform, evaluation
    
    def evaluate_registration(self, source, target, transformation, voxel_size):
        """评估配准结果"""
        source_transformed = copy.deepcopy(source)
        source_transformed.transform(transformation)
        
        source_points = np.asarray(source_transformed.points)
        target_points = np.asarray(target.points)
        
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_points)
        distances, _ = nbrs.kneighbors(source_points)
        
        # 计算统计信息
        mean_distance = np.mean(distances)
        median_distance = np.median(distances)
        max_distance = np.max(distances)
        min_distance = np.min(distances)
        std_distance = np.std(distances)
        
        # 内点比率
        inlier_threshold = voxel_size * 2
        inlier_ratio = np.sum(distances < inlier_threshold) / len(distances)
        
        print("\n" + "="*50)
        print("配准结果评估")
        print("="*50)
        print(f"平均点对点距离: {mean_distance:.6f}")
        print(f"中位数距离:    {median_distance:.6f}")
        print(f"最大距离:      {max_distance:.6f}")
        print(f"最小距离:      {min_distance:.6f}")
        print(f"标准差:        {std_distance:.6f}")
        print(f"内点比率 (<{inlier_threshold:.3f}): {inlier_ratio:.3%}")
        
        return {
            'mean_distance': mean_distance,
            'median_distance': median_distance,
            'inlier_ratio': inlier_ratio,
            'source_transformed': source_transformed
        }
    
    def visualize_results(self, source, target, source_down, target_down,
                         source_keypoints, target_keypoints,
                         keypoint_pairs, inlier_indices,
                         transformation, compatibility_matrix, components):
        """可视化结果"""
        print("\n👀 可视化配准结果...")
        
        # 1. 原始点云
        print("\n[1] 原始点云")
        source_vis = copy.deepcopy(source_down)
        target_vis = copy.deepcopy(target_down)
        
        source_vis.paint_uniform_color([1, 0, 0])  # 红色
        target_vis.paint_uniform_color([0, 1, 0])  # 绿色
        
        o3d.visualization.draw_geometries(
            [source_vis, target_vis],
            window_name="1. 原始点云 (红:源, 绿:目标)",
            width=1000, height=800
        )
        
        # 2. 关键点和匹配
        print("\n[2] 关键点匹配")
        # 创建匹配线
        lines = []
        colors = []
        
        source_kp_np = np.asarray(source_keypoints.points)
        target_kp_np = np.asarray(target_keypoints.points)
        
        # 显示前50个匹配
        n_matches_to_show = min(50, len(keypoint_pairs))
        
        line_points = []
        for i in range(n_matches_to_show):
            src_idx, tgt_idx = keypoint_pairs[i]
            
            if src_idx < len(source_kp_np) and tgt_idx < len(target_kp_np):
                lines.append([i*2, i*2+1])
                
                # 内点用绿色，外点用红色
                if inlier_indices is not None and i in inlier_indices:
                    color = [0, 1, 0]  # 绿色
                else:
                    color = [1, 0, 0]  # 红色
                
                colors.append(color)
                
                if i == 0:
                    line_points = np.array([source_kp_np[src_idx], target_kp_np[tgt_idx]])
                else:
                    line_points = np.vstack([line_points, 
                                           [source_kp_np[src_idx], target_kp_np[tgt_idx]]])
        
        if len(lines) > 0:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            source_kp_vis = copy.deepcopy(source_keypoints)
            target_kp_vis = copy.deepcopy(target_keypoints)
            
            source_kp_vis.paint_uniform_color([1, 0, 0])
            target_kp_vis.paint_uniform_color([0, 1, 0])
            
            o3d.visualization.draw_geometries(
                [source_kp_vis, target_kp_vis, line_set],
                window_name=f"2. 关键点匹配 (显示{n_matches_to_show}个匹配)",
                width=1000, height=800
            )
        
        # 3. 兼容性矩阵可视化
        print("\n[3] 兼容性矩阵")
        if compatibility_matrix is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(compatibility_matrix, cmap='Blues', aspect='auto')
            plt.colorbar(label='兼容性')
            plt.title('兼容性矩阵', fontsize=16)
            plt.xlabel('匹配对索引', fontsize=12)
            plt.ylabel('匹配对索引', fontsize=12)
            
            # 标记连通组件
            for comp in components[:5]:  # 只标记前5个组件
                if len(comp) > 1:
                    # 在矩阵上画框
                    min_idx = min(comp)
                    max_idx = max(comp)
                    rect = plt.Rectangle((min_idx-0.5, min_idx-0.5), 
                                       max_idx-min_idx+1, max_idx-min_idx+1,
                                       fill=False, edgecolor='red', linewidth=2)
                    plt.gca().add_patch(rect)
            
            plt.tight_layout()
            plt.savefig('compatibility_matrix.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("  📸 兼容性矩阵已保存: compatibility_matrix.png")
        
        # 4. 配准结果
        print("\n[4] 配准结果")
        source_registered = copy.deepcopy(source_down)
        source_registered.transform(transformation)
        source_registered.paint_uniform_color([0, 0, 1])  # 蓝色
        
        target_vis = copy.deepcopy(target_down)
        target_vis.paint_uniform_color([0, 1, 0])  # 绿色
        
        o3d.visualization.draw_geometries(
            [source_registered, target_vis],
            window_name="4. 配准结果 (蓝:源, 绿:目标)",
            width=1000, height=800
        )
        
        # 5. 完整点云配准
        print("\n[5] 完整点云配准")
        source_full = copy.deepcopy(source)
        source_full.transform(transformation)
        
        source_full.paint_uniform_color([1, 0, 0])  # 红色
        target_full = copy.deepcopy(target)
        target_full.paint_uniform_color([0, 1, 0])  # 绿色
        
        o3d.visualization.draw_geometries(
            [source_full, target_full],
            window_name="5. 完整点云配准结果",
            width=1000, height=800
        )
    
    def save_results(self, source, target, transformation, evaluation, 
                    source_name, target_name, voxel_size):
        """保存结果"""
        print("\n💾 保存配准结果...")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"cc_ransac_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存点云
        source_transformed = evaluation['source_transformed']
        
        o3d.io.write_point_cloud(f"{output_dir}/source_original.ply", source)
        o3d.io.write_point_cloud(f"{output_dir}/target_original.ply", target)
        o3d.io.write_point_cloud(f"{output_dir}/source_registered.ply", source_transformed)
        
        # 保存变换矩阵
        with open(f"{output_dir}/transformation.txt", "w", encoding='utf-8') as f:
            f.write("CC-RANSAC 配准变换矩阵\n")
            f.write("="*50 + "\n\n")
            
            f.write("📁 文件信息:\n")
            f.write("-"*40 + "\n")
            f.write(f"源点云: {source_name}\n")
            f.write(f"目标点云: {target_name}\n")
            f.write(f"体素大小: {voxel_size:.3f}\n")
            f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("📈 配准结果:\n")
            f.write("-"*40 + "\n")
            f.write(f"平均距离: {evaluation['mean_distance']:.6f}\n")
            f.write(f"中位数距离: {evaluation['median_distance']:.6f}\n")
            f.write(f"内点比率: {evaluation['inlier_ratio']:.3%}\n\n")
            
            f.write("🔢 变换矩阵 (4x4):\n")
            f.write("-"*40 + "\n")
            for i, row in enumerate(transformation):
                f.write(f"[{row[0]:>10.6f} {row[1]:>10.6f} {row[2]:>10.6f} {row[3]:>10.6f}]\n")
        
        print(f"结果已保存到目录: {output_dir}")
        return output_dir

def main():
    """主函数"""
    print("="*60)
    print("CC-RANSAC 点云配准系统")
    print("基于一点采样一致性和连通性分析")
    print("="*60)
    
    # 创建算法实例
    algorithm = CCRANSAC()
    
    # 加载点云
    source, target, source_name, target_name = algorithm.load_point_clouds()
    
    if source is None or target is None:
        print("❌ 无法加载点云，退出程序")
        return
    
    print(f"\n处理文件:")
    print(f"  源点云: {source_name}")
    print(f"  目标点云: {target_name}")
    
    # 让用户选择体素大小
    print("\n选择体素大小 (下采样参数):")
    print("  [1] 0.005 ")
    print("  [2] 0.05 ")
    print("  [3] 0.1 ")
    
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
    
    algorithm.params['voxel_size'] = voxel_size
    print(f"\n🔧 使用体素大小: {voxel_size}")
    
    # 运行CC-RANSAC配准
    final_transform, evaluation = algorithm.cc_ransac_registration(
        source, target, visualize=True
    )
    
    # 保存结果
    output_dir = algorithm.save_results(
        source, target, final_transform, evaluation,
        source_name, target_name, voxel_size
    )
    
    print("\n" + "="*60)
    print("✅ CC-RANSAC 配准完成!")
    print("="*60)
    print(f"📊 关键结果:")
    print(f"  - 内点比率: {evaluation['inlier_ratio']:.3%}")
    print(f"  - 平均距离: {evaluation['mean_distance']:.6f}")
    print(f"  - 中位数距离: {evaluation['median_distance']:.6f}")
    print(f"  - 结果保存到: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()