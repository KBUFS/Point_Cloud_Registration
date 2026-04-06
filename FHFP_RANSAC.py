import open3d as o3d
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os
import glob

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
print("✅ 已设置中文字体支持")

# 设置随机种子
np.random.seed(42)

def load_point_clouds():
    """加载点云文件"""
    print("搜索点云文件...")
    
    # 查找data目录下的ply文件
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
        return create_test_data()
    
    print(f"源点云点数: {len(source.points):,}")
    print(f"目标点云点数: {len(target.points):,}")
    
    return source, target, os.path.basename(file1), os.path.basename(file2)

def create_test_data():
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

def preprocess_point_cloud(pcd, voxel_size):
    """预处理点云：下采样和估计法线"""
    print(f"  下采样 (体素大小: {voxel_size})...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 2
    print(f"  估计法线 (搜索半径: {radius_normal})...")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    return pcd_down

def extract_fpfh_features(pcd_down, voxel_size):
    """提取FPFH特征"""
    radius_feature = voxel_size * 5
    print(f"  提取FPFH特征 (搜索半径: {radius_feature})...")
    
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """执行全局配准 (RANSAC)"""
    print("\n执行全局配准 (RANSAC)...")
    
    distance_threshold = voxel_size * 1.5
    print(f"  距离阈值: {distance_threshold}")
    
    start_time = time.time()
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,  # RANSAC采样点数
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999))
    
    elapsed_time = time.time() - start_time
    print(f"  RANSAC耗时: {elapsed_time:.2f}秒")
    
    return result

def execute_local_refinement(source, target, initial_transformation, voxel_size):
    """执行局部精配准 (ICP)"""
    print("\n执行局部精配准 (ICP)...")
    
    distance_threshold = voxel_size * 1.5
    print(f"  ICP距离阈值: {distance_threshold}")
    
    start_time = time.time()
    
    reg_result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100,
            relative_fitness=1e-6,
            relative_rmse=1e-6
        )
    )
    
    elapsed_time = time.time() - start_time
    print(f"  ICP耗时: {elapsed_time:.2f}秒")
    
    return reg_result

def visualize_registration_step_by_step(source, target, source_down, target_down, 
                                      ransac_result, icp_result, voxel_size):
    """逐步可视化配准过程"""
    # 1. 原始点云
    print("\n可视化步骤1: 原始点云")
    source_original = copy.deepcopy(source)
    target_original = copy.deepcopy(target)
    
    source_original.paint_uniform_color([1, 0, 0])  # 红色
    target_original.paint_uniform_color([0, 1, 0])  # 绿色
    
    o3d.visualization.draw_geometries(
        [source_original, target_original],
        window_name="1. 原始点云 (红:源, 绿:目标)",
        width=1000, height=800
    )
    
    # 2. 下采样后的点云
    print("可视化步骤2: 下采样后的点云")
    source_down_vis = copy.deepcopy(source_down)
    target_down_vis = copy.deepcopy(target_down)
    
    source_down_vis.paint_uniform_color([1, 0, 0])
    target_down_vis.paint_uniform_color([0, 1, 0])
    
    o3d.visualization.draw_geometries(
        [source_down_vis, target_down_vis],
        window_name=f"2. 下采样点云 (体素大小: {voxel_size})",
        width=1000, height=800
    )
    
    # 3. RANSAC粗配准结果
    print("可视化步骤3: RANSAC粗配准结果")
    source_ransac = copy.deepcopy(source_down)
    source_ransac.transform(ransac_result.transformation)
    
    source_ransac.paint_uniform_color([0, 0, 1])  # 蓝色: RANSAC结果
    target_down_vis = copy.deepcopy(target_down)
    target_down_vis.paint_uniform_color([0, 1, 0])  # 绿色: 目标
    
    o3d.visualization.draw_geometries(
        [source_ransac, target_down_vis],
        window_name=f"3. RANSAC粗配准结果 (适应度: {ransac_result.fitness:.3f})",
        width=1000, height=800
    )
    
    # 4. ICP精配准结果
    print("可视化步骤4: ICP精配准结果")
    source_icp = copy.deepcopy(source_down)
    source_icp.transform(icp_result.transformation)
    
    source_icp.paint_uniform_color([1, 0.5, 0])  # 橙色: ICP结果
    target_down_vis = copy.deepcopy(target_down)
    target_down_vis.paint_uniform_color([0, 1, 0])  # 绿色: 目标
    
    o3d.visualization.draw_geometries(
        [source_icp, target_down_vis],
        window_name=f"4. ICP精配准结果 (适应度: {icp_result.fitness:.3f})",
        width=1000, height=800
    )
    
    # 5. 最终完整点云配准结果
    print("可视化步骤5: 完整点云配准结果")
    source_full = copy.deepcopy(source)
    source_full.transform(icp_result.transformation)
    
    source_full.paint_uniform_color([1, 0, 0])  # 红色: 源
    target_original = copy.deepcopy(target)
    target_original.paint_uniform_color([0, 1, 0])  # 绿色: 目标
    
    o3d.visualization.draw_geometries(
        [source_full, target_original],
        window_name="5. 完整点云配准结果",
        width=1000, height=800
    )

def evaluate_transformation(source, target, transformation, voxel_size):
    """评估变换矩阵的质量"""
    print("\n" + "="*50)
    print("配准结果评估")
    print("="*50)
    
    # 应用变换
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    
    # 计算最近邻距离
    source_points = np.asarray(source_transformed.points)
    target_points = np.asarray(target.points)
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_points)
    distances, indices = nbrs.kneighbors(source_points)
    
    # 计算统计信息
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    max_distance = np.max(distances)
    min_distance = np.min(distances)
    std_distance = np.std(distances)
    
    # 内点比率 (距离小于2倍体素大小)
    inlier_threshold = voxel_size * 2
    inlier_ratio = np.sum(distances < inlier_threshold) / len(distances)
    
    print(f"平均点对点距离: {mean_distance:.6f}")
    print(f"中位数距离:    {median_distance:.6f}")
    print(f"最大距离:      {max_distance:.6f}")
    print(f"最小距离:      {min_distance:.6f}")
    print(f"标准差:        {std_distance:.6f}")
    print(f"内点比率 (<{inlier_threshold:.3f}): {inlier_ratio:.3%}")
    
    # 距离分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(distances.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=mean_distance, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_distance:.4f}')
    plt.axvline(x=median_distance, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_distance:.4f}')
    plt.xlabel('点对点距离', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title('点对点距离分布', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('distance_distribution.png', dpi=150)
    print("距离分布图已保存为 'distance_distribution.png'")
    
    return mean_distance, inlier_ratio, source_transformed

def fpfh_ransac_registration(source, target, voxel_size=0.05, visualize=True):
    """完整的FPFH+RANSAC+ICP配准流程"""
    print("="*60)
    print("FPFH + RANSAC + ICP 点云配准")
    print("="*60)
    print(f"体素大小: {voxel_size}")
    print(f"源点云点数: {len(source.points):,}")
    print(f"目标点云点数: {len(target.points):,}")
    
    # 步骤1: 预处理
    print("\n[步骤1] 预处理点云")
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)
    
    print(f"  下采样后源点数: {len(source_down.points):,}")
    print(f"  下采样后目标点数: {len(target_down.points):,}")
    
    # 步骤2: 提取FPFH特征
    print("\n[步骤2] 提取FPFH特征")
    source_fpfh = extract_fpfh_features(source_down, voxel_size)
    target_fpfh = extract_fpfh_features(target_down, voxel_size)
    
    print(f"  源点云FPFH特征维度: {source_fpfh.data.shape}")
    print(f"  目标点云FPFH特征维度: {target_fpfh.data.shape}")
    
    # 步骤3: RANSAC全局配准
    ransac_result = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    
    print(f"\nRANSAC结果:")
    print(f"  适应度: {ransac_result.fitness:.4f}")
    print(f"  RMSE: {ransac_result.inlier_rmse:.6f}")
    print(f"  对应点数量: {ransac_result.correspondence_set}")
    
    if ransac_result.fitness < 0.1:
        print("⚠️  RANSAC适应度过低，可能配准失败")
    
    # 步骤4: ICP局部精配准
    icp_result = execute_local_refinement(
        source_down, target_down, ransac_result.transformation, voxel_size)
    
    print(f"\nICP精配准结果:")
    print(f"  适应度: {icp_result.fitness:.4f}")
    print(f"  RMSE: {icp_result.inlier_rmse:.6f}")
    
    # 步骤5: 评估结果
    mean_distance, inlier_ratio, source_registered = evaluate_transformation(
        source, target, icp_result.transformation, voxel_size)
    
    # 步骤6: 可视化
    if visualize:
        visualize_registration_step_by_step(
            source, target, source_down, target_down, ransac_result, icp_result, voxel_size)
    
    # 保存结果
    save_registration_results(source, target, source_registered, 
                            ransac_result, icp_result, voxel_size)
    
    return icp_result.transformation, icp_result.fitness, mean_distance, inlier_ratio

def save_registration_results(source, target, source_registered, 
                            ransac_result, icp_result, voxel_size):
    """保存配准结果"""
    print("\n保存配准结果...")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"FPFE_registration_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存点云
    o3d.io.write_point_cloud(f"{output_dir}/source_original.ply", source)
    o3d.io.write_point_cloud(f"{output_dir}/target_original.ply", target)
    o3d.io.write_point_cloud(f"{output_dir}/source_registered.ply", source_registered)
    
    # 保存变换矩阵
    with open(f"{output_dir}/ransac_transformation.txt", "w",encoding='utf-8') as f:
        f.write("RANSAC粗配准变换矩阵\n")
        f.write(f"适应度: {ransac_result.fitness:.6f}\n")
        f.write(f"RMSE: {ransac_result.inlier_rmse:.6f}\n\n")
        f.write("变换矩阵:\n")
        for row in ransac_result.transformation:
            f.write(f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f} {row[3]:.8f}\n")
    
    with open(f"{output_dir}/icp_transformation.txt", "w",encoding='utf-8') as f:
        f.write("ICP精配准变换矩阵\n")
        f.write(f"适应度: {icp_result.fitness:.6f}\n")
        f.write(f"RMSE: {icp_result.inlier_rmse:.6f}\n\n")
        f.write("变换矩阵:\n")
        for row in icp_result.transformation:
            f.write(f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f} {row[3]:.8f}\n")
    
    # 保存参数和结果总结
    with open(f"{output_dir}/summary.txt", "w",encoding='utf-8') as f:
        f.write("点云配准结果总结\n")
        f.write("="*50 + "\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"体素大小: {voxel_size}\n\n")
        
        f.write("RANSAC粗配准:\n")
        f.write(f"  适应度: {ransac_result.fitness:.6f}\n")
        f.write(f"  RMSE: {ransac_result.inlier_rmse:.6f}\n\n")
        
        f.write("ICP精配准:\n")
        f.write(f"  适应度: {icp_result.fitness:.6f}\n")
        f.write(f"  RMSE: {icp_result.inlier_rmse:.6f}\n")
    
    print(f"结果已保存到目录: {output_dir}")

def compare_different_voxel_sizes(source, target):
    """比较不同体素大小对配准效果的影响"""
    print("\n" + "="*60)
    print("比较不同体素大小对配准效果的影响")
    print("="*60)
    
    # 测试的体素大小范围
    voxel_sizes = [0.01, 0.02, 0.05, 0.1]
    results = []
    
    for i, voxel_size in enumerate(voxel_sizes, 1):
        print(f"\n>>> 测试 {i}/{len(voxel_sizes)}: 体素大小 = {voxel_size}")
        print("-" * 40)
        
        try:
            transformation, fitness, mean_distance, inlier_ratio = fpfh_ransac_registration(
                copy.deepcopy(source), copy.deepcopy(target), voxel_size, visualize=False)
            
            results.append({
                'voxel_size': voxel_size,
                'fitness': fitness,
                'mean_distance': mean_distance,
                'inlier_ratio': inlier_ratio,
                'transformation': transformation
            })
            
            print(f"  适应度: {fitness:.4f}")
            print(f"  平均距离: {mean_distance:.6f}")
            print(f"  内点比率: {inlier_ratio:.3%}")
            
        except Exception as e:
            print(f"  ❌ 配准失败: {e}")
            results.append({
                'voxel_size': voxel_size,
                'fitness': 0,
                'mean_distance': float('inf'),
                'inlier_ratio': 0,
                'transformation': np.eye(4)
            })
    
    # 绘制比较结果
    plot_comparison_results(results)
    
    # 找出最佳体素大小（适应度最高）
    valid_results = [r for r in results if r['fitness'] > 0.1]  # 过滤掉完全失败的
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['fitness'])
        best_voxel = best_result['voxel_size']
        print(f"\n✅ 最佳体素大小: {best_voxel}")
        print(f"   适应度: {best_result['fitness']:.4f}")
        print(f"   内点比率: {best_result['inlier_ratio']:.3%}")
        
        return best_voxel, best_result['transformation']
    else:
        print("❌ 没有找到有效的配准结果")
        return None, None

def plot_comparison_results(results):
    """绘制比较结果图表"""
    if not results:
        return
    
    # 过滤有效结果
    valid_results = [r for r in results if r['fitness'] > 0]
    if not valid_results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    voxel_sizes = [r['voxel_size'] for r in valid_results]
    fitnesses = [r['fitness'] for r in valid_results]
    mean_distances = [r['mean_distance'] for r in valid_results]
    inlier_ratios = [r['inlier_ratio'] for r in valid_results]
    
    # 1. 适应度 vs 体素大小
    axes[0, 0].plot(voxel_sizes, fitnesses, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('体素大小', fontsize=12)
    axes[0, 0].set_ylabel('适应度', fontsize=12)
    axes[0, 0].set_title('适应度 vs 体素大小', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.1])
    
    # 2. 平均距离 vs 体素大小
    axes[0, 1].plot(voxel_sizes, mean_distances, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('体素大小', fontsize=12)
    axes[0, 1].set_ylabel('平均距离', fontsize=12)
    axes[0, 1].set_title('平均距离 vs 体素大小', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 内点比率 vs 体素大小
    axes[1, 0].plot(voxel_sizes, inlier_ratios, 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('体素大小', fontsize=12)
    axes[1, 0].set_ylabel('内点比率', fontsize=12)
    axes[1, 0].set_title('内点比率 vs 体素大小', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.1])
    
    # 4. 适应度 vs 内点比率（散点图）
    scatter = axes[1, 1].scatter(fitnesses, inlier_ratios, c=voxel_sizes, s=100, cmap='viridis')
    axes[1, 1].set_xlabel('适应度', fontsize=12)
    axes[1, 1].set_ylabel('内点比率', fontsize=12)
    axes[1, 1].set_title('适应度 vs 内点比率', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('体素大小', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('voxel_size_comparison_enhanced.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n比较图表已保存为 'voxel_size_comparison_enhanced.png'")
    
    # 打印比较表格
    print("\n" + "="*60)
    print("体素大小比较结果")
    print("="*60)
    print(f"{'体素大小':<12} {'适应度':<10} {'平均距离':<15} {'内点比率':<12}")
    print("-" * 60)
    for r in results:
        if r['fitness'] > 0:
            print(f"{r['voxel_size']:<12.4f} {r['fitness']:<10.4f} {r['mean_distance']:<15.6f} {r['inlier_ratio']:<12.3%}")
        else:
            print(f"{r['voxel_size']:<12.4f} {'失败':<10} {'N/A':<15} {'N/A':<12}")

def main():
    """主函数"""
    print("="*60)
    print("FPFH + RANSAC 点云配准系统")
    print("="*60)
    
    # 加载点云
    source, target, source_name, target_name = load_point_clouds()
    
    if source is None or target is None:
        print("❌ 无法加载点云，退出程序")
        return
    
    print(f"\n处理文件:")
    print(f"  源点云: {source_name}")
    print(f"  目标点云: {target_name}")
    
    # 让用户选择体素大小，一般下采样点数不少于1000点
    print("\n选择体素大小 (下采样参数):")
    print("  [1] 0.01 (精细，速度慢,推荐)")
    print("  [2] 0.02 (中等)")
    print("  [3] 0.05 ")
    print("  [4] 0.1 (粗糙，速度快)")
    print("  [5] 比较不同体素大小")
    
    try:
        choice = int(input("\n请选择 (1-5): "))
    except:
        print("输入错误，使用默认值 0.05")
        choice = 3  # 默认选择0.05
    
    if choice == 1:
        voxel_size = 0.01
        transformation, fitness, mean_distance, inlier_ratio = fpfh_ransac_registration(
            source, target, voxel_size, visualize=True)
        
    elif choice == 2:
        voxel_size = 0.02
        transformation, fitness, mean_distance, inlier_ratio = fpfh_ransac_registration(
            source, target, voxel_size, visualize=True)
        
    elif choice == 3:
        voxel_size = 0.05
        transformation, fitness, mean_distance, inlier_ratio = fpfh_ransac_registration(
            source, target, voxel_size, visualize=True)
        
    elif choice == 4:
        voxel_size = 0.1
        transformation, fitness, mean_distance, inlier_ratio = fpfh_ransac_registration(
            source, target, voxel_size, visualize=True)
        
    elif choice == 5:
        # 比较不同体素大小
        best_voxel, best_result = compare_different_voxel_sizes(source, target)
        
        if best_voxel is not None:
            print(f"\n🎯 使用最佳体素大小: {best_voxel}")
            print("使用最佳参数进行最终配准并可视化...")
            transformation, fitness, mean_distance, inlier_ratio = fpfh_ransac_registration(
                source, target, best_voxel, visualize=True)
        else:
            print("❌ 所有体素大小都配准失败，使用默认值 0.05")
            voxel_size = 0.05
            transformation, fitness, mean_distance, inlier_ratio = fpfh_ransac_registration(
                source, target, voxel_size, visualize=True)
    else:
        print("输入无效，使用默认值 0.05")
        voxel_size = 0.05
        transformation, fitness, mean_distance, inlier_ratio = fpfh_ransac_registration(
            source, target, voxel_size, visualize=True)
    
    print("\n" + "="*60)
    print("配准完成!")
    print("="*60)
    print(f"最终适应度: {fitness:.4f}")
    print(f"平均点对点距离: {mean_distance:.6f}")
    print(f"内点比率: {inlier_ratio:.3%}")
    
    if fitness > 0.6 and inlier_ratio > 0.5:
        print("✅ 配准成功!")
    elif fitness > 0.3 and inlier_ratio > 0.3:
        print("⚠️  配准效果一般")
    else:
        print("❌ 配准效果较差，可能需要调整参数")

if __name__ == "__main__":
    main()