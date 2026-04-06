import open3d as o3d
import numpy as np
import copy
import time
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
print("✅ 已设置中文字体支持")

# 设置随机种子
np.random.seed(42)

def find_all_pointcloud_files():
    """查找所有的点云文件"""
    print("搜索目录下的点云文件...")
    
    extensions = ['*.ply', '*.pcd', '*.xyz', '*.pts', '*.txt', '*.obj', '*.stl']
    all_files = []
    
    for ext in extensions:
        files = glob.glob(f"./data/{ext}", recursive=True)
        all_files.extend(files)
    
    if not all_files:
        for ext in extensions:
            files = glob.glob(f"*{ext}")
            all_files.extend(files)
    
    all_files = list(set(all_files))
    
    if not all_files:
        print("❌ 没有找到任何点云文件！")
        return []
    
    print(f"找到 {len(all_files)} 个点云文件:")
    for i, file in enumerate(all_files, 1):
        size = os.path.getsize(file) if os.path.exists(file) else 0
        print(f"  {i}. {file} ({size:,} 字节)")
    
    return all_files

def select_pointcloud_files(file_list):
    """让用户选择点云文件"""
    if not file_list:
        return None, None
    
    print("\n请选择两个点云文件进行配准:")
    print("="*50)
    
    for i, file in enumerate(file_list, 1):
        print(f"[{i}] {os.path.basename(file)}")
    
    print("\n选项:")
    print("  [a] 自动选择前两个文件")
    print("  [m] 手动输入编号")
    print("  [r] 随机选择两个文件")
    
    choice = input("\n请选择操作 (a/m/r): ").strip().lower()
    
    if choice == 'a':
        if len(file_list) >= 2:
            file1, file2 = file_list[0], file_list[1]
            print(f"选择: {os.path.basename(file1)} 和 {os.path.basename(file2)}")
            return file1, file2
        else:
            print("❌ 至少需要两个点云文件")
            return None, None
    
    elif choice == 'r':
        if len(file_list) >= 2:
            indices = np.random.choice(len(file_list), 2, replace=False)
            file1, file2 = file_list[indices[0]], file_list[indices[1]]
            print(f"随机选择: {os.path.basename(file1)} 和 {os.path.basename(file2)}")
            return file1, file2
        else:
            print("❌ 至少需要两个点云文件")
            return None, None
    
    elif choice == 'm':
        try:
            idx1 = int(input("输入第一个文件的编号: ")) - 1
            if idx1 < 0 or idx1 >= len(file_list):
                print("❌ 编号无效")
                return None, None
            
            idx2 = int(input("输入第二个文件的编号: ")) - 1
            if idx2 < 0 or idx2 >= len(file_list) or idx1 == idx2:
                print("❌ 编号无效或选择了相同文件")
                return None, None
            
            file1, file2 = file_list[idx1], file_list[idx2]
            print(f"选择: {os.path.basename(file1)} 和 {os.path.basename(file2)}")
            return file1, file2
        except ValueError:
            print("❌ 请输入有效数字")
            return None, None
    
    else:
        print("❌ 无效选择")
        return None, None

def load_point_cloud_file(filepath):
    """加载点云文件"""
    print(f"\n加载文件: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return None
    
    file_size = os.path.getsize(filepath)
    print(f"  文件大小: {file_size:,} 字节")
    
    try:
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ['.ply', '.pcd']:
            pcd = o3d.io.read_point_cloud(filepath)
            if pcd and len(pcd.points) > 0:
                print(f"  ✅ 成功加载点云，点数: {len(pcd.points):,}")
                return pcd
        
        print(f"  ❌ 无法加载文件格式: {ext}")
        return None
        
    except Exception as e:
        print(f"  ❌ 加载失败: {str(e)[:100]}")
        return None

def preprocess_point_cloud(pcd, voxel_size=0.01, min_points=100, max_points=10000):
    """预处理点云"""
    if pcd is None or len(pcd.points) == 0:
        return None
    
    print(f"  原始点数: {len(pcd.points):,}")
    
    # 移除无效点
    pcd = pcd.remove_non_finite_points()
    
    # 下采样
    if len(pcd.points) > 1000:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"  下采样后点数: {len(pcd.points):,}")
    
    # 估计法线
    if len(pcd.points) > 0 and len(pcd.normals) == 0:
        print("  估计法线...")
        radius = voxel_size * 2
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=30
            )
        )
    
    return pcd

def load_and_prepare_data():
    """加载并准备点云数据"""
    print("="*60)
    print("点云配准 - 数据准备")
    print("="*60)
    
    all_files = find_all_pointcloud_files()
    if not all_files:
        print("创建合成测试数据...")
        return create_test_data()
    
    file1, file2 = select_pointcloud_files(all_files)
    if file1 is None or file2 is None:
        print("使用前两个文件...")
        if len(all_files) >= 2:
            file1, file2 = all_files[0], all_files[1]
        else:
            print("❌ 没有足够的点云文件")
            return create_test_data()
    
    print(f"\n选择的文件:")
    print(f"  源点云: {file1}")
    print(f"  目标点云: {file2}")
    
    source = load_point_cloud_file(file1)
    target = load_point_cloud_file(file2)
    
    if source is None or target is None:
        print("❌ 点云加载失败，使用合成数据")
        return create_test_data()
    
    print("\n预处理点云...")
    # 使用较大的体素大小，因为你的点云密度很高
    source = preprocess_point_cloud(source, voxel_size=0.02)
    target = preprocess_point_cloud(target, voxel_size=0.02)
    
    if source is None or target is None or len(source.points) == 0 or len(target.points) == 0:
        print("❌ 点云预处理失败，使用合成数据")
        return create_test_data()
    
    print(f"\n预处理完成:")
    print(f"  源点云: {len(source.points):,} 个点")
    print(f"  目标点云: {len(target.points):,} 个点")
    
    return source, target, file1, file2

def create_test_data():
    """创建测试数据"""
    print("创建合成测试点云...")
    
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh.compute_vertex_normals()
    
    source = mesh.sample_points_poisson_disk(number_of_points=2000)
    
    angle = np.radians(30)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    t = np.array([0.5, 0.3, 0.2])
    
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    
    target = copy.deepcopy(source)
    target.transform(transformation)
    
    points = np.asarray(target.points)
    noise = np.random.normal(0, 0.01, points.shape)
    target.points = o3d.utility.Vector3dVector(points + noise)
    
    source.estimate_normals()
    target.estimate_normals()
    
    print(f"创建合成点云完成:")
    print(f"  源点云: {len(source.points):,} 个点")
    print(f"  目标点云: {len(target.points):,} 个点")
    
    return source, target, "合成源点云", "合成目标点云"

def point_to_point_icp_manual(source, target, max_iterations=50, tolerance=1e-6):
    """手动实现点对点ICP"""
    print(f"\n手动实现点对点ICP (最大迭代: {max_iterations})")
    
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    
    if len(source_points) < 3 or len(target_points) < 3:
        print("❌ 点云点数不足")
        return np.eye(4), []
    
    transformation = np.eye(4)
    errors = []
    
    for i in range(max_iterations):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_points)
        distances, indices = nbrs.kneighbors(source_points)
        
        current_error = np.mean(distances)
        errors.append(current_error)
        
        if i > 0 and len(errors) > 1 and abs(errors[-2] - errors[-1]) < tolerance:
            print(f"  迭代 {i+1}: 收敛! 误差: {current_error:.6f}")
            break
        
        correspondences = target_points[indices.flatten()]
        
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(correspondences, axis=0)
        
        source_centered = source_points - source_centroid
        target_centered = correspondences - target_centroid
        
        W = np.dot(source_centered.T, target_centered)
        
        U, S, Vt = np.linalg.svd(W)
        
        R = np.dot(Vt.T, U.T)
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        t = target_centroid - np.dot(R, source_centroid)
        
        delta_transform = np.eye(4)
        delta_transform[:3, :3] = R
        delta_transform[:3, 3] = t
        transformation = np.dot(delta_transform, transformation)
        
        source_points = (np.dot(R, source_points.T) + t.reshape(-1, 1)).T
        
        if (i + 1) % 10 == 0 or i == 0 or i == max_iterations - 1:
            print(f"  迭代 {i+1}: 误差 = {current_error:.6f}")
    
    print(f"✅ 手动ICP完成，最终误差: {errors[-1]:.6f}")
    return transformation, errors

def icp_open3d(source, target, method="point_to_point", max_iterations=50):
    """使用Open3D的ICP实现 - 修复版"""
    print(f"\nOpen3D {method} ICP")
    
    if len(source.points) < 3 or len(target.points) < 3:
        print("❌ 点云点数不足")
        return np.eye(4), 0, 0
    
    # 计算合适的距离阈值
    source_points = np.asarray(source.points)
    bbox = source.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_extent()
    threshold = np.max(bbox_size) * 0.05  # 使用包围盒大小的5%
    
    trans_init = np.eye(4)
    start_time = time.time()
    
    try:
        if method == "point_to_point":
            reg_result = o3d.pipelines.registration.registration_icp(
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
        elif method == "point_to_plane":
            reg_result = o3d.pipelines.registration.registration_icp(
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
        else:
            print(f"❌ 未知方法: {method}")
            return np.eye(4), 0, 0
    except Exception as e:
        print(f"❌ Open3D ICP失败: {e}")
        return np.eye(4), 0, 0
    
    elapsed_time = time.time() - start_time
    
    # 打印结果 - 注意不同Open3D版本的属性名
    print(f"  阈值: {threshold:.4f}")
    print(f"  适应度: {reg_result.fitness:.6f}")
    print(f"  RMSE: {reg_result.inlier_rmse:.6f}")
    print(f"  耗时: {elapsed_time:.3f}秒")
    
    return reg_result.transformation, reg_result.fitness, reg_result.inlier_rmse

def evaluate_registration(source, target, transformation):
    """评估配准结果"""
    print("\n" + "="*50)
    print("配准结果评估")
    print("="*50)
    
    if len(source.points) == 0 or len(target.points) == 0:
        print("❌ 点云为空")
        return float('inf'), None
    
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    
    source_points = np.asarray(source_transformed.points)
    target_points = np.asarray(target.points)
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_points)
    distances, _ = nbrs.kneighbors(source_points)
    mean_distance = np.mean(distances)
    
    print(f"平均点对点距离: {mean_distance:.6f}")
    print(f"最大距离: {np.max(distances):.6f}")
    print(f"最小距离: {np.min(distances):.6f}")
    print(f"标准差: {np.std(distances):.6f}")
    
    return mean_distance, source_transformed

def visualize_results(source, target, transformed=None, 
                     source_name="源点云", target_name="目标点云"):
    """可视化配准结果"""
    source_vis = copy.deepcopy(source)
    target_vis = copy.deepcopy(target)
    
    source_vis.paint_uniform_color([1, 0, 0])  # 红色
    target_vis.paint_uniform_color([0, 1, 0])  # 绿色
    
    geometries = [source_vis, target_vis]
    
    if transformed is not None:
        transformed_vis = copy.deepcopy(transformed)
        transformed_vis.paint_uniform_color([0, 0, 1])  # 蓝色
        geometries.append(transformed_vis)

    #坐标轴显示
    #coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    #geometries.append(coord_frame)
    
    title = f"ICP点云配准: {source_name} -> {target_name}"
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1024,
        height=768
    )

def plot_error_convergence(errors, title="ICP误差收敛曲线", save_path="icp_convergence.png"):
    """绘制ICP误差收敛曲线"""
    if not errors or len(errors) < 2:
        print("❌ 没有足够的误差数据绘制收敛曲线")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(errors) + 1), errors, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('平均对应点距离', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if len(errors) > 1:
        final_error = errors[-1]
        initial_error = errors[0]
        improvement = (initial_error - final_error) / initial_error * 100
        plt.text(0.5, 0.95, f'改进: {improvement:.1f}%', 
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"收敛曲线已保存为 '{save_path}'")

def main():
    """主函数"""
    print("="*60)
    print("多视角点云ICP配准系统")
    print("="*60)
    
    source, target, source_name, target_name = load_and_prepare_data()
    
    print(f"\n点云信息:")
    print(f"  源点云: {os.path.basename(source_name)}")
    print(f"    点数: {len(source.points):,}")
    print(f"  目标点云: {os.path.basename(target_name)}")
    print(f"    点数: {len(target.points):,}")
    
    # 可视化原始点云
    print("\n正在可视化原始点云...")
    visualize_results(source, target, None, 
                     os.path.basename(source_name), os.path.basename(target_name))
    
    print("\n" + "="*60)
    print("运行ICP配准算法")
    print("="*60)
    
    # 运行不同ICP算法
    results = []
    
    # 1. 手动实现ICP
    T_manual, errors_manual = point_to_point_icp_manual(
        copy.deepcopy(source), copy.deepcopy(target)
    )
    error_manual, source_manual = evaluate_registration(source, target, T_manual)
    results.append(("手动ICP", error_manual, T_manual))
    
    # 2. Open3D点对点ICP
    T_pp, fitness_pp, rmse_pp = icp_open3d(
        copy.deepcopy(source), copy.deepcopy(target), "point_to_point"
    )
    error_pp, source_pp = evaluate_registration(source, target, T_pp)
    results.append(("Open3D点对点", error_pp, T_pp))
    
    # 3. Open3D点对面ICP
    T_pl, fitness_pl, rmse_pl = icp_open3d(
        copy.deepcopy(source), copy.deepcopy(target), "point_to_plane"
    )
    error_pl, source_pl = evaluate_registration(source, target, T_pl)
    results.append(("Open3D点对面", error_pl, T_pl))
    
    # 比较结果
    print("\n" + "="*60)
    print("ICP算法比较")
    print("="*60)
    
    for name, error, T in results:
        print(f"\n{name}:")
        print(f"  平均误差: {error:.6f}")
        print(f"  变换矩阵:")
        for i, row in enumerate(T[:3]):
            print(f"    [{row[0]:.4f} {row[1]:.4f} {row[2]:.4f} | {row[3]:.4f}]")
        print(f"    [0.0000 0.0000 0.0000 | 1.0000]")
    
    # 选择最佳结果
    best_name, best_error, best_T = min(results, key=lambda x: x[1])
    print(f"\n✅ 最佳算法: {best_name} (误差: {best_error:.6f})")
    
    # 可视化最佳结果
    print(f"\n正在可视化最佳配准结果 ({best_name})...")
    _, source_best = evaluate_registration(source, target, best_T)
    visualize_results(source, target, source_best, 
                     os.path.basename(source_name), os.path.basename(target_name))
    
    # 绘制收敛曲线
    if errors_manual:
        plot_error_convergence(errors_manual, 
                              f"手动ICP误差收敛曲线 ({os.path.basename(source_name)})")
    
    # 保存结果
    print("\n保存配准结果...")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"icp_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    o3d.io.write_point_cloud(f"{output_dir}/source_original.ply", source)
    o3d.io.write_point_cloud(f"{output_dir}/target_original.ply", target)
    
    for name, error, T in results:
        source_reg = copy.deepcopy(source)
        source_reg.transform(T)
        filename = f"{output_dir}/source_registered_{name.replace(' ', '_')}.ply"
        o3d.io.write_point_cloud(filename, source_reg)
        
        with open(f"{output_dir}/transformation_{name.replace(' ', '_')}.txt", "w") as f:
            f.write(f"{name} 配准结果\n")
            f.write(f"平均误差: {error:.6f}\n")
            f.write("变换矩阵:\n")
            for row in T:
                f.write(f"[{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}]\n")
    
    # 保存总结报告
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write("ICP点云配准结果总结\n")
        f.write("="*50 + "\n")
        f.write(f"源点云: {source_name}\n")
        f.write(f"目标点云: {target_name}\n")
        f.write(f"源点数: {len(source.points)}\n")
        f.write(f"目标点数: {len(target.points)}\n\n")
        
        f.write("算法比较:\n")
        for i, (name, error, T) in enumerate(results, 1):
            f.write(f"\n{i}. {name}:\n")
            f.write(f"   平均误差: {error:.6f}\n")
            f.write(f"   变换矩阵:\n")
            for row in T:
                f.write(f"     [{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}]\n")
        
        best_idx = results.index((best_name, best_error, best_T))
        f.write(f"\n最佳算法: {best_name} (第{best_idx+1}个)\n")
    
    print(f"\n所有结果已保存到目录: {output_dir}")
    print("包含文件:")
    print(f"  source_original.ply - 原始源点云")
    print(f"  target_original.ply - 原始目标点云")
    print(f"  source_registered_*.ply - 各算法配准结果")
    print(f"  transformation_*.txt - 各算法变换矩阵")
    print(f"  summary.txt - 总结报告")
    
    print("\n" + "="*60)
    print("ICP点云配准完成！")
    print("="*60)

if __name__ == "__main__":
    main()