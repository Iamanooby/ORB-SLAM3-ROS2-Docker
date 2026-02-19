import numpy as np
import open3d as o3d
import copy

def prepare_dataset(source, target, voxel_size):
    # Downsample and compute Fast Point Feature Histograms (FPFH)
    def preprocess(pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        return pcd_down, fpfh

    source_down, source_fpfh = preprocess(source, voxel_size)
    target_down, target_fpfh = preprocess(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # RANSAC based on feature matching
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def draw_registration_result(source, target, transformation):
    # Deep copy to avoid modifying the original objects
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    # Paint them so we can distinguish them
    # RGB: Red (1,0,0) and Blue (0,0,1)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    
    # Apply the calculated transformation matrix to the source
    source_temp.transform(transformation)
    
    print("Opening visualization window...")
    print("Controls: Left-click to rotate, Shift+Left-click to pan, Scroll to zoom.")
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      window_name="ICP Match Result",
                                      width=1024, height=768)



# --- Main Logic ---
voxel_size = 0.05  # Adjust based on your cloud scale (e.g., 5cm)
source = csv_to_pcd("../csv_files/lidar_map_points.csv") # Using previous helper
target = csv_to_pcd("../csv_files/orb_map_points.csv")

# 1. Global Registration (The "Rough" Match)
s_down, t_down, s_fpfh, t_fpfh = prepare_dataset(source, target, voxel_size)
result_ransac = execute_global_registration(s_down, t_down, s_fpfh, t_fpfh, voxel_size)

# 2. Local ICP (The "Fine" Match)
# We use the RANSAC result as the starting point (trans_init)
result_icp = o3d.pipelines.registration.registration_icp(
    source, target, voxel_size / 2, result_ransac.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print(f"Final Similarity (Fitness): {result_icp.fitness:.4f}")
print(f"Final Precision (RMSE):    {result_icp.inlier_rmse:.4f}")
# --- Usage (Add this at the end of your main script) ---
draw_registration_result(source, target, result_icp.transformation)