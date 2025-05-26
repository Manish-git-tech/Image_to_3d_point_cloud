"""
Complete Pipeline for IITISOC Selection Criteria:
1. FastSAM inference for object segmentation
2. Depth Pro for RGBD image generation and display
3. Point cloud generation and 3D visualization for each object
"""

# Import all required libraries
from fastsam import FastSAM
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
import open3d as o3d
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os

class CompletePipeline:
    def __init__(self):
        """Initialize all models and processors"""
        print("🚀 Initializing Complete Pipeline...")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize FastSAM for object segmentation
        print("Loading FastSAM model...")
        self.fastsam_model = FastSAM("FastSAM-s.pt")
        
        # Initialize Depth Pro for depth estimation
        print("Loading Depth Pro model...")
        self.depth_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        self.depth_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(self.device)
        self.depth_model.eval()
        
        print("✅ All models loaded successfully!")
    
    def run_fastsam_segmentation(self, image_path):
        """
        Step 1: Run FastSAM inference for object segmentation
        """
        print(f"\n📋 Step 1: Running FastSAM segmentation on: {image_path}")
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}")
            return None, None
        
        # Check if file is a valid image
        try:
            test_image = Image.open(image_path)
            test_image.verify()  # Verify it's a valid image
            print(f"✅ Image file validated: {test_image.size}")
        except Exception as e:
            print(f"❌ Error: Invalid image file: {e}")
            return None, None
        
        try:
            # Run FastSAM inference with error handling
            print("Running FastSAM model...")
            segmentation_results = self.fastsam_model(
                image_path,
                device=str(self.device),
                retina_masks=True,
                imgsz=1024,
                conf=0.4,
                iou=0.9,
                verbose=True  # Add verbose for debugging
            )
            
            # Check if results are valid
            if segmentation_results is None:
                print("❌ FastSAM returned None - check image path and format")
                return None, None
                
            if len(segmentation_results) == 0:
                print("❌ FastSAM returned empty results")
                return None, None
                
            # Check if first result has masks
            first_result = segmentation_results[0]
            if first_result is None:
                print("❌ First segmentation result is None")
                return None, None
                
            # Extract masks safely
            masks = None
            if hasattr(first_result, 'masks') and first_result.masks is not None:
                masks = first_result.masks.data.cpu().numpy()
                print(f"✅ Found {len(masks)} objects in the image")
            else:
                print("⚠️ No objects detected or no masks attribute")
            
            return masks, segmentation_results
            
        except Exception as e:
            print(f"❌ FastSAM inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def run_depth_estimation(self, image_path):
        """
        Step 2: Run Depth Pro for depth estimation and RGBD creation
        """
        print("\n🔍 Step 2: Running Depth Pro estimation...")
        
        # Load image
        image_pil = Image.open(image_path).convert("RGB")
        
        # Process with Depth Pro
        inputs = self.depth_processor(images=image_pil, return_tensors="pt").to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
        
        # Post-process outputs
        post_processed = self.depth_processor.post_process_depth_estimation(
            outputs, target_sizes=[(image_pil.height, image_pil.width)]
        )
        
        # Extract results
        depth_map = post_processed[0]["predicted_depth"].squeeze().cpu().numpy()
        field_of_view = post_processed[0].get("field_of_view", torch.tensor(60.0)).item()
        focal_length = post_processed[0].get("focal_length", torch.tensor(500.0)).item()
        
        # Normalize for visualization
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        print(f"✅ Depth estimation completed")
        print(f"   - Depth range: {depth_map.min():.2f} to {depth_map.max():.2f} meters")
        print(f"   - Estimated FOV: {field_of_view:.1f} degrees")
        print(f"   - Estimated focal length: {focal_length:.1f} pixels")
        
        return depth_map, depth_normalized, image_pil, field_of_view, focal_length
    
    def generate_object_point_clouds(self, image_pil, depth_map, masks, focal_length=500.0):
        """
        Generate separate point clouds for each detected object
        """
        print(f"\n🔗 Step 3: Generating point clouds for {len(masks)} objects...")
        
        # Convert PIL image to numpy array
        rgb_array = np.array(image_pil)
        height, width = rgb_array.shape[:2]
        
        # Convert depth to appropriate scale for Open3D
        depth_array = (depth_map * 1000).astype(np.uint16)
        
        # Set up camera intrinsics
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=focal_length,
            fy=focal_length,
            cx=width / 2.0,
            cy=height / 2.0
        )
        
        object_point_clouds = []
        combined_point_cloud = o3d.geometry.PointCloud()
        
        # Generate colors for each object
        colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))
        
        for i, mask in enumerate(masks):
            print(f"   Processing object {i+1}/{len(masks)}...")
            
            # Apply mask to RGB and depth images
            masked_rgb = rgb_array.copy()
            masked_depth = depth_array.copy()
            
            # Convert mask to boolean and ensure correct shape
            mask_bool = mask.astype(bool)
            
            # Apply mask (set non-object pixels to zero/black)
            masked_rgb[~mask_bool] = [0, 0, 0]  # Black background
            masked_depth[~mask_bool] = 0  # Zero depth for non-object pixels
            
            # Create Open3D images for this object
            color_o3d = o3d.geometry.Image(masked_rgb)
            depth_o3d = o3d.geometry.Image(masked_depth)
            
            # Create RGBD image for this object
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                convert_rgb_to_intensity=False
            )
            
            # Generate point cloud for this object
            pcd_object = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                camera_intrinsic
            )
            
            # Transform for correct orientation
            pcd_object.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            
            # Remove points with zero coordinates (background)
            if len(pcd_object.points) > 0:
                points = np.asarray(pcd_object.points)
                colors_rgb = np.asarray(pcd_object.colors)
                
                # Filter out black/zero points
                valid_mask = np.any(colors_rgb > 0.01, axis=1)  # Keep points with some color
                
                if np.any(valid_mask):
                    filtered_points = points[valid_mask]
                    filtered_colors = colors_rgb[valid_mask]
                    
                    # Create filtered point cloud
                    pcd_filtered = o3d.geometry.PointCloud()
                    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
                    pcd_filtered.colors = o3d.utility.Vector3dVector(filtered_colors)
                    
                    object_point_clouds.append({
                        'point_cloud': pcd_filtered,
                        'object_id': i + 1,
                        'num_points': len(filtered_points)
                    })
                    
                    # Add to combined point cloud with unique color
                    pcd_colored = o3d.geometry.PointCloud()
                    pcd_colored.points = o3d.utility.Vector3dVector(filtered_points)
                    
                    # Assign unique color to each object
                    unique_color = colors[i][:3]  # RGB from colormap
                    object_colors = np.tile(unique_color, (len(filtered_points), 1))
                    pcd_colored.colors = o3d.utility.Vector3dVector(object_colors)
                    
                    combined_point_cloud += pcd_colored
                    
                    print(f"     ✅ Object {i+1}: {len(filtered_points)} points")
                else:
                    print(f"     ⚠️ Object {i+1}: No valid points after filtering")
            else:
                print(f"     ⚠️ Object {i+1}: No points generated")
        
        print(f"✅ Generated {len(object_point_clouds)} object point clouds")
        print(f"✅ Combined point cloud: {len(combined_point_cloud.points)} total points")
        
        return object_point_clouds, combined_point_cloud
    
    def visualize_multi_object_results(self, image_pil, masks, depth_normalized, object_point_clouds, combined_pcd, output_dir):
        """
        Create comprehensive visualization showing individual and combined point clouds
        """
        print("\n📊 Step 4: Creating multi-object visualizations...")
        
        # Create main results figure
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Original Image
        ax1 = fig.add_subplot(3, 4, 1)
        ax1.imshow(image_pil)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Plot 2: All Segmentation Masks
        ax2 = fig.add_subplot(3, 4, 2)
        if masks is not None:
            combined_mask = np.zeros_like(masks[0])
            for i, mask in enumerate(masks):
                combined_mask += mask * (i + 1)
            ax2.imshow(combined_mask, cmap='tab20')
            ax2.set_title(f'All Objects ({len(masks)} detected)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Plot 3: Depth Map
        ax3 = fig.add_subplot(3, 4, 3)
        im = ax3.imshow(depth_normalized, cmap='plasma')
        ax3.set_title('Depth Map', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # Plot 4: 3D Combined Point Cloud Preview
        ax4 = fig.add_subplot(3, 4, 4, projection='3d')
        if len(combined_pcd.points) > 0:
            points = np.asarray(combined_pcd.points)
            colors = np.asarray(combined_pcd.colors)
            
            # Subsample for visualization
            step = max(1, len(points) // 1000)
            points_sub = points[::step]
            colors_sub = colors[::step]
            
            ax4.scatter(points_sub[:, 0], points_sub[:, 1], points_sub[:, 2], 
                       c=colors_sub, s=1)
            ax4.set_title('Combined Point Cloud', fontsize=12, fontweight='bold')
        
        # Plot 5-8: Individual object masks
        for i in range(min(4, len(masks))):
            ax = fig.add_subplot(3, 4, 5 + i)
            ax.imshow(masks[i], cmap='gray')
            ax.set_title(f'Object {i+1}', fontsize=10)
            ax.axis('off')
        
        # Plot 9-12: Point cloud statistics
        for i, obj_data in enumerate(object_point_clouds[:4]):
            ax = fig.add_subplot(3, 4, 9 + i)
            ax.text(0.5, 0.5, f"Object {obj_data['object_id']}\n{obj_data['num_points']} points", 
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f"Object {obj_data['object_id']} Stats", fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save results
        results_path = os.path.join(output_dir, 'multi_object_pipeline_results.png')
        plt.savefig(results_path, dpi=150, bbox_inches='tight')
        print(f"✅ Results saved to: {results_path}")
        plt.show()
        
        return fig
    
    def display_multi_object_point_clouds(self, object_point_clouds, combined_pcd):
        """
        Display point clouds with options for individual and combined viewing
        """
        print("\n🎮 Step 6: Displaying multi-object point clouds...")
        
        # Option 1: Display combined point cloud (all objects with different colors)
        print("Displaying combined point cloud (all objects)...")
        o3d.visualization.draw_geometries(
            [combined_pcd],
            window_name="All Objects - Combined Point Cloud",
            width=1000,
            height=700
        )
        
        # Option 2: Display each object individually
        for obj_data in object_point_clouds:
            obj_id = obj_data['object_id']
            pcd = obj_data['point_cloud']
            num_points = obj_data['num_points']
            
            print(f"Displaying Object {obj_id} ({num_points} points)...")
            o3d.visualization.draw_geometries(
                [pcd],
                window_name=f"Object {obj_id} - Individual Point Cloud",
                width=800,
                height=600
            )
        
        # Option 3: Display all objects simultaneously but separately
        if len(object_point_clouds) > 0:
            all_individual_pcds = [obj['point_cloud'] for obj in object_point_clouds]
            print("Displaying all objects separately...")
            o3d.visualization.draw_geometries(
                all_individual_pcds,
                window_name="All Objects - Separate Point Clouds",
                width=1000,
                height=700
            )
    
    def save_multi_object_outputs(self, object_point_clouds, combined_pcd, depth_map, masks, output_dir):
        """
        Save all outputs to files
        """
        print("\n💾 Step 5: Saving outputs...")
        
        # Save combined point cloud
        combined_path = os.path.join(output_dir, 'combined_point_cloud.ply')
        o3d.io.write_point_cloud(combined_path, combined_pcd)
        print(f"✅ Combined point cloud saved: {combined_path}")
        
        # Save individual point clouds
        for obj_data in object_point_clouds:
            obj_id = obj_data['object_id']
            pcd = obj_data['point_cloud']
            obj_path = os.path.join(output_dir, f'object_{obj_id}_point_cloud.ply')
            o3d.io.write_point_cloud(obj_path, pcd)
            print(f"✅ Object {obj_id} point cloud saved: {obj_path}")
        
        # Save depth map
        depth_path = os.path.join(output_dir, 'depth_map.npy')
        np.save(depth_path, depth_map)
        print(f"✅ Depth map saved: {depth_path}")
        
        # Save segmentation masks
        if masks is not None:
            masks_path = os.path.join(output_dir, 'segmentation_masks.npy')
            np.save(masks_path, masks)
            print(f"✅ Segmentation masks saved: {masks_path}")
        
        return combined_path, depth_path
    
    def run_complete_pipeline(self, image_path, output_dir="./pipeline_output/"):
        """
        Execute the complete pipeline with multi-object point cloud generation
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("🎯 Starting Multi-Object Pipeline for IITISOC Selection...")
        print("=" * 60)
        
        try:
            # Step 1: FastSAM Segmentation
            masks, seg_results = self.run_fastsam_segmentation(image_path)
            
            if masks is None:
                print("❌ No objects detected. Cannot proceed with multi-object point clouds.")
                return None
            
            # Step 2: Depth Estimation
            depth_map, depth_norm, image_pil, fov, focal_len = self.run_depth_estimation(image_path)
            
            # Step 3: Multi-Object Point Cloud Generation
            object_point_clouds, combined_pcd = self.generate_object_point_clouds(
                image_pil, depth_map, masks, focal_len
            )
            
            # Step 4: Enhanced Visualization
            fig = self.visualize_multi_object_results(
                image_pil, masks, depth_norm, object_point_clouds, combined_pcd, output_dir
            )
            
            # Step 5: Save All Outputs
            self.save_multi_object_outputs(object_point_clouds, combined_pcd, depth_map, masks, output_dir)
            
            # Step 6: Interactive Display
            self.display_multi_object_point_clouds(object_point_clouds, combined_pcd)
            
            # Pipeline Summary
            print("\n" + "=" * 60)
            print("🎉 MULTI-OBJECT PIPELINE COMPLETED!")
            print("=" * 60)
            print("✅ IITISOC Selection Criteria Met:")
            print(f"   1. FastSAM Segmentation: {len(masks)} objects detected")
            print(f"   2. RGBD Image: ✅ Generated with Depth Pro")
            print(f"   3. Point Clouds: ✅ {len(object_point_clouds)} individual + 1 combined")
            print(f"   4. Total points: {len(combined_pcd.points)}")
            print("=" * 60)
            
            return {
                'masks': masks,
                'depth_map': depth_map,
                'object_point_clouds': object_point_clouds,
                'combined_point_cloud': combined_pcd,
                'image': image_pil,
                'output_dir': output_dir
            }
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# Main execution
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = CompletePipeline()
    
    # Set your image path
    image_path = "images\Arin.jpeg"  # Replace with your image path
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(image_path)
    
    if results:
        print("\n🚀 Pipeline completed! Check the pipeline_output/ directory for all results.")
    else:
        print("\n⚠️ Pipeline failed. Please check error messages above.")
