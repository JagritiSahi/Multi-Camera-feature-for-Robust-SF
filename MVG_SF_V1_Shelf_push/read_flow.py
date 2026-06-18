import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_flow_file(filename):
    """
    Read optical flow from file
    """
    print(f"Reading flow file: {filename}")
    try:
        # Read the image
        flow_img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if flow_img is None:
            raise ValueError(f"Could not read image: {filename}")
        
        print(f"Flow image shape: {flow_img.shape}")
        print(f"Flow image dtype: {flow_img.dtype}")
        print(f"Flow image min/max values: {flow_img.min()}/{flow_img.max()}")
        
        # Convert to float32
        flow = flow_img.astype(np.float32)
        print("flow===>" , np.unique(flow))
        
        # If the image is in the format where flow is encoded in RGB
        if flow.shape[2] == 3:
            # Convert from RGB to flow
            flow = (flow[:, :, 2:3] * 256 + flow[:, :, 1:2]) / 256.0
            flow = flow - 32768.0
            flow = flow / 64.0
            
            # Get the valid mask from the first channel
            valid = flow_img[:, :, 0:1] > 0
            
            # Stack the flow and valid mask
            flow = np.concatenate([flow, valid], axis=2)
        
        print(f"Processed flow shape: {flow.shape}")
        print(f"Processed flow min/max values: {flow.min()}/{flow.max()}")
        
        return flow
        
    except Exception as e:
        print(f"Error reading flow file: {str(e)}")
        return None

if __name__ == '__main__':
    # Path to your flow image
    flow_path = "/mnt/MIG_archive/Datasets/iota/Jagriti/MVGFormer_RAFT_v2/data/panoptic/171204_pose1_sample/hdFlow/00_00/00_00_00000000.png"
    
    # Read the flow data
    flow_data = read_flow_file(flow_path)
    
    # if flow_data is not None:
    #     print("\nFlow data statistics:")
    #     print(f"Shape: {flow_data.shape}")
    #     print(f"Data type: {flow_data.dtype}")
    #     print(f"Min value: {flow_data.min()}")
    #     print(f"Max value: {flow_data.max()}")
    #     print(f"Mean value: {flow_data.mean()}")
    #     print(f"Std value: {flow_data.std()}")
        
        # # Visualize the flow
        # plt.figure(figsize=(12, 4))
        
        # # Plot flow magnitude
        # plt.subplot(131)
        # flow_magnitude = np.sqrt(flow_data[:,:,0]**2 + flow_data[:,:,1]**2)
        # plt.imshow(flow_magnitude)
        # plt.colorbar()
        # plt.title('Flow Magnitude')
        
        # # Plot x component
        # plt.subplot(132)
        # plt.imshow(flow_data[:,:,0])
        # plt.colorbar()
        # plt.title('Flow X')
        
        # # Plot y component
        # plt.subplot(133)
        # plt.imshow(flow_data[:,:,1])
        # plt.colorbar()
        # plt.title('Flow Y')
        
        # plt.tight_layout()
        # plt.show() 