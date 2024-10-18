import os
import numpy as np
import scipy.io as sio
import shutil
import argparse

def prepare_training_data(input_dir, output_base_dir):
    """
    Prepare training, validation, and testing datasets for TSA-Net
    
    Args:
        input_dir: Directory containing the original TSA_simu_data
        output_base_dir: Base directory where processed data will be saved
    """

    dirs = {
        'train': os.path.join(output_base_dir, 'Training_truth'),
        'valid': os.path.join(output_base_dir, 'Valid_truth'),
        'test': os.path.join(output_base_dir, 'Testing_truth'),
        'mask': os.path.join(output_base_dir, 'Mask')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    truth_dir = os.path.join(input_dir, 'Truth')
    truth_files = [f for f in os.listdir(truth_dir) if f.endswith('.mat')]
    
    # Split data: 80% training, 20% validation
    np.random.seed(42)
    np.random.shuffle(truth_files)
    
    n_files = len(truth_files)
    n_train = int(0.8 * n_files)
    n_valid = int(0.2 * n_files)
    
    splits = {
        'train': truth_files[:n_train],
        'valid': truth_files[n_train:n_train+n_valid],
        'test': truth_files[n_train+n_valid:]
    }

    for split_name, files in splits.items():
        for file in files:
            input_path = os.path.join(truth_dir, file)
            output_path = os.path.join(dirs[split_name], file)
            
            data = sio.loadmat(input_path)
            img = data['img']
            
            if split_name == 'train':
                # Scale to 0-65535 for training data
                img = (img - img.min()) / (img.max() - img.min()) * 65535
            else:
                # Scale to 0-1 for validation and test data
                img = (img - img.min()) / (img.max() - img.min())
            
            if img.shape != (256, 256, 28):
                img = resize_data(img, (256, 256, 28))
            
            sio.savemat(output_path, {'img': img})
    
    mask_source = os.path.join(input_dir, 'mask.mat')
    if os.path.exists(mask_source):
        shutil.copy2(mask_source, os.path.join(dirs['mask'], 'mask.mat'))

def resize_data(data, target_shape):
    """
    Resize the data to target shape
    """
    from scipy.ndimage import zoom
    
    current_shape = np.array(data.shape)
    target_shape = np.array(target_shape)
    zoom_factors = target_shape / current_shape
    
    return zoom(data, zoom_factors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TSA-Net training, validation, and testing data.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the original TSA_simu_data")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory where processed data will be saved")
    
    args = parser.parse_args()
    
    prepare_training_data(args.input_dir, args.output_dir)
    print("Dataset preparation completed!")