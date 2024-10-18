import os
import scipy.io as sio
import scipy.ndimage as ndimage
import numpy as np
import argparse

def rescale_all_files_in_folder(input_folder, output_folder, target_size=(256, 256), var_name='img_expand', rescaled_var_name='img'):
    """
    Rescales all .mat files in the input_folder and saves the rescaled versions in the output_folder.

    Parameters:
    - input_folder: Path to the folder containing the original .mat files.
    - output_folder: Path to the folder where the rescaled files will be saved.
    - target_size: The desired spatial dimensions (height, width) after rescaling.
    - var_name: The variable name in the .mat files to be rescaled (default: 'img_expand').
    - rescaled_var_name: The variable name for saving the rescaled image (default: 'img').
    """

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.mat'):
            file_path = os.path.join(input_folder, file_name)
            try:
                # Load the .mat file
                data = sio.loadmat(file_path)

                # Check if the target variable is present in the file
                if var_name in data:
                    img_expand = data[var_name]

                    # Check if the third dimension is already 28 channels
                    if img_expand.shape[-1] == 28:
                        # Rescale the spatial dimensions
                        original_size = img_expand.shape[:2]
                        scale_factor = (target_size[0] / original_size[0], target_size[1] / original_size[1], 1)
                        img_rescaled = ndimage.zoom(img_expand, scale_factor, order=3)  # Bicubic interpolation

                        # Clip negative values and normalize
                        img_rescaled[img_rescaled < 0] = 0
                        img_rescaled = img_rescaled.astype(np.float32)

                        # Save the rescaled image in the output folder
                        output_file_path = os.path.join(output_folder, file_name)
                        sio.savemat(output_file_path, {rescaled_var_name: img_rescaled})

                        print(f"Rescaled and saved: {file_name}")
                    else:
                        print(f"Skipping {file_name}: Unexpected channel size {img_expand.shape[-1]}")
                else:
                    print(f"Skipping {file_name}: '{var_name}' not found.")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Rescale .mat files in a folder.")
    
    parser.add_argument('--input_dir', required=True, help="Path to the input folder containing .mat files.")
    parser.add_argument('--output_dir', required=True, help="Path to the output folder to save rescaled .mat files.")
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256], help="Target size (height width) for rescaling.")
    parser.add_argument('--var_name', type=str, default='img_expand', help="Variable name in the .mat file to be rescaled.")
    parser.add_argument('--rescaled_var_name', type=str, default='img', help="Variable name for the rescaled image.")
    
    args = parser.parse_args()

    # Call the rescaling function with provided arguments
    rescale_all_files_in_folder(
        input_folder=args.input_dir,
        output_folder=args.output_dir,
        target_size=tuple(args.target_size),
        var_name=args.var_name,
        rescaled_var_name=args.rescaled_var_name
    )

if __name__ == "__main__":
    main()
