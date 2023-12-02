import os
from tqdm import tqdm

def delete_files_except(folder_path, preserve_files):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    try:
        # Get a list of all files in the folder
        files_to_delete = [f for f in os.listdir(folder_path) if f not in preserve_files]

        # Delete each file in the list
        for file_to_delete in tqdm(files_to_delete):
            file_path = os.path.join(folder_path, file_to_delete)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

        # Print preserved files
        for preserve_file in preserve_files:
            preserve_file_path = os.path.join(folder_path, preserve_file)
            print(f"Preserved: {preserve_file_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
folder_path = "/work/u5832291/yixian/TarDAL_edit/experiments/tardal_ct/20231130_default_without_pretrained_fusionnet"
preserve_files = ["00274-0.1594.pth", "00300.pth", "meta.txt"]

delete_files_except(folder_path, preserve_files)
