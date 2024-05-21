from pathlib import Path
import glob
import shutil

def find_root(root_file_name="paths.py"):
    """
    Find the root of the project by looking for a file named root_file_name.
    
    Parameters:
    root_file_name (str): The name of the file to look for (default is "paths.py").
    
    Returns:
    str: The path to the directory containing root_file_name, or None if not found.
    """
    current_path = Path.cwd()
    root_path = current_path.root

    while current_path != root_path:
        if (current_path / root_file_name).is_file():
            return str(current_path)
        current_path = current_path.parent

    return None

def clean_gams_temp_files():
    """
    Clean up temporary files and folders created by GAMS.
    """
    for dir in glob.glob("225*/"):
        shutil.rmtree(dir)