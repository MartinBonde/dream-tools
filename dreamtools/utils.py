from pathlib import Path
import glob
import shutil

def find_root(root_file_name="paths.py", max_iter=20):
    """
    Find the root of the project by looking for a file named root_file_name.
    
    Parameters:
    root_file_name (str): The name of the file to look for (default is "paths.py").
    max_iter (int): Maximum number of parent directories to check before giving up.
    
    Returns:
    str: The path to the directory containing root_file_name, or None if not found.
    """
    current = Path.cwd()
    for _ in range(max_iter):
        if (current / root_file_name).exists():
            return str(current)
        current = current.parent
    return None

def clean_gams_temp_files():
    """
    Clean up temporary files and folders created by GAMS.
    """
    for dir in glob.glob("225*/"):
        shutil.rmtree(dir)