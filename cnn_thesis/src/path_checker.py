import os

def check_path(test_path):
    print(f"\nChecking path: {test_path}")
    
    # Check if the path exists
    exists = os.path.exists(test_path)
    print(f"Path exists: {exists}")
    
    # Break down the path
    dirname = os.path.dirname(test_path)
    basename = os.path.basename(test_path)
    parent_dir = os.path.basename(dirname)
    
    print(f"\nPath breakdown:")
    print(f"Directory: {dirname}")
    print(f"File name: {basename}")
    print(f"Parent folder: {parent_dir}")
    
    # Check if parent directory exists
    if os.path.exists(dirname):
        print(f"\nParent directory exists and contains:")
        try:
            files = os.listdir(dirname)
            for f in files[:5]:  # Show first 5 files
                print(f"- {f}")
            if len(files) > 5:
                print(f"... and {len(files)-5} more files")
        except PermissionError:
            print("Permission denied to list directory contents")
    else:
        print("\nParent directory does not exist")

if __name__ == "__main__":
    # Test path
    test_path = "/teamspace/studios/this_studio/images/Winde/11VTR/F_FOTO_WINDE_5dcbebcbbdc1d900019dbec5.jpg"
    
    check_path(test_path)
    
    # Also check current working directory
    print("\nCurrent working directory:")
    print(os.getcwd())