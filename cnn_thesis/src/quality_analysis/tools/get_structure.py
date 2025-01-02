import os

def is_visible(name):
    return not name.startswith('.')

def scan_directory(directory, output_file, prefix=''):
    try:
        contents = sorted(filter(is_visible, os.listdir(directory)))
        files = []
        dirs = []

        for item in contents:
            full_path = os.path.join(directory, item)
            if os.path.isfile(full_path):
                files.append(item)
            elif os.path.isdir(full_path):
                dirs.append(item)

        # Process directories
        dir_count = len(dirs)
        if dir_count > 5:
            dirs_to_display = dirs[:3]
            dirs_to_display.append(f"... {dir_count - 3} more subfolders")  # Indicate truncation
        else:
            dirs_to_display = dirs

        for i, d in enumerate(dirs_to_display):
            is_last = (i == len(dirs_to_display) - 1 and len(files) == 0)
            output_file.write(f"{prefix}{'└── ' if is_last else '├── '}{d}/\n")
            new_prefix = prefix + ('    ' if is_last else '│   ')

            # Skip specified folders
            if isinstance(d, str) and d in ['experiments', 'taguchi_experiments', 'output_files', '__pycache__', 'check_gpu', 'global_image_cache']:
                continue

            # Continue only if `d` is not a truncation message
            if not isinstance(d, str) or not d.startswith("..."):
                scan_directory(os.path.join(directory, d), output_file, new_prefix)

        # Process files
        file_count = len(files)
        if file_count > 100:
            files_to_display = files[:5]
            files_to_display.append(f"... {file_count - 5} more files")  # Indicate truncation
        else:
            files_to_display = files

        for i, f in enumerate(files_to_display):
            is_last = (i == len(files_to_display) - 1)
            output_file.write(f"{prefix}{'└── ' if is_last else '├── '}{f}\n")
    except PermissionError:
        output_file.write(f"{prefix}[Permission Denied]\n")
    except Exception as e:
        output_file.write(f"{prefix}[Error: {str(e)}]\n")

def create_structure_file(root_directory, output_filename):
    with open(output_filename, 'w') as output_file:
        output_file.write(f"{os.path.basename(root_directory)}/\n")
        scan_directory(root_directory, output_file, '')

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure the output_files directory exists
    output_dir = os.path.join(script_dir, 'output_files')
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file in the output_files folder
    output_filename = os.path.join(output_dir, 'structure.txt')
    
    parent_dir = os.path.dirname(script_dir)
    
    create_structure_file(parent_dir, output_filename)
    print(f"Structure file '{output_filename}' has been created for {parent_dir}")
