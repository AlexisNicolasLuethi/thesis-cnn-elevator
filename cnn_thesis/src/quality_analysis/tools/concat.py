import os

def is_visible(name):
    return not name.startswith('.')

def is_python_file(name):
    return name.endswith('.py')

def collect_python_files(directory):
    python_files = []
    try:
        for root, _, files in os.walk(directory):
            # Skip hidden directories and specific directories
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue
            if any(excluded in root for excluded in ['experiments', 'taguchi_experiments', 'output_files', '__pycache__', 'check_gpu', 'global_image_cache']):
                continue
                
            # Collect visible Python files
            for file in files:
                if is_visible(file) and is_python_file(file):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, directory)
                    python_files.append((rel_path, full_path))
    
    except PermissionError:
        print(f"Permission denied for directory: {directory}")
    except Exception as e:
        print(f"Error scanning directory {directory}: {str(e)}")
    
    return sorted(python_files)  # Sort files for consistent output

def concatenate_python_files(root_directory, output_filename):
    python_files = collect_python_files(root_directory)
    
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write(f"Python files found in: {root_directory}\n")
        outfile.write("=" * 80 + "\n\n")
        
        for rel_path, full_path in python_files:
            try:
                with open(full_path, 'r', encoding='utf-8') as infile:
                    outfile.write(f"File: {rel_path}\n")
                    outfile.write("-" * 80 + "\n")
                    content = infile.read()
                    outfile.write(content)
                    if not content.endswith('\n'):
                        outfile.write('\n')
                    outfile.write("\n" + "=" * 80 + "\n\n")
                    
            except Exception as e:
                outfile.write(f"Error reading file {rel_path}: {str(e)}\n")
                outfile.write("=" * 80 + "\n\n")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure the output_files directory exists
    output_dir = os.path.join(script_dir, 'output_files')
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file in the output_files folder
    output_filename = os.path.join(output_dir, 'python_files_content.txt')
    
    # Start from the parent directory of the script
    parent_dir = os.path.dirname(script_dir)
    
    concatenate_python_files(parent_dir, output_filename)
    print(f"Python files have been concatenated into {output_filename}")