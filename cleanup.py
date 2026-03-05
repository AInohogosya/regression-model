"""
Cleanup script for the Sin(x) Regression Neural Network project.
Removes all generated files and directories to restore clean state.
"""

import os
import shutil
import glob


def cleanup_project():
    """Remove all generated files and directories"""
    print("Starting cleanup...")
    
    # Files and directories to remove
    cleanup_items = [
        './Model',
        './sample_data',
        '__pycache__',
        '*.pyc',
        '*_cache',
        '.pytest_cache',
        '*.log'
    ]
    
    removed_count = 0
    
    # Remove directories
    for item in cleanup_items:
        if item.startswith('./') and not item.endswith('*'):
            # Directory path
            if os.path.exists(item):
                try:
                    shutil.rmtree(item)
                    print(f"✅ Removed directory: {item}")
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Failed to remove directory {item}: {e}")
        elif '*' in item:
            # Pattern matching
            for path in glob.glob(item):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"✅ Removed directory: {path}")
                    else:
                        os.remove(path)
                        print(f"✅ Removed file: {path}")
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Failed to remove {path}: {e}")
    
    # Remove any remaining Python cache files
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and Model directory (should be removed above)
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'Model']
        
        for file in files:
            if file.endswith('.pyc') or file.endswith('.pyo'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"✅ Removed cache file: {file_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Failed to remove {file_path}: {e}")
    
    # Check for __pycache__ directories in subdirectories
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(cache_path)
                print(f"✅ Removed cache directory: {cache_path}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {cache_path}: {e}")
    
    print(f"\nCleanup completed! Removed {removed_count} items.")
    
    # Show what remains
    print("\nRemaining files:")
    remaining_files = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if not file.startswith('.'):
                rel_path = os.path.relpath(os.path.join(root, file), '.')
                remaining_files.append(rel_path)
    
    if remaining_files:
        for file in sorted(remaining_files):
            print(f"  📄 {file}")
    else:
        print("  (No remaining files)")


def main():
    """Main cleanup function"""
    print("=" * 60)
    print("SIN(X) REGRESSION NEURAL NETWORK - CLEANUP")
    print("=" * 60)
    print("This will remove all generated files and directories:")
    print("  - ./Model/ (trained model and results)")
    print("  - ./sample_data/ (generated CSV data)")
    print("  - __pycache__/ directories")
    print("  - *.pyc files")
    print()
    
    # Ask for confirmation
    response = input("Do you want to continue? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        cleanup_project()
    else:
        print("Cleanup cancelled.")


if __name__ == "__main__":
    main()
