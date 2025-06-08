import csv
from pathlib import Path

# Supported image file extensions
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.gif',
    '.tiff', '.webp', '.dng', '.raw', '.heic'
}

def validate_image_subfolders(root_dir: Path):
    """
    Scan each immediate subfolder under root_dir
    Report folders or files that are not valid image files
    Return a set of image file extensions that are actually found
    """
    found_extensions = set()

    for subfolder in root_dir.iterdir():
        if not subfolder.is_dir():
            continue

        non_image_entries = []

        for entry in subfolder.iterdir():
            if entry.is_dir():
                non_image_entries.append(f"Folder {entry.name}")
            elif entry.suffix.lower() not in IMAGE_EXTENSIONS:
                non_image_entries.append(f"Non image file {entry.name}")
            else:
                found_extensions.add(entry.suffix.lower())

        if non_image_entries:
            print(f"{subfolder.name} contains non image files or folders")
            for entry in non_image_entries:
                print(f"  {entry}")

    return found_extensions

def export_image_file_list(root_dir: Path, csv_path: str):
    """
    Recursively find all image files in root_dir
    Write filename, folder name, and absolute path to a CSV file
    """
    image_files = [
        (file.name, file.parent.name, str(file.resolve()))
        for file in root_dir.rglob('*')
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'foldername', 'filepath'])
        writer.writerows(image_files)

    print(f"{len(image_files)} image files written to {csv_path}")

def load_image_file_list(csv_path: str):
    """
    Load image file list from a CSV file
    Returns a list of dictionaries with keys: filename, foldername, filepath
    """
    image_list = []

    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_list.append({
                'filename': row['filename'],
                'foldername': row['foldername'],
                'filepath': row['filepath']
            })

    return image_list


if __name__ == "__main__":
    image_root_dir = '/home/ngroup/TFG-Students/jojo_code/correction_test/iHealthImages'
    output_csv_path = './data/file_list.csv'
    root_path = Path(image_root_dir)

    detected_extensions = validate_image_subfolders(root_path)
    print(f"Image extensions found in {image_root_dir} are")
    for ext in sorted(detected_extensions):
        print(f"  {ext}")

    export_image_file_list(root_path, output_csv_path)

    # Load and preview image file list from CSV
    image_data = load_image_file_list(output_csv_path)
    print(f"Loaded {len(image_data)} entries from {output_csv_path}")
    print("First 3 entries:")
    for item in image_data[:3]:
        print(item)


    print("Done")
