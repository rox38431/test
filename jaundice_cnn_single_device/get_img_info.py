from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', '.dng', '.raw', '.heic'}

def check_subfolders_for_non_images(base_dir: Path):
    found_image_exts = set()

    for subfolder in base_dir.iterdir():
        if not subfolder.is_dir():
            continue
        invalid_items = []
        for item in subfolder.iterdir():
            if item.is_dir():
                invalid_items.append(f'資料夾: {item.name}')
            elif item.suffix.lower() not in IMAGE_EXTS:
                invalid_items.append(f'非影像檔案: {item.name}')
            else:
                found_image_exts.add(item.suffix.lower())
        if invalid_items:
            print(f"{subfolder.name} 有以下非影像檔案或資料夾：")
            for invalid in invalid_items:
                print(f"  - {invalid}")
    return found_image_exts

def check_subfolders_for_non_images(base_dir: Path):

    for subfolder in base_dir.iterdir():
        if not subfolder.is_dir():
            continue
        for item in subfolder.iterdir():
            if item.is_dir() or item.suffix.lower() not in IMAGE_EXTS:
                continue
            else:
                found_image_exts.add(item.suffix.lower())

def save_filenames_and_paths(base_dir: Path, output_csv: str):
    file_entries = [
        (p.name, str(p.resolve()))
        for p in base_dir.iterdir()
        if p.is_file()
    ]

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'filepath'])  # 標頭
        writer.writerows(file_entries)

    print(f"✅ 共寫入 {len(file_entries)} 筆檔案資訊至 {output_csv}")
            

if __name__ == "__main__":
    raw_img_dir = '/home/ngroup/TFG-Students/jojo_code/correction_test/iHealthImages'
    output_csv_path = './data/file_list.csv'
    base_dir = Path(raw_img_dir)
    found_exts = check_subfolders_for_non_images(base_dir)
    save_filenames_and_paths(base_dir, output_csv_path)

    print("\n整個資料夾中找到的影像檔案類型：")
    for ext in sorted(found_exts):
        print(f"  - {ext}")
