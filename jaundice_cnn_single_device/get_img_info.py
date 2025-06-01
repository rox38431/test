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

if __name__ == "__main__":
    raw_img_dir = '/home/ngroup/TFG-Students/jojo_code/correction_test/iHealthImages'
    base_dir = Path(raw_img_dir)
    found_exts = check_subfolders_for_non_images(base_dir)

    print("\n整個資料夾中找到的影像檔案類型：")
    for ext in sorted(found_exts):
        print(f"  - {ext}")
