import csv
from pathlib import Path


FILE_LIST_CSV_PATH = '/Users/jojo/Downloads/NTUH_document/code/test/jaundice_cnn_single_device/data/file_list.csv'
LAB_CSV_PATH = './result/NTUH_patient_table_with_tbil_0608.csv'

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

def load_lab_data(csv_path: str):
    """
    Load lab data from csv
    """
    lab_data = []
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # print(row.keys())
            if (row['拍照流水號'].strip() == ''):
                continue
            lab_data.append({
                'patient_camera_date': row['拍照流水號'],
                'tbil_val': row["T-bil value\u2028(收案日前)"],
                'tbil_date': row["T-bil date\u2028(收案日前)"]
            })
    return lab_data

def get_tbil_value(lab_data, foldername):
    for row in lab_data:
        if (foldername == row["patient_camera_date"]):
            tbil_val = row['tbil_val']
            if (tbil_val.strip() != ''):
                return float(tbil_val)
    return -1

def load_img_list(csv_path: str):
    """
    Load image list data from csv
    """
    image_list = []

    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            foldername = row['foldername']
            tbil_value = get_tbil_value(lab_data, foldername)

            if (tbil_value == '-1'):
                tbil_value = ''

            image_list.append({
                'filename': row['filename'],
                'foldername': row['foldername'],
                'filepath': row['filepath'],
                'tbil_value': str(tbil_value)
            })

    return image_list



if __name__ == '__main__':
    lab_data = load_lab_data(LAB_CSV_PATH)
    img_list = load_img_list(FILE_LIST_CSV_PATH)
    for data in lab_data:
        print(data)
