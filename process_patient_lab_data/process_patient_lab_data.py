import glob
import os
import pandas as pd
from datetime import datetime, timedelta

# === Configuration ===
PATIENT_INFO_PATH = '/Users/jojo/Downloads/NTUH_document/code/ihealth_portal_crawler/data/身體徵象資料(20250225)_嘉瑋_7.xlsx'
DATA_DIR = '/Users/jojo/Downloads/NTUH_document/code/ihealth_portal_crawler/data/20250311'
OUTPUT_PATH = './result/NTUH_patient_table_with_tbil_0608.csv'

# === Column Constants ===
PATIENT_CAMERA_DATE = 2  # Index for patient camera date column
PATIENT_INDEX = 3        # Index for patient ID column

# Lab test value columns and their date columns in patient info table
TBIL_COL, TBIL_DATE_COL = 56, 57
DBIL_COL, DBIL_DATE_COL = 60, 61
HB_COL, HB_DATE_COL = 64, 65
HBA1C_COL, HBA1C_DATE_COL = 68, 69

# === Helper Functions ===
def format_date(dt):
    """
    Format datetime object to 'YYYY/M/D' string.
    Return empty string if input is empty.
    """
    return '' if dt == '' else f'{dt.year}/{dt.month}/{dt.day}'

def parse_excel_datetime(text):
    """
    Parse Excel lab datetime string ("YYYY/MM/DD HH:MM").
    Return None if parsing fails.
    """
    try:
        return datetime.strptime(text, "%Y/%m/%d %H:%M")
    except Exception:
        return None

def parse_patient_table_date(text):
    """
    Extract and parse date from patient camera date string.
    Example format: "...-..._YYYYMMDD".
    Return None if parsing fails.
    """
    try:
        date_str = text.split('-')[-1].split('_')[-1]
        return datetime.strptime(date_str, "%Y%m%d")
    except Exception:
        return None

# === Parse patient info Excel and initialize patient data dictionary ===
def parse_patient_info(path):
    """
    Read patient info Excel and create nested dictionary keyed by patient ID and date.
    Initialize each entry with empty lab values and dates.
    """
    patient_data = {}
    df = pd.read_excel(path, header=None)

    for i, row in df.iterrows():
        # Skip first 3 rows and rows missing key fields
        if i < 3 or pd.isnull(row[PATIENT_INDEX]) or pd.isnull(row[PATIENT_CAMERA_DATE]):
            continue

        pid = str(row[PATIENT_INDEX])  # Patient ID
        date = parse_patient_table_date(str(row[PATIENT_CAMERA_DATE]))  # Extract date from camera date
        if not date:
            continue
        date_str = date.strftime('%Y/%m/%d')

        # Initialize dictionary entry if not exists
        if pid not in patient_data:
            patient_data[pid] = {}
        if date_str not in patient_data[pid]:
            patient_data[pid][date_str] = {
                'date': date,
                'tbil': '', 'tbil_date': '',
                'dbil': '', 'dbil_date': '',
                'hb': '', 'hb_date': '',
                'hba1c': '', 'hba1c_date': ''
            }

    return patient_data

def process_lab_value(patient_data, pid, lab_date, lab_value, val_key, date_key):
    """
    Update lab value for specified patient and date.
    Only update if lab_date is within one day after the camera date.
    Choose lab value with closest date to camera date.
    """
    if pid not in patient_data or lab_date is None:
        return

    for entry in patient_data[pid].values():
        # # Ignore lab values with date more than one day after the camera date
        # if lab_date > entry['date'] + timedelta(days=1):
        #     continue

        current_date = entry[date_key]
        # Update if no current date or new lab date is closer to camera date
        if current_date == '' or abs((entry['date'] - lab_date).total_seconds()) < abs((entry['date'] - current_date).total_seconds()):
            entry[val_key] = lab_value
            entry[date_key] = lab_date

def parse_lab_files(data_dir, patient_data):
    """
    Parse all lab files in data directory.
    Each file corresponds to one patient with multiple lab records and dates.
    Update patient_data dictionary with lab results by patient ID and lab date.
    """
    data_files = glob.glob(f'{data_dir}/*')
    for file_path in data_files:
        df = pd.read_excel(file_path, header=None)
        pid = os.path.splitext(os.path.basename(file_path))[0]  # Patient ID from filename

        # State machine for parsing lab file
        state = 'find'
        indices = {'tbil': [], 'dbil': [], 'hb': [], 'hba1c': []}
        keys = {
            'tbil': ['T-BIL(mg/dL)', 'T-BIL(STAT)(mg/dL)', 'T-BIL總膽紅素(mg/dL)'],
            'dbil': ['D-BIL(mg/dL)', 'D-BIL(STAT)(mg/dL)', 'D-BIL直接型膽色素(mg/dL)'],
            'hb': ['HB(g/dL)', 'Hb(急重症自行檢驗)(g/dL)'],
            'hba1c': ['HbA1c(%)', 'HbA1c糖化血色素(%)']
        }

        for i, row in df.iterrows():
            # End of data block detected by empty first column, reset state
            if state == 'process_data' and pd.isnull(row[0]):
                state = 'find'

            for j, cell in enumerate(row):
                if state == 'find' and pd.notnull(cell):
                    # Found start of new data block, start reading header
                    state = 'process_title'
                    for key in indices:
                        indices[key] = []
                    break

                elif state == 'process_title':
                    # Read header row and record lab test column indices
                    # if j == 0 or pd.isnull(cell):
                    #     continue
                    for key, candidates in keys.items():
                        if str(cell) in candidates:
                            indices[key].append(j)
                    # Header line ends at last column, switch to data reading state
                    if j == len(row) - 1:
                        state = 'process_data'

                elif state == 'process_data':
                    # Read data row
                    if j == 0:
                        dt = parse_excel_datetime(str(cell))  # Parse lab datetime
                    else:
                        if pd.isnull(cell):
                            continue
                        val = str(cell).strip()
                        # For each lab test, check if column matches and update patient data
                        for key in indices:
                            if j in indices[key]:
                                process_lab_value(patient_data, pid, dt, val, key, f'{key}_date')

if __name__ == '__main__':
    # Parse patient info and initialize data dictionary
    patient_data = parse_patient_info(PATIENT_INFO_PATH)

    # Parse lab files and update patient data
    parse_lab_files(DATA_DIR, patient_data)

    # Reload patient info Excel and update lab results in corresponding columns
    df = pd.read_excel(PATIENT_INFO_PATH, header=None)
    for i, row in df.iterrows():
        # Skip first 3 rows and rows missing patient ID or camera date
        if i < 3 or pd.isnull(row[PATIENT_INDEX]) or pd.isnull(row[PATIENT_CAMERA_DATE]):
            continue

        pid = str(row[PATIENT_INDEX])
        date = parse_patient_table_date(str(row[PATIENT_CAMERA_DATE]))
        if not date:
            continue
        date_str = date.strftime('%Y/%m/%d')

        # Update lab results if patient and date exist in dictionary
        if pid in patient_data and date_str in patient_data[pid]:
            data = patient_data[pid][date_str]
            df.at[i, TBIL_COL] = data['tbil']
            df.at[i, TBIL_DATE_COL] = format_date(data['tbil_date'])
            df.at[i, DBIL_COL] = data['dbil']
            df.at[i, DBIL_DATE_COL] = format_date(data['dbil_date'])
            df.at[i, HB_COL] = data['hb']
            df.at[i, HB_DATE_COL] = format_date(data['hb_date'])
            df.at[i, HBA1C_COL] = data['hba1c']
            df.at[i, HBA1C_DATE_COL] = format_date(data['hba1c_date'])

    # Drop first two rows (usually headers or notes)
    df = df.drop(index=[0, 1])

    # Ensure output directory exists and save the final CSV without index or header
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, header=False)
