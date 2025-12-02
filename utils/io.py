import pandas as pd
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_excel(output_path, sheets_dict):
    with pd.ExcelWriter(output_path) as writer:
        for name, df in sheets_dict.items():
            df.to_excel(writer, sheet_name=name, index=False)
