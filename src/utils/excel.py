from pathlib import Path
from typing import Union

import pandas as pd

from utils.misc import PathUtils


class ExcelUtils:
    EXCEL_CACHE = {}

    @classmethod
    def get_session_data(cls, session_name: str, sheet: str = 'Apr_May 2025', excel_file_path: Union[str, Path] = 'Capture Sessions.xls') -> pd.Series:
        if not Path(excel_file_path).exists():
            excel_file_path = PathUtils.resources_path() / str(excel_file_path)
        excel_file_path = Path(excel_file_path).absolute()
        assert excel_file_path.exists(), f"Excel file {excel_file_path} does not exist. Please provide a valid path."
        cache_key = str(excel_file_path) + f'_{sheet}'
        if cache_key not in cls.EXCEL_CACHE:
            # Read the Excel file
            cls.EXCEL_CACHE[cache_key] = pd.read_excel(str(excel_file_path), sheet_name=sheet)
        return cls.EXCEL_CACHE[cache_key][cls.EXCEL_CACHE[cache_key]["Session Name"] == session_name].iloc[0]