import pandas as pd

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        df = pd.read_excel(self.file_path)
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['pizza_name'] = df['pizza_name'].astype(str)
        return df
