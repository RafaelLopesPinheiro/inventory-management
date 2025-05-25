import pandas as pd

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def transform(self) -> pd.DataFrame:
        df = self.df.copy()
        df['day'] = df['order_date'].dt.day
        df['month'] = df['order_date'].dt.month
        df['dayofweek'] = df['order_date'].dt.dayofweek

        grouped = df.groupby(['order_date', 'pizza_name'])['quantity'].sum().reset_index()
        pivoted = grouped.pivot(index='order_date', columns='pizza_name', values='quantity').fillna(0)
        pivoted = pivoted.reset_index()
        return pivoted
