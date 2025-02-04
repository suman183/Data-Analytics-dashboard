import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class StockDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.data = None
        self.processed_data = None
        
    def load_data(self, data):
        """Load and preprocess the stock data"""
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = pd.DataFrame(data)
            
        expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'Name']
        if not all(col in self.data.columns for col in expected_columns):
            raise ValueError(f"Data must contain columns: {expected_columns}")
            
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)
        self.data.sort_index(inplace=True)
        return self.data
    
    def prepare_data(self, sequence_length=5):
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        scaled_data = self.scaler.fit_transform(self.data[numeric_columns])
        self.processed_data = pd.DataFrame(
            scaled_data,
            columns=numeric_columns,
            index=self.data.index
        )
        return self.processed_data
    
    def create_sequences(self, sequence_length=5):
        X, y = [], []
        data = self.processed_data.values
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, 3])
        return np.array(X), np.array(y)
    
    def get_train_test_split(self, sequence_length=5, test_size=0.2):
        X, y = self.create_sequences(sequence_length)
        return train_test_split(X, y, test_size=test_size, shuffle=False)