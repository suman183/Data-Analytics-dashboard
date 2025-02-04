import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class StockVisualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        
    def plot_predictions(self, true_values, predictions, dates=None):
        plt.figure(figsize=self.figsize)
        if dates is None:
            plt.plot(true_values, label='Actual', color='blue')
            plt.plot(predictions, label='Predicted', color='red')
        else:
            plt.plot(dates, true_values, label='Actual', color='blue')
            plt.plot(dates, predictions, label='Predicted', color='red')
            
        plt.title('Stock Price Prediction vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt
    
    def plot_training_history(self, history):
        plt.figure(figsize=self.figsize)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        return plt
    
    def plot_stock_analysis(self, data):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Price plot
        ax1.plot(data.index, data['close'], label='Close Price')
        ax1.set_title('Stock Price Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # Volume plot
        ax2.bar(data.index, data['volume'], color='gray', alpha=0.5)
        ax2.set_title('Trading Volume')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        plt.tight_layout()
        return plt