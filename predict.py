import pandas_datareader.data as web
import seaborn as sb
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn import model_selection

# To ignore a warning.
pd.options.mode.chained_assignment = None

class Predict:
    '''
    This class will predict the closing price of a stock.
    '''
    def __init__(self, stock_name, start, end):
        self.stock_name = stock_name
        self.start = start
        self.end = end
        self.data = None
        self.prediction = None
        self.res_df = None
        self.model = None
        # get the model
        self.model = joblib.load('trained_linear_regression_model')
        self.X = None
        self.y = None
    
    def show_correlation(self):
        corr = self.data.corr(method='pearson')
        sb.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap='Greys')
        plt.show()
    
    def get_X_y(self):
        # get data set from yahoo
        self.data = web.DataReader(self.stock_name, 'yahoo', start=self.start, end=self.end)
        # to access Date as column
        self.data.reset_index(inplace=True)
        # make a new df containing Date as day, month and year
        Xy_df = self.data[['Date','High', 'Low', 'Open', 'Close']]
        Xy_df = self.manage_dates(Xy_df)
        X = Xy_df.iloc[:, 0:-1]
        y = Xy_df['Close']
        return X, y
    
    def manage_dates(self, Xy_df):
        # convert Date cols to day, month, year.
        Xy_df['Day'] = Xy_df['Date'].dt.day
        Xy_df['Month'] = Xy_df['Date'].dt.month
        Xy_df['Year'] = Xy_df['Date'].dt.year
        Xy_df = Xy_df[['Day', 'Month','Year', 'High', 'Low', 'Open', 'Close']]
        return Xy_df
    def predict(self):
        # get input, output datasets.
        self.X, self.y = self.get_X_y()
        # get the prediction of X.
        self.prediction = self.model.predict(self.X)
    def visualize(self):
        self.res_df = pd.DataFrame({'Actual': self.data['Close'], 'Predicted': self.prediction})
        plt.figure(figsize=(16,8))
        plt.style.use('fivethirtyeight')
        plt.plot(self.data['Date'], self.data['Close'], c='b', label='actual')
        plt.plot(self.data['Date'], self.prediction, c='r', label='predicted')
        plt.xlabel('Date')
        plt.ylabel('Closing price in USD')
        plt.title(self.stock_name + ' stock prediction')
        plt.legend()
        plt.show()
    def get_model_score(self):
        return self.model.score(self.X, self.y) * 100
    def get_kfold_score(self):
        kfold = model_selection.KFold(n_splits=20, shuffle=True, random_state=100)
        results_kfold = model_selection.cross_val_score(self.model, self.X, self.y.astype('int'), cv=kfold)
        kfold_score = results_kfold.mean() * 100
        return kfold_score

if __name__ == "__main__":
    stock_name = input('Enter the stock name(Yahoo): ')
    start = input('Enter starting date(yyyy-mm-dd), eg-2017-01-21: ')
    end = input('Enter starting date(yyyy-mm-dd), eg-2017-01-21: ')
    predict = Predict(stock_name, start, end)
    predict.predict()
    predict.show_correlation()
    predict.visualize()
    print('regular score: ', predict.get_model_score())
    print('kfold score: ', predict.get_kfold_score())
    input('Hit Return(Enter) to exit. ')

