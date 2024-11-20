import json
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from arch import arch_model
from arch.univariate.base import ARCHModelResult, ARCHModelForecast

from model.index import Index

DEFAULT_LOOKUPS = 365
DEFAULT_END = datetime.today()
DEFAULT_DATE_FORMAT = '%Y-%m-%d'
LINES = '-----------------------------------------------------------'
ADJUSTED_CLOSE = 'Adjusted close'
GARCH = "GARCH"
DAILY_LOG_RETURN = 'Daily Log Return'


class Executor:

    def __init__(self, config: dict):
        self.previsions: list[ARCHModelForecast] = []
        if not isinstance(config, dict):
            print("please pass a dictionary")
        self.indexes: list[Index] = []
        self.processes: list[ARCHModelResult] = []
        self.config: dict = config
        self.__inject_data_from_yahoo()
        self.setup_daily_log_return()
        self.setup_volatility()

    def __inject_data_from_yahoo(self) -> None:

        try:
            lookup_days = self.config.get('lookup_days', DEFAULT_LOOKUPS)
            end_date = self.config.get('end_date', DEFAULT_END)
            date_format = self.config.get('date_format', DEFAULT_DATE_FORMAT)
            start_date = (end_date - timedelta(days=lookup_days)).strftime(date_format)
            end_date = end_date.strftime(date_format)
            index_list = self.config.get('indexes', [])
            for index in index_list:
                from_yahoo = yf.download(
                    index_list[index],
                    start=start_date,
                    end=end_date
                )
                from_yahoo = pd.DataFrame({key[0]: value for key, value in from_yahoo.items()})
                from_yahoo.reset_index(inplace=True)
                from_yahoo.rename(columns={'index': 'Date'}, inplace=True)
                from_yahoo.rename(columns={'Adj Close': ADJUSTED_CLOSE}, inplace=True)
                from_yahoo['Date'] = from_yahoo['Date'].dt.date
                index_bean = Index(index_name=index,
                                   index_ticker=index_list[index],
                                   data=from_yahoo)
                self.indexes.append(index_bean)

        except json.JSONDecodeError:
            print("Error in decoding json")
            exit(-1)
        except AttributeError:
            print("some attribute is None")
            exit(-1)

    def setup_daily_log_return(self):

        for index in self.indexes:
            index.daily_log_return = pd.DataFrame(index.df['Date'])
            index.daily_log_return[DAILY_LOG_RETURN] = 100 * np.log(
                index.df[ADJUSTED_CLOSE] / index.df[ADJUSTED_CLOSE].shift(1)
            )
            index.daily_log_return.set_index('Date', inplace=True)
            index.daily_log_return.pct_change().dropna()

    def setup_volatility(self, window=30):
        for index in self.indexes:
            dummy = pd.DataFrame(index.df[[ADJUSTED_CLOSE, 'Date']])
            dummy[ADJUSTED_CLOSE].pct_change().dropna()
            dummy.set_index('Date', inplace=True)
            index.volatility = dummy[ADJUSTED_CLOSE].rolling(window=window).std().dropna()

    def print_indexes(self):
        print(LINES)
        print("Analyzing this index list: {")
        for index in self.indexes:
            print(f'\t{index}')
        print(f'}}\n{LINES}')

    def exec(self):
        models = []
        for index in self.indexes:
            model = arch_model(index.daily_log_return[DAILY_LOG_RETURN].dropna(),
                                     vol=GARCH, p=1, q=1, dist='normal')
            models.append(model)
            self.processes.append(model.fit())

        self.__prevision()

    def show(self):
        print(LINES)
        print("process summary:")
        for process in self.processes:
            print(process.summary())

        print(LINES)
        print("prediction summary:")
        for prevision in self.previsions:
            print(prevision.variance[-1:])

    def __prevision(self):
        for process in self.processes:
            self.previsions.append(process.forecast(horizon=5))


