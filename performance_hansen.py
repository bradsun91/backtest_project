import pandas as pd
import numpy as np

data = pd.read_csv('df_equity_data.csv')

def performance(data):
    data['daily_PL'] = data['total']-data['total'].shift(1)

    trade = False
    profits = []
    total_profit_per_trade = 0

    for i in range(len(data)): 
        if data.iloc[i]['market_value'] == 0.0 and data.iloc[i]['returns'] == 0.0 and trade:
            trade = False
        
            profits.append(total_profit_per_trade)
            total_profit_per_trade = 0

        if trade:
            total_profit_per_trade += data.iloc[i]['daily_PL']

        if data.iloc[i]['market_value'] != 0.0 and not trade:
            trade = True
            total_profit_per_trade += data.iloc[i]['daily_PL']

    win_trade = list(filter(lambda x: x >0, profits))
    loss_trade = list(filter(lambda x: x <0, profits))
    num_win_trade = len(win_trade)
    num_total_trade = len(profits)

    print('win %:', num_win_trade/num_total_trade*100)
    print('PL Ratio:', np.mean(win_trade)/(-np.mean(loss_trade)))
    
    return num_win_trade/num_total_trade*100, np.mean(win_trade)/(-np.mean(loss_trade))

performance(data)
print(data.head())