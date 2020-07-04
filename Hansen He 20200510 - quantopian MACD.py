#Imports
from quantopian.pipeline.data.builtin import USEquityPricing
import statsmodels.api as sm 
import quantopian.pipeline.data 
import numpy as np
import pandas as pd
import talib
import scipy

def initialize(context):

    set_benchmark(symbol('DBC'))
    context.DBC = symbol('DBC')
    # context.allocation = 1
    
    # context.TakeProfitPct = 0.25
    # context.StopLossPct = 0.05
    # context.BuyPrice = 0
    
    context.bought = False
    context.sold = False

    # 30 min scheduler
    # for x in [0,1,2,3,4,5]:
        # schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=x, minutes=29))
        # schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=x, minutes=59))
        
    # daily scheduler 
    schedule_function(my_rebalance_close, date_rules.every_day(), time_rules.market_close(minutes=5))
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close(minutes=5))
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_close(minutes=5))
    
    #Set commission and slippage
    set_commission(commission.PerTrade(cost=1)) 
    set_slippage(slippage.FixedSlippage(spread=0.01))

def my_rebalance(context,data):
    DBC_prices = data.history(context.DBC, "price", 100, "1d")
   
    ema12 = talib.EMA(DBC_prices,12)
    ema26 = talib.EMA(DBC_prices,26)
    macd = ema12 - ema26
    signal = talib.EMA(macd,9)
    
    record(MACD_val=macd[-1] - signal[-1])
    record(macd=macd[-1])
    record(exp3=signal[-1])
    
    if macd[-2] < signal[-2] and macd[-1] > signal[-1] and not context.bought:
        order(context.DBC, 40)
        log.info("------------------------------------")
        log.info("Opened Long Position; MACD_val: %.3f; PNL: %.3f; Market Value: %.5f" % (macd[-1]-signal[-1], context.portfolio.pnl, context.portfolio.portfolio_value))
        # log.info(context.portfolio.positions)
        context.boughtPrice = data.current(context.DBC, 'price')
        context.bought = True
        context.sold = False
        
    if macd[-2] > signal[-2] and macd[-1] < signal[-1] and not context.sold:
        order(context.DBC, -40)
        log.info("------------------------------------")
        log.info("Opened Short Position; MACD_val: %.5f; PNL: %.5f; Market Value: %.5f" % (macd[-1]-signal[-1], context.portfolio.pnl, context.portfolio.portfolio_value))
        # log.info(context.portfolio.positions)
        context.boughtPrice = data.current(context.DBC, 'price')
        context.bought = False
        context.sold = True

def my_record_vars(context, data):
    DBC_prices = data.history(context.DBC, "price", 100, "1d")
   
    ema12 = talib.EMA(DBC_prices,12)
    ema26 = talib.EMA(DBC_prices,26)
    macd = ema12 - ema26
    signal = talib.EMA(macd,9)
    
    if context.bought or context.sold:
        log.info("Holding position; MACD_val: %.5f; PNL: %.5f; Market Value: %.5f" % (macd[-1]-signal[-1], context.portfolio.pnl, context.portfolio.portfolio_value))
        # log.info(context.portfolio.positions)
    # leverage = context.account.leverage
    # record(leverage=leverage)
    
def my_rebalance_close(context, data): 
    #If we have a position check sell conditions
    DBC_prices = data.history(context.DBC, "price", 100, "1d")
   
    ema12 = talib.EMA(DBC_prices,12)
    ema26 = talib.EMA(DBC_prices,26)
    macd = ema12 - ema26
    signal = talib.EMA(macd,9)
    
    if macd[-2] > signal[-2] and macd[-1] < signal[-1] and context.portfolio.positions[context.DBC].amount != 0 and context.bought:
        order_target_percent(context.DBC, 0)
        log.info("Closed Long Position; MACD_val: %.5f; PNL: %.5f; Market Value: %.5f" % (macd[-1]-signal[-1], context.portfolio.pnl, context.portfolio.portfolio_value))
        # log.info(context.portfolio.positions)
        context.bought = False
            
    if macd[-2] < signal[-2] and macd[-1] > signal[-1] and context.portfolio.positions[context.DBC].amount != 0 and context.sold:
        order_target_percent(context.DBC, 0)
        log.info("Closed Short Position; MACD_val: %.5f; PNL: %.5f; Market Value: %.5f" % (macd[-1]-signal[-1], context.portfolio.pnl, context.portfolio.portfolio_value))
        # log.info(context.portfolio.positions)
        context.sold = False
  