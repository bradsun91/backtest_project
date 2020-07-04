#Imports
from quantopian.pipeline.data.builtin import USEquityPricing
from scipy.optimize import minimize
import statsmodels.api as sm 
import quantopian.pipeline.data 
import numpy as np
import pandas as pd
import talib
import scipy

def initialize(context):

    # set_benchmark(symbol('DBC'))
    context.DBC = symbol('DBC')
    context.SPY = symbol('SPY')
    context.IEI = symbol('IEI')
    context.GLD = symbol('GLD')
    context.IEF = symbol('IEF')
    context.TLT = symbol('TLT')
 
    context.tickers = [context.DBC, context.SPY, context.IEI, context.GLD, context.IEF, context.TLT]
    
    context.weights = {}
    
    context.bought = {}
    context.sold = {}
    for ticker in context.tickers:
        context.bought[ticker]= False
        context.sold[ticker]=False
    
    context.boughtPrice = {}
    
    # 30 min scheduler
    # for x in [0,1,2,3,4,5]:
        # schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=x, minutes=29))
        # schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=x, minutes=59))
    
    # scheduler 
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_close(minutes=10))
    schedule_function(separate, date_rules.every_day(), time_rules.market_close(minutes=10))
    schedule_function(close_position, date_rules.every_day(), time_rules.market_close(minutes=10))
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close(minutes=10))
    schedule_function(open_position, date_rules.every_day(), time_rules.market_close(minutes=10))
    schedule_function(daily, date_rules.every_day(), time_rules.market_close())
    
    #Set commission and slippage
    set_commission(commission.PerTrade(cost=1)) 
    set_slippage(slippage.FixedSlippage(spread=0.05))
  
def separate(context, data):
    log.info("-------------------------------------------------------")

def daily(context, data):
    log.info("Weights: DBC %.5f, SPY %.5f, IEI %.5f, GLD %.5f, IEF %.5f, TLT %.5f" % (context.weights[context.DBC], context.weights[context.SPY], context.weights[context.IEI], context.weights[context.GLD], context.weights[context.IEF], context.weights[context.TLT]))
    log.info("Amount: DBC %.5f, SPY %.5f, IEI %.5f, GLD %.5f, IEF %.5f, TLT %.5f" % (context.portfolio.positions[context.DBC].amount, context.portfolio.positions[context.SPY].amount, context.portfolio.positions[context.IEI].amount, context.portfolio.positions[context.GLD].amount, context.portfolio.positions[context.IEF].amount, context.portfolio.positions[context.TLT].amount))
    
    log.info("Prices: DBC %.5f, SPY %.5f, IEI %.5f, GLD %.5f, IEF %.5f, TLT %.5f" % (data.current(context.DBC,'price'), data.current(context.SPY,'price'), data.current(context.IEI,'price'), data.current(context.GLD,'price'), data.current(context.IEF,'price'), data.current(context.TLT,'price')))
    
    DBC = context.portfolio.positions[context.DBC].amount * data.current(context.DBC,'price')
    SPY = context.portfolio.positions[context.SPY].amount * data.current(context.SPY,'price')
    IEI = context.portfolio.positions[context.IEI].amount * data.current(context.IEI,'price')
    GLD = context.portfolio.positions[context.GLD].amount * data.current(context.GLD,'price')
    IEF = context.portfolio.positions[context.IEF].amount * data.current(context.IEF,'price')
    TLT= context.portfolio.positions[context.TLT].amount * data.current(context.TLT,'price')
    
    log.info("Market Value: DBC %.5f, SPY %.5f, IEI %.5f, GLD %.5f, IEF %.5f, TLT %.5f" % (DBC, SPY, IEI, GLD, IEF, TLT))
    log.info("Portfolio Value: %.5f" % (context.portfolio.portfolio_value))

def rebalance(context, data):
    # check if it is January
    if get_datetime().month not in [1]:
        return
    
    # read and produce returns dataframe
    context.returns_df = pd.DataFrame()
    for ticker in context.tickers:
        prices = data.history(ticker, "price", 252, "1d")
        df = pd.DataFrame(data={ticker:prices})
        context.returns_df = pd.concat([context.returns_df, df],axis=1)
        
    context.returns_df.dropna(inplace=True)
    pct = context.returns_df
  
    #协方差矩阵
    cov_mat = pct.cov()
    # log.info(cov_mat)
    if not isinstance(cov_mat, pd.DataFrame):
        raise ValueError('cov_mat should be pandas DataFrame！')

    omega = np.matrix(cov_mat.values)  # 协方差矩阵

    a, b = np.linalg.eig(np.array(cov_mat)) #a为特征值,b为特征向量
    a = np.matrix(a)
    b = np.matrix(b)
    # 定义目标函数

    def fun1(x):
        tmp = (omega * np.matrix(x).T).A1
        risk = x * tmp/ np.sqrt(np.matrix(x) * omega * np.matrix(x).T).A1[0]
        delta_risk = [sum((i - risk)**2) for i in risk]
        return sum(delta_risk)

    # 初始值 + 约束条件 
    x0 = np.ones(omega.shape[0]) / omega.shape[0]  
    bnds = tuple((0,None) for x in x0)
    cons = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
    options={'disp':False, 'maxiter':1000, 'ftol':1e-20}

    try:
        res = minimize(fun1, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    except:
        raise ValueError('method error！！！')

    # 权重调整
    if res['success'] == False:
        # print res['message']
        pass
    wts = pd.Series(index=cov_mat.index, data=res['x'])

    # weight adjust
    wts = wts / wts.sum()

    risk = pd.Series(wts * (omega * np.matrix(wts).T).A1 / np.sqrt(np.matrix(wts) * omega * np.matrix(wts).T).A1[0],index = cov_mat.index)
    risk[risk<0.0] = 0.0

    context.weights=wts

def _get_macd(data, ticker):
    prices = data.history(ticker, "price", 100, "1d")
    prices.dropna(inplace=True)

    ema12 = talib.EMA(prices,12)
    ema26 = talib.EMA(prices,26)
    macd = ema12 - ema26
    signal = talib.EMA(macd,9)
    
    return macd, signal

def open_position(context,data):
    for ticker in context.tickers:
        macd, signal = _get_macd(data, ticker)

        if macd[-1] > signal[-1] and not context.bought[ticker]:
            order_target_percent(ticker, context.weights[ticker])
           
            context.boughtPrice[ticker] = data.current(ticker, 'price')
            context.bought[ticker] = True
            
            log.info("Opened Long %s Position; MACD_val: %.5f; Open Position Price: %.5f; Bought: %s; Sold: %s" % (ticker, macd[-1]-signal[-1], context.boughtPrice[ticker], context.bought[ticker], context.sold[ticker]))

        if macd[-1] < signal[-1] and not context.sold[ticker]:
            order_target_percent(ticker, -context.weights[ticker])
               
            context.boughtPrice[ticker] = data.current(ticker, 'price')
            context.sold[ticker] = True
            
            log.info("Opened Short %s Position; MACD_val: %.5f; Open Position Price: %.5f; Bought: %s; Sold: %s" % (ticker, macd[-1]-signal[-1], context.boughtPrice[ticker], context.bought[ticker], context.sold[ticker]))
            
def my_record_vars(context, data):
    for ticker in context.tickers:
        macd,signal = _get_macd(data, ticker)

        if context.bought[ticker] or context.sold[ticker]:
            log.info("Holding %s position; MACD_val: %.5f; Bought: %s; Sold: %s" % (ticker, macd[-1]-signal[-1], context.bought[ticker], context.sold[ticker]))
      
    
def close_position(context, data): 
    for ticker in context.tickers:
        #If we have a position check sell conditions
        macd, signal = _get_macd(data, ticker)

        if macd[-1] < signal[-1] and context.portfolio.positions[ticker].amount != 0 and context.bought[ticker]:
            order_target_percent(ticker, 0)
            
            context.bought[ticker] = False
           
            log.info("Closed Long %s Position; MACD_val: %.5f; Close Position Price: %.5f; Bought: %s; Sold: %s" % (ticker, macd[-1]-signal[-1], data.current(ticker,'price'), context.bought[ticker], context.sold[ticker]))

        if macd[-1] > signal[-1] and context.portfolio.positions[ticker].amount != 0 and context.sold[ticker]:
            order_target_percent(ticker, 0)

            context.sold[ticker] = False
            
            log.info("Closed Short %s Position; MACD_val: %.5f; Close Position Price: %.5f; Bought: %s; Sold: %s" % (ticker, macd[-1]-signal[-1], data.current(ticker,'price'), context.bought[ticker], context.sold[ticker]))