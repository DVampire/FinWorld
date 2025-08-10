import numpy as np

def buy(cash, position, price, fee_ratio):
    buy_position = int(np.floor(cash / (price * (1 + fee_ratio))))
    position += buy_position
    cash -= buy_position * price * (1 + fee_ratio)
    return cash, position

def sell(cash, position, price, fee_ratio):
    cash += position * price * (1 - fee_ratio)
    position = 0
    return cash, position

def hold(cash, position, price, fee_ratio):
    return cash, position

def value(cash, position, price):
    return cash + position * price