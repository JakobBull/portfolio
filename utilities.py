
def calculate_fees(price, number):
    trade_cost = 7.9 + 0.0025*price*number

    if trade_cost < 12.9:
        trade_cost=12.9
    elif trade_cost > 62.9:
        trade_cost=62.9
    
    return trade_cost