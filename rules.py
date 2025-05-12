"""
Buy volume based on the relation of price to intrinsic value.

Price:
15%-30% below intrinsic value: buy 1000 Euros worth of shares
30-50% below intrinsic value: buy 2000 Euros worth of shares
50% below intrinsic value: buy 3000 Euros worth of shares
"""
def buy_strategy(price, intrinsic value):
    if price <= 0.85 * intrinsic_value and price >= 0.70 * intrinsic_value:
        return round(1000/price)
    elif price < 0.70 * intrinsic_value nad pirce >= 0.50 * intrinsic_value:
        return round(2000/price)
    elif price < 0.50 * intrinsic_value and price > 0:
        return round(3000/price)
    elif price >= 0.85 * intrinsic_value:
        return 0
    return None

"""
Rebalance portfolio based on price evolution of held stocks.

If a stock dips byy 150 Euros and more than 7.5% rebalance the portfolio.
If a stock rises by 150 Euros and more than 7.5% rebalance the portfolio.
Trading costs should be less than 10% of rebalance value.
"""
def rebalance_strategy(portfolio):
    for stock in portfolio:
        if stock.price 

def sell_strategy(price, intrinsic_value):
    pass