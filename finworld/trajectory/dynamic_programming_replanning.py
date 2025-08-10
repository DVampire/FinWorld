from finworld.trajectory.operation import buy, sell, hold, value
import numpy as np
import pandas as pd

def max_profit_with_actions(initial_cash, initial_position, prices, fee_ratio):
    """
    Computes the maximum value achievable starting from a given cash and position,
    and returns the final value and sequence of actions.

    Returns:
        tuple: (maximum value, list of actions ['BUY', 'SELL', 'HOLD'])
    """
    n = len(prices)
    if n == 0:
        return 0, []

    # DP arrays: dp[i][s] is max value at day i in state s (0: not holding, 1: holding)
    dp = [[0 for _ in range(2)] for _ in range(n)]
    actions = [['HOLD' for _ in range(2)] for _ in range(n)]
    cashs = [[0 for _ in range(2)] for _ in range(n)]
    positions = [[0 for _ in range(2)] for _ in range(n)]

    # Initialize day 0 based on initial position
    if initial_position == 0:
        # State 0: Not holding - HOLD
        cashs[0][0] = initial_cash
        positions[0][0] = 0
        dp[0][0] = value(cashs[0][0], positions[0][0], prices[0])
        actions[0][0] = 'HOLD'

        # State 1: Holding - BUY if possible
        cash_buy, position_buy = buy(initial_cash, 0, prices[0], fee_ratio)
        if position_buy > 0:  # Can buy at least one stock
            cashs[0][1] = cash_buy
            positions[0][1] = position_buy
            dp[0][1] = value(cashs[0][1], positions[0][1], prices[0])
            actions[0][1] = 'BUY'
        else:
            dp[0][1] = -np.inf  # Cannot buy due to insufficient cash
    else:
        # State 1: Holding - HOLD initial position
        cashs[0][1] = initial_cash
        positions[0][1] = initial_position
        dp[0][1] = value(cashs[0][1], positions[0][1], prices[0])
        actions[0][1] = 'HOLD'

        # State 0: Not holding - SELL all
        cash_sell, position_sell = sell(initial_cash, initial_position, prices[0], fee_ratio)
        cashs[0][0] = cash_sell
        positions[0][0] = position_sell
        dp[0][0] = value(cashs[0][0], positions[0][0], prices[0])
        actions[0][0] = 'SELL'

    # Fill DP table for subsequent days
    for i in range(1, n):
        # State 0: Not holding
        # Option 1: HOLD from previous not holding
        cash_noop, position_noop = hold(cashs[i-1][0], positions[i-1][0], prices[i], fee_ratio)
        value_noop = value(cash_noop, position_noop, prices[i])

        # Option 2: SELL from previous holding (if holding anything)
        if positions[i-1][1] > 0:
            cash_sell, position_sell = sell(cashs[i-1][1], positions[i-1][1], prices[i], fee_ratio)
            value_sell = value(cash_sell, position_sell, prices[i])
        else:
            value_sell = -np.inf  # Cannot sell if no position

        if value_noop > value_sell:
            dp[i][0] = value_noop
            cashs[i][0], positions[i][0] = cash_noop, position_noop
            actions[i][0] = 'HOLD'
        else:
            dp[i][0] = value_sell
            cashs[i][0], positions[i][0] = cash_sell, position_sell
            actions[i][0] = 'SELL'

        # State 1: Holding
        # Option 1: HOLD from previous holding
        cash_noop_holding, position_noop_holding = hold(cashs[i-1][1], positions[i-1][1], prices[i], fee_ratio)
        value_noop_holding = value(cash_noop_holding, position_noop_holding, prices[i])

        # Option 2: BUY from previous not holding (if not holding)
        if positions[i-1][0] == 0:
            cash_buy, position_buy = buy(cashs[i-1][0], positions[i-1][0], prices[i], fee_ratio)
            value_buy = value(cash_buy, position_buy, prices[i]) if position_buy > 0 else -np.inf
        else:
            value_buy = -np.inf  # Cannot buy if already holding

        if value_noop_holding > value_buy:
            dp[i][1] = value_noop_holding
            cashs[i][1], positions[i][1] = cash_noop_holding, position_noop_holding
            actions[i][1] = 'HOLD'
        else:
            dp[i][1] = value_buy
            cashs[i][1], positions[i][1] = cash_buy, position_buy
            actions[i][1] = 'BUY'

    # Backtrack to get the action sequence
    final_value = max(dp[n-1])
    is_holding = dp[n-1].index(final_value)
    final_actions = []
    for i in range(n-1, -1, -1):
        final_actions.append(actions[i][is_holding])
        is_holding = 1 - is_holding if actions[i][is_holding] in ['BUY', 'SELL'] else is_holding
    return final_value, list(reversed(final_actions))

def get_first_action(initial_cash, initial_position, prices, fee_ratio):
    """
    Computes the DP for the given prices and returns the optimal action for the first day.
    
    Returns:
        str: Action for the first day ('BUY', 'SELL', or 'HOLD').
    """
    if prices.size == 0:
        return 'HOLD'
    _, actions = max_profit_with_actions(initial_cash, initial_position, prices, fee_ratio)
    return actions[0] if actions else 'HOLD'

def max_profit_with_replanning(prices, initial_cash, fee_ratio):
    """
    Computes the maximum value by replanning DP after each operation, maintaining current position
    and operation history, and returns the final value and action sequence.

    Returns:
        tuple: (final value, list of actions ['BUY', 'SELL', 'HOLD'])
    """
    current_cash = initial_cash
    current_position = 0
    actions = []

    # Process each day
    for i in range(len(prices)):
        remaining_prices = prices[i:]
        # Recompute DP for remaining prices and get the action for the current day
        action = get_first_action(current_cash, current_position, remaining_prices, fee_ratio)
        actions.append(action)

        # Update state based on the action
        if action == 'BUY':
            current_cash, current_position = buy(current_cash, current_position, remaining_prices[0], fee_ratio)
        elif action == 'SELL':
            current_cash, current_position = sell(current_cash, current_position, remaining_prices[0], fee_ratio)
        else:
            current_cash, current_position = hold(current_cash, current_position, remaining_prices[0], fee_ratio)

    # Compute final value
    final_value = value(current_cash, current_position, prices[-1])
    return final_value, actions

if __name__ == '__main__':
    examples = [
        [1, 2, 3, 4, 5, 6],  # Increasing prices
        [6, 5, 4, 3, 2, 1],  # Decreasing prices
        [1, 3, 6, 3, 5, 1],  # Fluctuating prices
    ]

    for prices in examples:
        cash = 1000
        fee_ratio = 0.001
        max_profit, actions = max_profit_with_replanning(prices, cash, fee_ratio)
        print(f"Prices: {prices}")
        print(f"Max profit: {max_profit}, Actions: {actions}")

        # Simulate and print state after each action for verification
        current_cash = cash
        current_position = 0
        for action, price in zip(actions, prices):
            if action == 'BUY':
                current_cash, current_position = buy(current_cash, current_position, price, fee_ratio)
            elif action == 'SELL':
                current_cash, current_position = sell(current_cash, current_position, price, fee_ratio)
            else:
                current_cash, current_position = hold(current_cash, current_position, price, fee_ratio)
            value_ = value(current_cash, current_position, price)
            print(f"Action: {action}, Price: {price}, Cash: {current_cash:.2f}, Position: {current_position}, Value: {value_:.2f}")
        print()
