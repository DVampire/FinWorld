import numpy as np
import pandas as pd

from finworld.trajectory.operation import buy, sell, hold, value

def max_profit_with_actions(prices, cash, fee_ratio):
    """
    Computes the maximum profit from buying and selling stocks with a fee, without any limit on SELL operations.
    Returns the maximum profit and the sequence of actions (BUY, SELL, HOLD).

    Returns:
        tuple: (maximum profit, list of actions)
    """
    n = len(prices)
    if n == 0:
        return 0, []

    # Initialize DP arrays
    dp = [[0 for _ in range(2)] for _ in range(n)]  # 0: Not holding, 1: Holding
    actions = [['HOLD' for _ in range(2)] for _ in range(n)]
    cashs = [[cash for _ in range(2)] for _ in range(n)]
    positions = [[0 for _ in range(2)] for _ in range(n)]

    # Initialize for the first day
    cashs[0][0] = cash
    positions[0][0] = 0
    dp[0][0] = value(cash, 0, prices[0])
    actions[0][0] = 'HOLD'
    
    cash_buy, position_buy = buy(cash, 0, prices[0], fee_ratio)
    if position_buy > 0:
        cashs[0][1], positions[0][1] = cash_buy, position_buy
        dp[0][1] = value(cash_buy, position_buy, prices[0])
        actions[0][1] = 'BUY'
    else:
        dp[0][1] = -np.inf

    # Fill DP table
    for i in range(1, n):
        # Not holding
        # Option 1: HOLD from not holding
        cash_noop, position_noop = hold(cashs[i-1][0], positions[i-1][0], prices[i], fee_ratio)
        value_noop = value(cash_noop, position_noop, prices[i])
        # Option 2: SELL from holding
        cash_sell, position_sell = sell(cashs[i-1][1], positions[i-1][1], prices[i], fee_ratio)
        value_sell = value(cash_sell, position_sell, prices[i])

        if value_noop > value_sell:
            dp[i][0] = value_noop
            cashs[i][0], positions[i][0] = cash_noop, position_noop
            actions[i][0] = 'HOLD'
        else:
            dp[i][0] = value_sell
            cashs[i][0], positions[i][0] = cash_sell, position_sell
            actions[i][0] = 'SELL'

        # Holding
        # Option 1: HOLD from holding
        cash_noop_holding, position_noop_holding = hold(cashs[i-1][1], positions[i-1][1], prices[i], fee_ratio)
        value_noop_holding = value(cash_noop_holding, position_noop_holding, prices[i])
        # Option 2: BUY from not holding
        cash_buy, position_buy = buy(cashs[i-1][0], positions[i-1][0], prices[i], fee_ratio)
        value_buy = value(cash_buy, position_buy, prices[i]) if position_buy > positions[i-1][0] else -np.inf

        if value_noop_holding > value_buy:
            dp[i][1] = value_noop_holding
            cashs[i][1], positions[i][1] = cash_noop_holding, position_noop_holding
            actions[i][1] = 'HOLD'
        else:
            dp[i][1] = value_buy
            cashs[i][1], positions[i][1] = cash_buy, position_buy
            actions[i][1] = 'BUY'

    # Backtrack to find actions
    final_value = max(dp[n-1])
    is_holding = dp[n-1].index(final_value)
    final_actions = []

    for i in range(n-1, -1, -1):
        final_actions.append(actions[i][is_holding])
        is_holding = 1 - is_holding if actions[i][is_holding] in ['BUY', 'SELL'] else is_holding

    return final_value, list(reversed(final_actions))

def max_profit_with_actions_threshold(initial_cash, initial_position, prices, fee_ratio, max_count_sell):
    """
    Computes the maximum profit from buying and selling stocks with a fee, limited by max_count_sell SELL operations.
    Handles non-zero initial positions and returns the maximum profit and sequence of actions.
    
    Returns:
        tuple: (maximum profit, list of actions)
    """
    n = len(prices)
    if n == 0:
        return 0, []

    # Initialize DP arrays
    dp = [[[0 for _ in range(2)] for _ in range(max_count_sell + 1)] for _ in range(n)]
    actions = [[['HOLD' for _ in range(2)] for _ in range(max_count_sell + 1)] for _ in range(n)]
    cashs = [[[0 for _ in range(2)] for _ in range(max_count_sell + 1)] for _ in range(n)]
    positions = [[[0 for _ in range(2)] for _ in range(max_count_sell + 1)] for _ in range(n)]

    # Initialize for day 0
    for j in range(max_count_sell + 1):
        if initial_position == 0:
            # State 0: Not holding
            cashs[0][j][0] = initial_cash
            positions[0][j][0] = 0
            dp[0][j][0] = value(initial_cash, 0, prices[0])
            actions[0][j][0] = 'HOLD'
            # State 1: Holding
            cash_buy, position_buy = buy(initial_cash, 0, prices[0], fee_ratio)
            if position_buy > 0:
                cashs[0][j][1] = cash_buy
                positions[0][j][1] = position_buy
                dp[0][j][1] = value(cash_buy, position_buy, prices[0])
                actions[0][j][1] = 'BUY'
            else:
                dp[0][j][1] = -np.inf
        else:
            # State 1: Holding
            cashs[0][j][1] = initial_cash
            positions[0][j][1] = initial_position
            dp[0][j][1] = value(initial_cash, initial_position, prices[0])
            actions[0][j][1] = 'HOLD'
            # State 0: Not holding
            if j >= 1:
                cash_sell, position_sell = sell(initial_cash, initial_position, prices[0], fee_ratio)
                cashs[0][j][0] = cash_sell
                positions[0][j][0] = position_sell
                dp[0][j][0] = value(cash_sell, position_sell, prices[0])
                actions[0][j][0] = 'SELL'
            else:
                dp[0][j][0] = -np.inf

    # Fill DP table
    for i in range(1, n):
        for j in range(max_count_sell + 1):
            # Not holding
            # Option 1: HOLD from not holding
            cash_noop, position_noop = hold(cashs[i-1][j][0], positions[i-1][j][0], prices[i], fee_ratio)
            value_noop = value(cash_noop, position_noop, prices[i])
            # Option 2: SELL from holding, if j > 0
            if j > 0:
                cash_sell, position_sell = sell(cashs[i-1][j-1][1], positions[i-1][j-1][1], prices[i], fee_ratio)
                value_sell = value(cash_sell, position_sell, prices[i])
            else:
                value_sell = -np.inf

            if value_noop > value_sell:
                dp[i][j][0] = value_noop
                cashs[i][j][0], positions[i][j][0] = cash_noop, position_noop
                actions[i][j][0] = 'HOLD'
            else:
                dp[i][j][0] = value_sell
                cashs[i][j][0], positions[i][j][0] = cash_sell, position_sell
                actions[i][j][0] = 'SELL'

            # Holding
            # Option 1: HOLD from holding
            cash_noop_holding, position_noop_holding = hold(cashs[i-1][j][1], positions[i-1][j][1], prices[i], fee_ratio)
            value_noop_holding = value(cash_noop_holding, position_noop_holding, prices[i])
            # Option 2: BUY from not holding
            cash_buy, position_buy = buy(cashs[i-1][j][0], positions[i-1][j][0], prices[i], fee_ratio)
            value_buy = value(cash_buy, position_buy, prices[i]) if position_buy > positions[i-1][j][0] else -np.inf

            if value_noop_holding > value_buy:
                dp[i][j][1] = value_noop_holding
                cashs[i][j][1], positions[i][j][1] = cash_noop_holding, position_noop_holding
                actions[i][j][1] = 'HOLD'
            else:
                dp[i][j][1] = value_buy
                cashs[i][j][1], positions[i][j][1] = cash_buy, position_buy
                actions[i][j][1] = 'BUY'

    # Find the best ending state
    final_values = [max(dp[n-1][j]) for j in range(max_count_sell + 1)]
    best_j = np.argmax(final_values)
    final_value = final_values[best_j]
    is_holding = dp[n-1][best_j].index(final_value)
    final_actions = []
    count_sell = best_j

    # Backtrack actions
    for i in range(n-1, -1, -1):
        now_action = actions[i][count_sell][is_holding]
        final_actions.append(now_action)
        if now_action == 'SELL':
            count_sell -= 1
        is_holding = 1 - is_holding if now_action in ['BUY', 'SELL'] else is_holding

    return final_value, list(reversed(final_actions))

def get_first_action_threshold(initial_cash, initial_position, prices, fee_ratio, remaining_sell):
    """
    Computes the DP for the remaining days and returns the optimal action for the first day.
    
    Returns:
        str: Optimal action for the first day ('BUY', 'SELL', or 'HOLD').
    """
    if prices.size == 0:
        return 'HOLD'
    _, actions = max_profit_with_actions_threshold(initial_cash, initial_position, prices, fee_ratio, remaining_sell)
    return actions[0] if actions else 'HOLD'

def max_profit_with_replanning_threshold(prices, initial_cash, fee_ratio, max_count_sell):
    """
    Computes the maximum profit using a replanning strategy, limited by max_count_sell SELL operations.
    Recomputes the optimal action for each day based on the current state and remaining SELL operations.
    
    Returns:
        tuple: (maximum profit, list of actions)
    """
    current_cash = initial_cash
    current_position = 0
    used_sell = 0
    actions = []

    for i in range(len(prices)):
        remaining_sell = max_count_sell - used_sell
        if remaining_sell < 0:
            remaining_sell = 0  # Ensure non-negative
        action = get_first_action_threshold(current_cash, current_position, prices[i:], fee_ratio, remaining_sell)
        actions.append(action)
        if action == 'BUY':
            current_cash, current_position = buy(current_cash, current_position, prices[i], fee_ratio)
        elif action == 'SELL':
            current_cash, current_position = sell(current_cash, current_position, prices[i], fee_ratio)
            used_sell += 1
        else:
            current_cash, current_position = hold(current_cash, current_position, prices[i], fee_ratio)

    final_value = value(current_cash, current_position, prices[-1])
    return final_value, actions

if __name__ == '__main__':
    examples = [
        [1, 2, 3, 4, 5, 6],
        [6, 5, 4, 3, 2, 1],
        [1, 3, 6, 3, 5, 4],
        [1, 3, 6, 3, 5, 1]
    ]

    threshold = 0.5

    for prices in examples:
        cash = 1000
        fee_ratio = 0.001
        position = 0
        # Unconstrained case
        max_profit, actions = max_profit_with_actions(prices, cash, fee_ratio)
        print("max_profit:", max_profit, "actions:", actions)
        for action, price in zip(actions, prices):
            if action == 'BUY':
                cash, position = buy(cash, position, price, fee_ratio)
            elif action == 'SELL':
                cash, position = sell(cash, position, price, fee_ratio)
            else:
                cash, position = hold(cash, position, price, fee_ratio)
            value_ = value(cash, position, price)
            print("action:", action, "price:", price, "cash:", cash, "position:", position, "value:", value_)
        print()

        #========================================= Threshold with Replanning ===========================================#

        # Compute max_count_sell based on unconstrained SELL operations
        count_sell = sum(1 for action in actions if action == 'SELL')
        max_count_sell = int(count_sell * threshold)
        
        cash = 1000
        position = 0
        # Use replanning strategy
        max_profit, actions = max_profit_with_replanning_threshold(prices, cash, fee_ratio, max_count_sell)
        print(f"================== max_count_sell : {max_count_sell} ==================")
        print("max_profit:", max_profit, "actions:", actions)
        for action, price in zip(actions, prices):
            if action == 'BUY':
                cash, position = buy(cash, position, price, fee_ratio)
            elif action == 'SELL':
                cash, position = sell(cash, position, price, fee_ratio)
            else:
                cash, position = hold(cash, position, price, fee_ratio)
            value_ = value(cash, position, price)
            print("action:", action, "price:", price, "cash:", cash, "position:", position, "value:", value_)
        print("=========================================================")
        print()
