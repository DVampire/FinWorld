import numpy as np
import os

from finworld.registry import EVALUATOR
from finworld.metric import ARR, SR, MDD, CR, SOR, VOL
from finworld.utils import TradingRecords




@EVALUATOR.register_module(force=True)
class TradingEvaluator():

    def __init__(self,
                 *args,
                 config=None,
                 environment=None,
                 agent=None,
                 logger=None,
                 wandb=None,
                 tensorboard=None,
                 **kwargs):
        self.config = config
        self.environment = environment
        self.agent = agent
        self.logger = logger
        self.wandb = wandb
        self.tensorboard = tensorboard

    async def __call__(self):

        prefix = "evaluator"

        trading_records = TradingRecords()

        # TRY NOT TO MODIFY: start the game
        state, info = self.environment.reset()
        # Update the trading records with initial state information
        trading_records.add(
            dict(
                timestamp=info["timestamp"],
                price=info["price"],
                cash=info["cash"],
                position=info["position"],
                value=info["value"],
            ),
        )

        while True:

            res = await self.agent.run(state = state, info = info, reset=False)
            action = res.output
            state, reward, done, truncted, info = self.environment.step(action)

            trading_records.add(
                dict(
                    action=info["action"],
                    action_label=info["action_label"],
                    ret=info["ret"],
                    total_profit=info["total_profit"],
                    timestamp=info["timestamp"],  # next timestamp
                    price=info["price"],  # next price
                    cash=info["cash"],  # next cash
                    position=info["position"],  # next position
                    value=info["value"],  # next value
                ),
            )

            if "final_info" in info:
                break

        # End of the environment, add the final record
        trading_records.add(
            dict(
                action=info["action"],
                action_label=info["action_label"],
                ret=info["ret"],
                total_profit=info["total_profit"],
            )
        )

        rets = trading_records.data["ret"]
        positions = trading_records.data["position"]
        actions = trading_records.data["action"]

        rets = np.array(rets)
        arr = ARR(rets)  # take as reward
        sr = SR(rets)
        dd = MDD(rets)
        mdd = MDD(rets)
        cr = CR(rets, mdd=mdd)
        sor = SOR(rets, dd=dd)
        vol = VOL(rets)

        positions = np.array(positions).flatten()
        turnover_rate = TurnoverRate(positions)

        actions = np.array(actions).flatten()
        num_trades = NumTrades(actions)
        num_buys = NumBuys(actions)
        num_sells = NumSells(actions)
        avg_hold_period = AvgHoldPeriod(actions)
        activity_rate = ActivityRate(actions)
        avg_trade_interval = AvgTradeInterval(actions)
        buy_to_sell_ratio = BuyToSellRatio(actions)

        self.logger.info(
            f"ARR%={arr * 100}, "
            f"SR={sr}, "
            f"CR={cr}, "
            f"SOR={sor}, "
            f"DD={dd}, "
            f"MDD%={mdd * 100}, "
            f"VOL={vol}, "
            f"TurnoverRate={turnover_rate}, "
            f"NumTrades={num_trades}, "
            f"NumBuys={num_buys}, "
            f"NumSells={num_sells}, "
            f"AvgHoldPeriod={avg_hold_period}, "
            f"ActivityRate%={activity_rate * 100}, "
            f"AvgTradeInterval={avg_trade_interval}, "
            f"BuyToSellRatio={buy_to_sell_ratio}"
        )

        # Save the trading records
        trading_records_df = trading_records.to_dataframe()
        trading_records_df.to_csv(os.path.join(self.config.exp_path, f"{prefix}_records.csv"), index=True)