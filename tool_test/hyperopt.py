import sys 
import ray
from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from decimal import Decimal
from typing import Dict,Optional,Any 
from nautilus_trader.common.enums import LogLevel
from nautilus_trader.common.component import Logger
from nautilus_trader.common.component import LiveClock
from nautilus_trader.config import StrategyConfig
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.backtest.config import BacktestRunConfig
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig

class RayBacktestNode(BacktestNode):
    def __init__(self, base_config: BacktestRunConfig):
        super().__init__(configs=[base_config])

        self.config: BacktestRunConfig = base_config
        self.strategy_path: Optional[str] = None
        self.config_path: Optional[str] = None
        self.strategy_config: Optional[StrategyConfig] = None

    def set_strategy_config(
        self,
        strategy_path: str,
        config_path: str,
    ) -> None:
        """
        Set strategy parameters which can be passed to the hyperopt objective.

        Parameters
        ----------
        strategy_path : str
            The path to the strategy.
        config_path : str
            The path to the strategy config.

        """
        self.strategy_path = strategy_path
        self.config_path = config_path

    def ray_search(
        self,
        params: Dict[str, Any],
        minimum_positions: int = 50,
        num_samples: int = 50,
        cpu_nums: int = 4,
        gpu_nums: int = 0
    ):
        def objective(config):
                strategies = [
                    ImportableStrategyConfig(
                        strategy_path=self.strategy_path,
                        config_path=self.config_path,
                        config=config,
                    ),
                ]
                local_config = BacktestRunConfig(
                    engine=BacktestEngineConfig(
                        strategies=strategies,
                        logging=LoggingConfig(log_level="OFF"),
                    ),
                    venues=self.config.venues,
                    data=self.config.data 
                )
                print(self.config.data)
                result = self._run(
                    run_config_id=local_config.id,
                    engine_config=local_config.engine,
                    venue_configs=local_config.venues,
                    data_configs=local_config.data,
                    batch_size_bytes=local_config.batch_size_bytes,
                )

                sharp_ratio = result.stats_returns['Sharpe Ratio (252 days)']

                # if (
                #     sharp_ratio <= 0
                #     or result.total_positions < minimum_positions
                # ):
                #     train.report({"loss":float("inf")})
                # else:
                train.report({"loss": -sharp_ratio})

        search_space = params

        # Define the scheduler for early stopping
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            grace_period=1,
            reduction_factor=2
        )

        reporter = CLIReporter(
            parameter_columns=list(params.keys()),
            metric_columns=["loss", "training_iteration"]
        )

        analysis = tune.run(
            objective,
            config=search_space,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            resources_per_trial={"cpu": cpu_nums, "gpu": gpu_nums}
        )

        return analysis
