import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from decimal import Decimal
from nautilus_trader.common.component LiveClock
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.backtest.config import BacktestRunConfig

class RayBacktestNode(HyperoptBacktestNode):
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
        max_evals: int = 50,
        cpu_nums: int = 4,
        gpu_nums: int = 0
    ):
        def objective(config):
            try:
                strategies = [
                    ImportableStrategyConfig(
                        strategy_path=self.strategy_path,
                        config_path=self.config_path,
                        config=self.strategy_config(
                            **config,
                        ),
                    ),
                ]

                local_config = self.config
                local_config = local_config.replace(strategies=strategies)
                local_config.check()

                result = self._run(
                    run_config_id=local_config.id,
                    engine_config=local_config.engine,
                    venue_configs=local_config.venues,
                    data_configs=local_config.data,
                    batch_size_bytes=local_config.batch_size_bytes,
                )

                base_currency = self.config.venues[0].base_currency
                profit_factor = result.stats_returns["Profit Factor"]

                if (
                    profit_factor <= 0
                    or result.total_positions < minimum_positions
                ):
                    tune.report(loss=float("inf"))
                else:
                    tune.report(loss=(1 / profit_factor))

            except Exception as e:
                tune.report(loss=float("inf"))

        search_space = params

        # Define the scheduler for early stopping
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=100,
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
            num_samples=max_evals,
            scheduler=scheduler,
            progress_reporter=reporter,
            resources_per_trial={"cpu": cpu_nums, "gpu": gpu_nums}
        )

        return analysis.best_config

# Example usage:
base_config = BacktestRunConfig(...)  # Initialize with appropriate values
node = RayBacktestNode(base_config)

params = {
    'param1': tune.uniform(0, 1),
    'param2': tune.uniform(0, 10),
    'param3': tune.choice([0, 1, 2, 3, 4])
}

best_params = node.ray_search(
    params=params,
    minimum_positions=50,
    max_evals=100
)

print("Best parameters found:", best_params)



