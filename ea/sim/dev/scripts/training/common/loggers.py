from abc import ABC, abstractmethod

from loguru import logger

class Logger(ABC):
    round_digits: int = 4

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        raise NotImplementedError

    @abstractmethod
    def log_interval_metrics(self, metrics: dict[str, tuple[float, float, float]], step: int | None = None):
        raise NotImplementedError


class ConsoleLogger(Logger):
    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        for metric_name, metric_value in metrics.items():
            logger.info(f"Step {step}, {metric_name}: {round(metric_value, self.round_digits)}")

    def log_interval_metrics(self, metrics: dict[str, tuple[float, float, float]], step: int | None = None):
        for metric_name, (value, left, right) in metrics.items():
            step_prefix = f"Step {step}, " if step is not None else ""
            logger.info(
                f"{step_prefix}{metric_name}: "
                f"{round(value, self.round_digits)} "
                f"({round(left, self.round_digits)}, {round(right, self.round_digits)})"
            )
