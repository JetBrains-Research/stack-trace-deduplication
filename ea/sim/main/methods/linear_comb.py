import numpy as np
from sklearn.linear_model import LinearRegression

from ea.sim.main.methods.base import SimStackModel


class LinearSimStackModel(SimStackModel):
    def __init__(self, models: list[SimStackModel]):
        self.models = models
        self.lin_reg = LinearRegression()

    def partial_fit(
            self,
            sim_train_data: list[tuple[int, int, int]] | None = None,
            unsup_data: list[int] | None = None
    ) -> 'LinearSimStackModel':
        fs = []
        y = np.array([x[2] for x in sim_train_data])

        for model in self.models:
            model.partial_fit(sim_train_data, unsup_data)
            fs.append(model.predict_pairs(sim_train_data))

        fs = np.array(fs).T
        # with open("train_lin_reg.csv", 'w') as f:
        #     f.write(",".join([model.name() for model in self.models]) + ",target\n")
        #     for x, t in zip(fs, y):
        #         f.write(",".join(map(str, x)) + "," + str(t) + "\n")
        self.lin_reg.fit(fs, y)
        print("Lin model coefs", self.lin_reg.coef_)

        return self

    def predict(self, anchor_id: int, stack_ids: list[int]) -> list[float]:
        fs = []
        for model in self.models:
            fs.append(model.predict(anchor_id, stack_ids))
        fs = np.array(fs).T
        return self.lin_reg.predict(fs)

    def name(self) -> str:
        return "linreg_on__" + ",".join([model.name() for model in self.models])

    @property
    def min_score(self) -> float:
        return float("-inf")
