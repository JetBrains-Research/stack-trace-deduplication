from ea.sim.main.methods.base import SimStackModel
from ea.sim.main.preprocess.seq_coder import SeqCoder


class BaselineModel(SimStackModel):
    def __init__(self, coder: SeqCoder, top_k: int = 10):
        self.coder = coder
        self.top_k = top_k

    def partial_fit(
            self,
            sim_train_data: list[tuple[int, int, int]] | None = None,
            unsup_data: list[int] | None = None
    ) -> 'SimStackModel':
        return self

    def _predict(self, frames_1: list[int], frames_2: list[int]) -> float:
        # bottom first
        equal = 0
        min_len = min(len(frames_1), len(frames_2), self.top_k)
        for frame_1, frame_2 in zip(frames_1[-min_len:], frames_2[-min_len:]):
            equal += (frame_1 == frame_2)

        return equal / self.top_k

    def predict(self, anchor_id: int, stack_ids: list[int]) -> list[float]:
        def encode(stack_id: int) -> list[int]:
            stack = self.coder(stack_id)
            return [token.value for token in stack]

        anchor = encode(anchor_id)

        return [self._predict(anchor, encode(stack_id)) for stack_id in stack_ids]

    def name(self) -> str:
        return self.coder.name() + "_baseline"

    @property
    def min_score(self) -> float:
        return 0.0
