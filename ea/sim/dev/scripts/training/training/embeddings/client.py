from dataclasses import dataclass
import numpy as np
import tiktoken
from openai import OpenAI

from ea.sim.dev.scripts.training.training.embeddings.models import OpenAIModelInfo, OpenAIModel


@dataclass
class Request:
    text: str


@dataclass
class Response:
    embedding: np.ndarray
    consumed_tokens: int


class PriceTracker:
    def __init__(self, model: OpenAIModel, alert_every_spent_dollar: int | None = None):
        self._model = model
        self._tokens = 0
        self._alert_every_spent_dollar = alert_every_spent_dollar
        self._next_alert = alert_every_spent_dollar

    def print_price(self) -> None:
        print(f"Consumed tokens: {self._tokens} | Total price: {self.price:.2f}$")

    def update(self, tokens: int) -> None:
        self._tokens += tokens
        if self._alert_every_spent_dollar is not None:
            if self.price >= self._next_alert:
                self.print_price()
                self._next_alert += self._alert_every_spent_dollar

    @property
    def price(self) -> float:
        return self._model.value.price(self._tokens)


class OpenAIClient:
    def __init__(self, model: OpenAIModelInfo, token: str):
        self._client = OpenAI(api_key=token)
        self._model = model

    def encode(self, request: Request) -> Response:
        encoding = tiktoken.encoding_for_model(self._model.name)
        tokens = encoding.encode(request.text)[-self._model.max_input:]
        # start_time = perf_counter()
        response = self._client.embeddings.create(input=tokens, model=self._model.name)
        # print()
        # print(f"{perf_counter() - start_time:.3f}")
        # print()
        embedding = np.array(response.data[0].embedding)
        response = Response(
            embedding,
            # np.array([0]),
            len(tokens)
        )
        return response
