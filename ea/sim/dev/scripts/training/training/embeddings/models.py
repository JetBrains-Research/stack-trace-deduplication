from dataclasses import dataclass
from enum import Enum


@dataclass
class OpenAIModelInfo:
    name: str
    price_per_million: float
    max_input: int

    def price(self, tokens: int) -> float:
        return (tokens / 1_000_000) * self.price_per_million


class OpenAIModel(Enum):
    V3_SMALL = OpenAIModelInfo("text-embedding-3-small", 0.02, 8191)
    V3_LARGE = OpenAIModelInfo("text-embedding-3-large", 0.13, 8191)
    ADA_V2 = OpenAIModelInfo("text-embedding-ada-002", 0.10, 8191)

    @staticmethod
    def by_name(name: str) -> "OpenAIModel":
        match name:
            case "v3-small":
                return OpenAIModel.V3_SMALL
            case "v3-large":
                return OpenAIModel.V3_LARGE
            case "ada-v2":
                return OpenAIModel.ADA_V2
            case _:
                raise ValueError("Unknown model name")
