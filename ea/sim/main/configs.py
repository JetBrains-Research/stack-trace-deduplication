from dataclasses import dataclass

from ea.sim.main.utils import Scope


@dataclass
class SeqCoderConfig:
    scope_id: int
    cased: bool
    bpe_cased: bool
    sep: str
    max_len: int | None

    @staticmethod
    def from_dict(config: dict) -> "SeqCoderConfig":
        return SeqCoderConfig(**config)

    @staticmethod
    def by_scope(scope: Scope) -> "SeqCoderConfig":
        return SeqCoderConfig(scope_id=scope.value, cased=True, bpe_cased=False, sep=".", max_len=None)


@dataclass
class NeuralModelConfig:
    @dataclass
    class Encoder:
        model_type: str
        dim: int
        hid_dim: int

    @dataclass
    class Classifier:
        model_type: str

    encoder: Encoder
    classifier: Classifier

    @staticmethod
    def from_dict(config: dict) -> "NeuralModelConfig":
        return NeuralModelConfig(
            encoder=NeuralModelConfig.Encoder(**config["encoder"]),
            classifier=NeuralModelConfig.Classifier(**config["classifier"])
        )

    @staticmethod
    def by_scope(scope: Scope) -> "NeuralModelConfig":
        if scope == Scope.MainProject:
            encoder = NeuralModelConfig.Encoder(model_type="LSTMEncoder", dim=50, hid_dim=100)
            classifier = NeuralModelConfig.Classifier(model_type="CosClassifier")
            return NeuralModelConfig(encoder, classifier)
        elif scope == Scope.SlowOps:
            encoder = NeuralModelConfig.Encoder(model_type="LSTMEncoder", dim=50, hid_dim=100)
            classifier = NeuralModelConfig.Classifier(model_type="CosClassifier")
            return NeuralModelConfig(encoder, classifier)
        elif scope == Scope.SideProject:
            encoder = NeuralModelConfig.Encoder(model_type="LSTMEncoder", dim=50, hid_dim=100)
            classifier = NeuralModelConfig.Classifier(model_type="CosClassifier")
            return NeuralModelConfig(encoder, classifier)

