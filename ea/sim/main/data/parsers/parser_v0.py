from ea.sim.main.data.objects.stack import Stack, Frame
from ea.sim.main.data.parsers.base import StackParser, MethodNameUnifier

"""
JSON format example:
    {
      "id": 7492740,
      "timestamp": "1625306295072",
      "class": [
        "java.lang.Throwable"
      ],
      "message": [
        "Slow operations are prohibited on EDT. See SlowOperations.assertSlowOperationsAreAllowed javadoc."
      ],
      "frames": [
        "....openapi.diagnostic.Logger.error",
        "....util.SlowOperations.assertSlowOperationsAreAllowed",
        "....util.indexing.FileBasedIndexImpl.ensureUpToDate"
      ]
    }
"""


class StackParserV0(StackParser):
    @staticmethod
    def parse_frames(frames: list[str]) -> list[Frame]:
        return [Frame(name=MethodNameUnifier.unify(frame)) for frame in frames]

    @classmethod
    def from_dict(cls, x: dict) -> Stack:
        return Stack(
            id=x["id"],
            timestamp=int(x["timestamp"]),
            errors=x.get("errors", None),
            frames=cls.parse_frames(x["frames"]),
            messages=x.get("message", None),
            comment=x.get("comment", None),
        )
