from ea.sim.main.data.objects.stack import Stack, Frame
from ea.sim.main.data.parsers.base import StackParser, MethodNameUnifier

"""
JSON format example:
    {
      "id": 15064545,
      "timestamp": "1686164868429",
      "errors": [
        "java.lang.Throwable"
      ],
      "keywords": [
    
      ],
      "messages": [
        "Slow operations are prohibited on EDT. See SlowOperations.assertSlowOperationsAreAllowed javadoc."
      ],
      "elements": [
        {
          "name": "....diagnostic.Logger.error",
          "file_name": "Logger.java",
          "line_number": 367,
          "commit_hash": null,
          "subsystem": "com....openapi.diagnostic"
        },
        {
          "name": "....SlowOperations.assertSlowOperationsAreAllowed",
          "file_name": "SlowOperations.java",
          "line_number": 129,
          "commit_hash": null,
          "subsystem": "com.....util"
        }
      ]
    }
"""


class StackParserV1(StackParser):
    @staticmethod
    def parse_frames(frames: list[dict]) -> list[Frame]:
        # 'None' to reduce the size of stack, add in the future if needed.
        return [
            Frame(
              name=MethodNameUnifier.unify(frame["name"]),
              file_name=frame["file_name"],
              line_number=frame["line_number"],
            )
            for frame in frames
        ]

    @classmethod
    def from_dict(cls, x: dict) -> Stack:
        return Stack(
            id=x["id"],
            timestamp=int(x["timestamp"]),
            errors=x.get("errors", None),
            frames=cls.parse_frames(x["elements"]),
            messages=x.get("message", None),
            comment=x.get("comment", None),
        )
