from abc import ABC, abstractmethod
from typing import Any, Type

from ea.sim.main.data.objects.stack import Stack
from ea.sim.main.preprocess.token import PreTokItem


class Entry2Seq(ABC):
    registry: dict[str, Type["Entry2Seq"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Entry2Seq.registry[cls.__name__] = cls

    @abstractmethod
    def __call__(self, stack: Stack) -> list[PreTokItem]:
        raise NotImplementedError

    @abstractmethod
    def state(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "state": self.state()}

    @staticmethod
    def from_dict(state: dict[str, Any]) -> "Entry2Seq":
        if state["type"] in Entry2Seq.registry:
            entry2seq_cl = Entry2Seq.registry[state["type"]]
            return entry2seq_cl.from_dict(state["state"])
        else:
            raise ValueError(f"Could not find Entry2Seq type: {state['type']}")

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class Entry2SeqHelper:
    def __init__(self, cased: bool = True, trim_len: int = 0, sep: str = '.'):
        self.sep = sep
        self.cased = cased
        self.trim_len = trim_len
        self._name = ("" if cased else "un") + "cs" + (f"_tr{trim_len}" if trim_len > 0 else "")

    def __call__(self, seq: list[PreTokItem]) -> list[PreTokItem]:
        # inplace operations
        if self.trim_len > 0:
            for s in seq:
                s.value = self.sep.join(s.value.split(self.sep)[:-self.trim_len])

        if not self.cased:
            for s in seq:
                s.value = s.value.lower()

        return seq

    def name(self) -> str:
        return self._name


class Stack2Seq(Entry2Seq):
    def __init__(self, cased: bool = True, trim_len: int = 0, sep: str = '.'):
        self.helper = Entry2SeqHelper(cased, trim_len, sep)
        self._args = {"cased": cased, "trim_lem": trim_len, "sep": sep}

    def __call__(self, stack: Stack) -> list[PreTokItem]:
        return self.helper([
            PreTokItem(frame.name, {
                "has_marker": frame.has_marker,
                "timestamp": stack.timestamp - frame.timestamp if frame.timestamp is not None else None,
            })
            for frame in stack.frames[::-1]
        ])

    def state(self) -> dict[str, Any]:
        return self._args

    @staticmethod
    def from_dict(state: dict[str, Any]) -> "Entry2Seq":
        return Stack2Seq(**state)

    def name(self) -> str:
        return "st_" + self.helper.name()


class Exception2Seq(Entry2Seq):
    def __init__(self, cased: bool = True, trim_len: int = 0, throw: bool = False, to_set: bool = True):
        self.helper = Entry2SeqHelper(cased, trim_len)
        self.throw = throw
        self.ex_transform = lambda x: (sorted(list(set(x)), reverse=True) if to_set else x)
        self._name = "ex_" + self.helper.name() + ("_" if throw else "_un") + "thr_" + (
            "st" if to_set else "lst")
        self._args = {"cased": cased, "trim_len": trim_len, "throw": throw, "to_set": to_set}

    def __call__(self, stack: Stack) -> list[PreTokItem]:
        exceptions = list(map(lambda x: x + (".throw" if self.throw else ""), self.ex_transform(stack.errors)))
        return self.helper([PreTokItem(x) for x in exceptions])

    def state(self) -> dict[str, Any]:
        return self._args

    @staticmethod
    def from_dict(state: dict[str, Any]) -> "Entry2Seq":
        return Exception2Seq(**state)

    def name(self) -> str:
        return self._name


class Message2Seq(Entry2Seq):
    def __init__(self, cased: bool = True):
        self.helper = Entry2SeqHelper(cased)
        self._args = {"cased": cased}

    def __call__(self, stack: Stack) -> list[PreTokItem]:
        msgs = list(filter(lambda x: x.strip() != '', stack.messages))
        return self.helper([PreTokItem(x) for x in msgs])

    def state(self) -> dict[str, Any]:
        return self._args

    @staticmethod
    def from_dict(state: dict[str, Any]) -> "Entry2Seq":
        return Message2Seq(**state)

    def name(self) -> str:
        return "msg_" + self.helper.name()


class MultiEntry2Seq(Entry2Seq):
    def __init__(self, e2ss: list[Entry2Seq]):
        self.e2ss = e2ss

    def __call__(self, stack: Stack) -> list[PreTokItem]:
        return [item for e2s in self.e2ss for item in e2s(stack)]

    def state(self) -> dict[str, Any]:
        return {"e2ss": [e2s.to_dict() for e2s in self.e2ss]}

    @staticmethod
    def from_dict(state: dict[str, Any]) -> "Entry2Seq":
        dicts = [st for st in state["e2ss"]]
        e2ss = [Entry2Seq.from_dict(d) for d in dicts]
        return MultiEntry2Seq(e2ss)

    def name(self) -> str:
        return "multe2s_" + "_".join(e2s.name() for e2s in self.e2ss)
