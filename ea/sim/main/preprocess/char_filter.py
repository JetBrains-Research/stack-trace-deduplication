from ea.sim.main.preprocess.token import PreTokItem


class CharFilter:
    def __init__(self):
        self.ok_symbols = set([chr(i) for i in range(ord('a'), ord('z') + 1)] + ['.', ',', '_'])  # $

    def filter_word(self, word: str) -> str:
        return "".join(filter(lambda x: x.lower() in self.ok_symbols, word))

    def __call__(self, seq: list[PreTokItem]) -> list[PreTokItem]:
        for word in seq:
            word.value = self.filter_word(word.value)
        return seq
