class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        words = set()
        for t in texts:
            for w in t.split():
                words.add(w.lower())

        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        vocab = special_tokens + sorted(words - set(special_tokens))

        self.word_to_id = {
            word : id for id, word in enumerate(vocab)
        }

        self.id_to_word = {
            id : word for word, id in self.word_to_id.items()
        }

        self.vocab_size = len(vocab)


        

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        unk = self.word_to_id[self.unk_token]
        
        return [
            self.word_to_id.get(w,unk)
            for w in text.lower().split()
        ]
        

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        return " ".join(
            self.id_to_word.get(id, self.unk_token)
            for id in ids
        )