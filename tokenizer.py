from dataclasses import dataclass
from typing import Dict, List

# char level tokenizer (details of how this works later)

@dataclass
class CharTokenizer:
    stoi: Dict[str, int] # str to int
    itos: Dict[int, str] # int to str

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
    
    @classmethod 
    def from_text(cls, text: str) -> "CharTokenizer":
        vocab = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)
    
    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s]
    
    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)
