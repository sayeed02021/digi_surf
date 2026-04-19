import re
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Tokenizer:

    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }

    REGEXP_ORDER = ["brackets", "ring_nums", "brcl"]

    def __init__(self):

        self.pad_token = "*"
        self.sos_token = "^"
        self.eos_token = "$"

        self.tok2idx = {}
        self.idx2tok = {}

    # ---------------------------------------------------
    # Core SMILES tokenization
    # ---------------------------------------------------

    def _split_by(self, data, regexps):

        if not regexps:
            return list(data)

        regexp = self.REGEXPS[regexps[0]]
        splitted = regexp.split(data)

        tokens = []

        for i, split in enumerate(splitted):
            if i % 2 == 0:
                tokens += self._split_by(split, regexps[1:])
            else:
                tokens.append(split)

        return tokens

    def str2token(self, smile, add_special=True):

        tokens = self._split_by(smile, self.REGEXP_ORDER)

        if add_special:
            tokens = [self.sos_token] + tokens + [self.eos_token]

        return tokens

    # ---------------------------------------------------
    # Vocabulary building
    # ---------------------------------------------------

    def build_vocab(self, smiles_list):

        tokens = set()

        for smi in smiles_list:
            tokens.update(self.str2token(smi, add_special=False))

        vocab = [self.pad_token, self.sos_token, self.eos_token] + sorted(tokens)

        self.tok2idx = {tok: i for i, tok in enumerate(vocab)}
        self.idx2tok = {i: tok for tok, i in self.tok2idx.items()}

    # ---------------------------------------------------
    # Encoding
    # ---------------------------------------------------

    def encode(self, smile, max_len=None):

        tokens = self.str2token(smile)

        ids = [self.tok2idx[t] if t in self.tok2idx else self.pad_idx for t in tokens]

        if max_len is not None:

            ids = ids[:max_len]

            if len(ids) < max_len:
                ids += [self.tok2idx[self.pad_token]] * (max_len - len(ids))

        return ids

    # ---------------------------------------------------
    # Decoding
    # ---------------------------------------------------

    def decode(self, ids):

        tokens = [self.idx2tok[i] for i in ids]

        smile = ""

        for tok in tokens:
            if tok == self.eos_token:
                break
            if tok not in [self.sos_token, self.pad_token]:
                smile += tok

        return smile


    def save_vocab(self, path):

        with open(path, "wb") as f:
            pickle.dump(self.tok2idx, f)

    def load_vocab(self, path):

        with open(path, "rb") as f:
            self.tok2idx = pickle.load(f)

        self.idx2tok = {v: k for k, v in self.tok2idx.items()}

    @property
    def vocab_size(self):
        return len(self.tok2idx)

    @property
    def pad_idx(self):
        return self.tok2idx[self.pad_token]

# if __name__=="__main__":
#     import pandas as pd
#     df = pd.read_csv('final_mmps_single.csv')
#     source_mol = df['Source_Mol'].tolist()
#     target_mol = df['Target_Mol'].tolist()
#     all_mol = source_mol+target_mol
    
#     tokenizer = Tokenizer()
#     tokenizer.build_vocab(all_mol)
#     print(tokenizer.encode(all_mol[23]))
#     print(tokenizer.decode(tokenizer.encode(all_mol[23])))
#     tokenizer.save_vocab('vocab.pkl')
#     print(tokenizer.tok2idx)


