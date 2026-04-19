import torch
from torch.utils.data import Dataset, DataLoader

class OptDataset(Dataset):
    def __init__(
            self,
            src_smiles, 
            src_prop,
            tgt_smiles,
            tgt_prop,
            tokenizer,
            max_seq_len, 
            scaler

    ):
        """
        src_smiles: np.array() (N,)
        tgt_prop: torch.tensor (N,num_prop)
        tgt_smiles: np.array() (N,)
        tgt_prop: torch.tensor (N,num_prop)
        tokenizer: build_vocab.Tokenizer()
        max_seq_len: int
        scaler: torch.tensor
        """
        
        self.src_smiles = src_smiles
        self.tgt_smiles = tgt_smiles
        self.src_prop = (src_prop-scaler[0,:])/(scaler[1,:]+1e-8)
        self.tgt_prop = (tgt_prop-scaler[2,:])/(scaler[3,:]+1e-8)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        self.src_tokens = []
        self.tgt_tokens = []
        for (src, tgt) in zip(self.src_smiles, self.tgt_smiles):
            self.src_tokens.append(torch.tensor(self.tokenizer.encode(src, max_len=self.max_seq_len)))
            self.tgt_tokens.append(torch.tensor(self.tokenizer.encode(tgt, max_len = self.max_seq_len)))
        
        self.diff = self.tgt_prop-self.src_prop
        self.all_prop = torch.stack((self.tgt_prop, self.src_prop, self.diff), dim=-1) # N,num_prop,3


    def __len__(self):
        return len(self.src_smiles)
    
    def __getitem__(self, index):

        # print(self.all_prop.shape)
        return self.src_tokens[index], self.tgt_tokens[index], self.all_prop[index, :, :]
    

if __name__=="__main__":
    import pandas as pd
    from build_vocab import Tokenizer
    df = pd.read_csv("data/final_mmps_single.csv")

    src_smi = df['Source_Mol'].to_numpy()
    tgt_smi = df['Target_Mol'].to_numpy()
    src_p = torch.tensor(df['Source_Mol_pCMC'].to_numpy(), dtype=torch.float32)
    tgt_p = torch.tensor(df['Target_Mol_pCMC'].to_numpy(), dtype=torch.float32)
    scaler = torch.load('single/train_scaler_values.pt', map_location='cpu', weights_only=False)

    tokenizer = Tokenizer()
    tokenizer.load_vocab('data/vocab.pkl')

    if src_p.ndim==1:
        src_p = src_p.unsqueeze(-1)
    if tgt_p.ndim==1:
        tgt_p = tgt_p.unsqueeze(-1)

    dataset = OptDataset(
        src_smiles = src_smi,
        src_prop=src_p,
        tgt_smiles=tgt_smi,
        tgt_prop=tgt_p,
        tokenizer=tokenizer,
        max_seq_len=200,
        scaler=scaler
    )

    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for src, tgt, p in loader:
        print(src.shape)
        print(tgt.shape)
        print(p.shape)
        