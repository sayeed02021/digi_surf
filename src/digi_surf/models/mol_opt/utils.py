import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import random


from .dataset import OptDataset
from .models import EncoderDecoderTrfm




def get_split_dataset(
        df, 
        tokenizer,
        max_seq_len,
        props,
        save_folder,
        split_ratio=[0.9, 0.1],
        create_val=True,

        
):
    """
    Divides dataset into train and validation/test dataset according to split ratio provided
    """
    
    if create_val is True:
        src_id = df['Source_ID'].to_numpy()
        unique_src_id = np.unique(src_id)
        indices = np.random.permutation(len(unique_src_id))
        train_len =  int(split_ratio[0]*len(indices))
        train_indices = indices[:train_len]
        val_indices = indices[train_len:]
        train_src_id = unique_src_id[train_indices]
        val_src_id = unique_src_id[val_indices]
        all_train_indices = np.where(np.isin(src_id, train_src_id))[0]
        all_val_indices = np.where(np.isin(src_id, val_src_id))[0]

        np.save(f'{save_folder}/train_indices.npy', all_train_indices)
        np.save(f'{save_folder}/val_indices.npy', all_val_indices)
    else:
        all_train_indices = np.load(f'{save_folder}/train_indices.npy')
        all_val_indices = np.load(f'{save_folder}/val_indices.npy')

    src_mol_prop_header = [f'Source_Mol_{p}' for p in props]
    tgt_mol_prop_header = [f'Target_Mol_{p}' for p in props]
    src_p_all = torch.tensor(df[src_mol_prop_header].values, dtype=torch.float32)
    tgt_p_all = torch.tensor(df[tgt_mol_prop_header].values, dtype=torch.float32)


    val_src_mol = df['Source_Mol'].to_numpy()[all_val_indices]
    val_tgt_mol = df['Target_Mol'].to_numpy()[all_val_indices]
    val_src_p = src_p_all[all_val_indices,:]
    val_tgt_p = tgt_p_all[all_val_indices,:]
    
    train_src_mol = df['Source_Mol'].to_numpy()[all_train_indices]
    train_tgt_mol = df['Target_Mol'].to_numpy()[all_train_indices]
    train_src_p = src_p_all[all_train_indices,:]
    train_tgt_p = tgt_p_all[all_train_indices,:]
    
    src_mean = train_src_p.mean(dim=0)
    src_std = train_src_p.std(dim=0)
    tgt_mean = train_tgt_p.mean(dim=0)
    tgt_std = train_tgt_p.std(dim=0)

    all_stats = torch.stack((src_mean, src_std, tgt_mean, tgt_std), dim=0) # (4,num_prop)
    torch.save(all_stats, f'{save_folder}/train_scaler_values.pt')
    print("Scaler saved")

    

    train_dataset = OptDataset(
        src_smiles = train_src_mol,
        src_prop = train_src_p,
        tgt_smiles = train_tgt_mol,
        tgt_prop = train_tgt_p,
        tokenizer= tokenizer,
        max_seq_len= max_seq_len,
        scaler = all_stats
    )

    val_dataset = OptDataset(
        src_smiles = val_src_mol,
        src_prop = val_src_p,
        tgt_smiles = val_tgt_mol,
        tgt_prop = val_tgt_p,
        tokenizer= tokenizer,
        max_seq_len= max_seq_len,
        scaler = all_stats
    )
    

    return train_dataset, val_dataset, all_stats


def get_dataset(
        df, 
        tokenizer,
        max_seq_len,
        props,
        scaler

        
):
    """
    Returns pytorch dataset from df provided(No splitting)
    """
    src_mol_prop_header = [f'Source_Mol_{p}' for p in props]
    tgt_mol_prop_header = [f'Target_Mol_{p}' for p in props]
    src_p = torch.tensor(df[src_mol_prop_header].values, dtype=torch.float32)
    tgt_p = torch.tensor(df[tgt_mol_prop_header].values, dtype=torch.float32)

    src_mol = df['Source_Mol'].to_numpy()
    tgt_mol = df['Target_Mol'].to_numpy()

    dataset = OptDataset(
        src_smiles = src_mol,
        src_prop = src_p,
        tgt_smiles = tgt_mol,
        tgt_prop = tgt_p,
        tokenizer= tokenizer,
        max_seq_len= max_seq_len,
        scaler = scaler
    )

    return dataset



def make_model(args, tokenizer, path=None):

    model = EncoderDecoderTrfm(
        num_embeddings=tokenizer.vocab_size,
        n_prop=len(args.props),
        padding_idx=tokenizer.pad_idx,
        d_model = args.d_model,
        nhead = args.nhead,
        dim_feedforward=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len
    )
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    model = model.to(args.device)
    return model

def seed_everything(seed: int):
    """
    Set seed for reproducibility across:
    - Python random
    - NumPy
    - PyTorch (CPU + CUDA)
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic CUDA ops (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # # Optional but recommended for newer PyTorch versions
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except Exception:
    #     pass

def save_generated_data(generated_data, loader):
    data = defaultdict(list)
    for data_batch, gen_batch in zip(loader, generated_data):
        src, tgt, p = data_batch 
        tgt2, gen_smiles_per_batch = gen_batch 

        assert (tgt2==tgt)

        tgt_prop = p[:,0]
        src_prop = p[:,1]

        for gen_smiles_per_tgt, src_smile, tgt_smile, prop in zip(gen_smiles_per_batch, src, tgt, p):
            data['Source_Mol'].append(src_smile)
            data['Target_Mol'].append(tgt_smile)
            data['Source_Mol_pCMC'].append(prop[1])
            data['Target_Mol_pCMC'].append(prop[0])
            for idx, gen_smi in enumerate(gen_smiles_per_tgt):
                data[f'Predicted_smi_{idx+1}'].append(gen_smi)

    df = pd.DataFrame(data)
    return df



class Logger():
    def __init__(self):
        self.logger = defaultdict(list)

    def log(self, val, val_key):
        self.logger[val_key].append(val)

    def save_log(self, save_path):
        df = pd.DataFrame(self.logger)
        df.to_csv(save_path, index=False)
    
    def plot(self, val_key, save_path):
        
        
        n_epoch = len(self.logger[val_key[0]])
        for key in val_key:
            if key not in self.logger.keys():
                print(f'{key} not logged! So not plotting!')
                continue
            plt.plot(np.arange(n_epoch)+1, self.logger[key], 'o--', label=key, alpha=0.4)
            plt.xlabel('Epoch')
            # plt.ylabel(val_key)
            plt.grid(True)
            plt.legend()
        plt.savefig(save_path)
        plt.close()






if __name__=="__main__":
    df_path = "final_mmps_single.csv"
    get_split_dataset(pd.read_csv(df_path), tokenizer=None)

        




    


