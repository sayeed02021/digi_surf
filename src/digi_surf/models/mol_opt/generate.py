import torch
from tqdm import tqdm
import numpy as np
from rdkit import Chem


# def generate_sequences(model, loader, 
#                        device, tokenizer, 
#                        max_seq_len,
#                        scaler,
#                        n_gen=1, temp=1.0):
#     model.eval()
#     sos_id = tokenizer.tok2idx[tokenizer.sos_token]
#     eos_id = tokenizer.tok2idx[tokenizer.eos_token]
#     all_tgt = []
#     all_src = []
#     all_gen = []
#     all_tgt_prop = []
#     all_src_prop = []
#     with torch.no_grad():
#         for src, tgt, p in loader:
#             src,p = src.to(device), p.to(device)
#             B, n_prop, _ = p.shape # p shape : B, n_prop, 3
#             input_tgt = torch.full(
#                 size=(B*n_gen,1),
#                 fill_value=sos_id,
#                 dtype=torch.long,
#                 device=device
#             )
#             src_expanded = src.unsqueeze(1).repeat(1,n_gen,1).reshape(B*n_gen,-1)
#             tgt_expanded = tgt.unsqueeze(1).repeat(1,n_gen,1).reshape(B*n_gen,-1) 
            
#             p_expanded = p.unsqueeze(1).repeat(1,n_gen,1,1).reshape(B*n_gen,n_prop,-1) # b*n_gen, n_prop, 3

#             for _ in range(max_seq_len-1):
                
#                 output = model(src_expanded, input_tgt, p_expanded) 
                
#                 next_token_dist = output[:,-1,:]
                
                
#                 next_token_samples = torch.multinomial(torch.softmax(next_token_dist/temp, dim=-1), num_samples=1)
                
#                 input_tgt = torch.cat([input_tgt, next_token_samples], dim=1)       

            
#             tgt_smi = np.array([tokenizer.decode(t.cpu().tolist()) for t in tgt_expanded]) # (B*n_gen,)
#             src_smi = np.array([tokenizer.decode(s.cpu().tolist()) for s in src_expanded]) # (B*n_gen,)
#             pred_smi = np.array([tokenizer.decode(g.cpu().tolist()) for g in input_tgt]) # (B*n_gen,) 

#             for s_smi,t_smi,p_smi, all_p in zip(src_smi, tgt_smi, pred_smi, p_expanded):
#                 if p_smi is not None:
#                     mol = Chem.MolFromSmiles(p_smi)
#                     if mol is not None:
#                         all_gen.append(p_smi)
#                         all_src.append(s_smi)
#                         all_tgt.append(t_smi)
                        
#                         all_tgt_prop.append(all_p[:,0].cpu()) # all_p shape: (n_prop,3)
#                         all_src_prop.append(all_p[:,1].cpu())

#     all_tgt_prop = torch.stack(all_tgt_prop, dim=0) # (N_valid, n_prop)
#     all_src_prop = torch.stack(all_src_prop, dim=0)
#     all_src_prop = all_src_prop*(scaler[1,:]+1e-8) + scaler[0,:]
#     all_tgt_prop = all_tgt_prop*(scaler[3,:]+1e-8) + scaler[2,:]
#     # print("After Scaling")
#     # print(all_tgt_prop)

#     # print(all_tgt_prop)
#     # print(all_src_prop)

#     return all_src, all_tgt, all_gen, all_tgt_prop, all_src_prop
            

def generate_sequences(model, loader, 
                       device, tokenizer, 
                       max_seq_len,
                       scaler,
                       n_gen=1, temp=1.0,
                       max_tries=100):
    model.eval()
    sos_id = tokenizer.tok2idx[tokenizer.sos_token]
    eos_id = tokenizer.tok2idx[tokenizer.eos_token]
    all_tgt = []
    all_src = []
    all_gen = []
    all_tgt_prop = []
    all_src_prop = []
    with torch.no_grad():
        for src, tgt, p in tqdm(loader, dynamic_ncols=True, leave=False):
            src, p = src.to(device), p.to(device)
            B, n_prop, _ = p.shape # p shape : B, n_prop, 3

            # results[b][slot] = valid smi string once found, else None
            results      = [[None] * n_gen for _ in range(B)]
            # trials[b][slot] = number of tries used for that slot
            trials       = [[0]    * n_gen for _ in range(B)]
            # store p_expanded per (b, slot) for property collection later
            p_expanded_store = [[None] * n_gen for _ in range(B)]

            src_expanded = src.unsqueeze(1).repeat(1, n_gen, 1).reshape(B*n_gen, -1)
            tgt_expanded = tgt.unsqueeze(1).repeat(1, n_gen, 1).reshape(B*n_gen, -1)
            p_expanded   = p.unsqueeze(1).repeat(1, n_gen, 1, 1).reshape(B*n_gen, n_prop, -1) # b*n_gen, n_prop, 3

            # store tgt/src smi once — they don't change across retries
            tgt_smi_expanded = np.array([tokenizer.decode(t.cpu().tolist()) for t in tgt_expanded]) # (B*n_gen,)
            src_smi_expanded = np.array([tokenizer.decode(s.cpu().tolist()) for s in src_expanded]) # (B*n_gen,)

            # store p_expanded per (b, slot)
            for b in range(B):
                for slot in range(n_gen):
                    p_expanded_store[b][slot] = p_expanded[b * n_gen + slot]

            while True:
                # collect which (b, slot) pairs still need an attempt
                pending = []
                for b in range(B):
                    for slot in range(n_gen):
                        if results[b][slot] is None and trials[b][slot] < max_tries:
                            pending.append((b, slot))

                if not pending:
                    break

                # build a batch of src/p/tgt for all pending pairs
                N = len(pending)
                pending_src = torch.stack([src_expanded[b * n_gen + slot] for b, slot in pending]) # (N, seq_len)
                pending_p   = torch.stack([p_expanded[b * n_gen + slot]   for b, slot in pending]) # (N, n_prop, 3)

                input_tgt = torch.full(
                    size=(N, 1),
                    fill_value=sos_id,
                    dtype=torch.long,
                    device=device
                )

                for _ in range(max_seq_len - 1):
                    output = model(pending_src, input_tgt, pending_p)

                    next_token_dist = output[:, -1, :]

                    next_token_samples = torch.multinomial(torch.softmax(next_token_dist / temp, dim=-1), num_samples=1)

                    input_tgt = torch.cat([input_tgt, next_token_samples], dim=1)

                    # early stopping: if all sequences have hit eos, no need to continue
                    if (input_tgt == eos_id).any(dim=1).all():
                        break

                pred_smi = np.array([tokenizer.decode(g.cpu().tolist()) for g in input_tgt]) # (N,)

                # validate each pending sequence
                for i, (b, slot) in enumerate(pending):
                    trials[b][slot] += 1
                    p_smi = pred_smi[i]
                    mol = Chem.MolFromSmiles(p_smi)
                    if mol is not None:
                        results[b][slot] = p_smi  # slot is now filled

            # collect valid results
            for b in range(B):
                for slot in range(n_gen):
                    p_smi  = results[b][slot]
                    s_smi  = src_smi_expanded[b * n_gen + slot]
                    t_smi  = tgt_smi_expanded[b * n_gen + slot]
                    all_p  = p_expanded_store[b][slot] # (n_prop, 3)

                    if p_smi is not None:
                        all_gen.append(p_smi)
                        all_src.append(s_smi)
                        all_tgt.append(t_smi)

                        all_tgt_prop.append(all_p[:, 0].cpu()) # all_p shape: (n_prop, 3)
                        all_src_prop.append(all_p[:, 1].cpu())

    all_tgt_prop = torch.stack(all_tgt_prop, dim=0) # (N_valid, n_prop)
    all_src_prop = torch.stack(all_src_prop, dim=0)
    all_src_prop = all_src_prop * (scaler[1, :] + 1e-8) + scaler[0, :]
    all_tgt_prop = all_tgt_prop * (scaler[3, :] + 1e-8) + scaler[2, :]

    return all_src, all_tgt, all_gen, all_tgt_prop, all_src_prop

# if __name__=="__main__":
#     from .dataset import OptDataset
#     from torch.utils.data import DataLoader
#     import pandas as pd
#     from .build_vocab import Tokenizer
#     from .utils import make_model
#     import argparse
#     import yaml
    


#     df = pd.read_csv("data/final_mmps_single.csv")

#     args = load_yaml('single/configs.yaml')
#     args.device = 'cpu'

#     src_smi = df['Source_Mol'].to_numpy()
#     tgt_smi = df['Target_Mol'].to_numpy()
#     src_p = torch.tensor(df['Source_Mol_pCMC'].to_numpy(), dtype=torch.float32)
#     tgt_p = torch.tensor(df['Target_Mol_pCMC'].to_numpy(), dtype=torch.float32)
#     scaler = torch.load('single/train_scaler_values.pt', map_location='cpu', weights_only=False)

#     tokenizer = Tokenizer()
#     tokenizer.load_vocab('data/vocab.pkl')

#     if src_p.ndim==1:
#         src_p = src_p.unsqueeze(-1)
#     if tgt_p.ndim==1:
#         tgt_p = tgt_p.unsqueeze(-1)

#     dataset = OptDataset(
#         src_smiles = src_smi[:50],
#         src_prop=src_p[:50],
#         tgt_smiles=tgt_smi[:50],
#         tgt_prop=tgt_p[:50],
#         tokenizer=tokenizer,
#         max_seq_len=200,
#         scaler=scaler
#     )

#     loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
#     model = make_model(
#         args=args,
#         tokenizer=tokenizer
#     )
#     print('generating sequences')
#     s, t, p = generate_sequences(
#         model, loader, 'cpu', tokenizer, args.max_seq_len, scaler=scaler, n_gen=10
#     )