import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import glob
from pathlib import Path

from .mol_opt.utils import make_model
from .mol_opt.build_vocab import Tokenizer
from .mol_opt.generate import generate_sequences
from .mol_opt.dataset import OptDataset
from ..utils import load_yaml, load_pickle

BASE_DIR = Path(__file__).resolve().parent
class TrfmGenerator(object):
    def __init__(
            self,
            mode='single',
            device = 'cpu',

    ):
        """
        model_path: path to saved model params
        device: cuda or cpu
        """
        self.device = device
        
        config_path = os.path.join('mol_opt', mode, 'configs.yaml')
        config_path = str(BASE_DIR / config_path)

        
        self.args = load_yaml(config_path)
        self.args.device = self.device

        self.tokenizer = Tokenizer()
        self.tokenizer.load_vocab(str(BASE_DIR / os.path.join('mol_opt', 'data','vocab.pkl')))

        model_path = BASE_DIR / os.path.join('model_paths', f'trfm_{mode}.pt')
        
        self.model = make_model(
            args=self.args,
            tokenizer=self.tokenizer,
            path=model_path
        )

        scaler_path = os.path.join('mol_opt', mode, 'train_scaler_values.pt')
        scaler_path = str(BASE_DIR / scaler_path)
        
        self.scaler = torch.load(scaler_path, map_location='cpu', weights_only=False)

    def __call__(self, src_smi, src_p, tgt_p, n_gen):
        """
        src_smi: list (N,)

        src_p: torch.Tensor (N,num_prop)
        
        tgt_p: torch.Tensor (N,num_prop)
        """

        dataset = OptDataset(
            src_smiles=src_smi,
            src_prop=src_p,
            tgt_smiles=src_smi, # tgt smiles are not known so taken as src_smi only
            tgt_prop = tgt_p,
            tokenizer=self.tokenizer,
            max_seq_len=self.args.max_seq_len,
            scaler = self.scaler
        )
        loader = DataLoader(dataset, batch_size=10)

        all_src, all_tgt, all_gen, all_tgt_p, all_src_p = generate_sequences(
            model=self.model,
            loader=loader, 
            device=self.device,
            tokenizer=self.tokenizer,
            max_seq_len=self.args.max_seq_len,
            scaler=self.scaler,
            n_gen=n_gen
        )

        return all_src, all_gen, all_src_p, all_tgt_p
    

        

        
            
        