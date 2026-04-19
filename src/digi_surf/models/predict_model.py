import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

import pickle
import numpy as np
import os
import torch
import glob
import yaml
import argparse
from torch_geometric.loader import DataLoader
from pathlib import Path

from .prop_pred.dataset import SMILESDataset
from .prop_pred.model import AttentiveFPModel


BASE_DIR = Path(__file__).resolve().parent

def create_dataset(
        generated_smiles, target_properties, 
        features = ['pCMC']
):
    """
    Create a smiles dataset given the input smiles and their target properties
    generated_smiles: (N,) smiles of generated molecules
    target_properties: (N,num_prop) TARGET properties
    """
    if not isinstance(target_properties, torch.Tensor):
        target_properties = torch.tensor(target_properties, dtype=torch.float32)
    if target_properties.ndim==1:
        target_properties = target_properties.unsqueeze(-1)

    dataset = SMILESDataset(
        smiles=generated_smiles,
        index= np.arange(len(generated_smiles)),
        mode='predict',
        feat = features,
        y_val = target_properties,
        scale_data=False,
        scaler=None, source_data=None, 
        transform=None
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader





class PredictionModel(object):
    def __init__(
            self , 
            device, 
            n_device=1
    ):  
        self.device = device

        config_path = BASE_DIR / os.path.join('prop_pred', 'config.yaml')
        with open(str(config_path), 'r') as f:
            config_data = yaml.safe_load(f)
        self.args = argparse.Namespace(**config_data)

        self.model = AttentiveFPModel(
            props = ['pCMC', 'AW_ST_CMC', 'Area_min'],
            args=self.args
        ).to(self.device) 
        
        model_folder = str(BASE_DIR / 'model_paths/prop_pred/*.pt')
        self.model_paths = glob.glob(model_folder)

        scaler_path = str(BASE_DIR / os.path.join('prop_pred', 'robust_scaler.pkl'))
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def predict(self, model, loader):
        all_smiles = []
        all_targets = []
        ensemble_preds = []

        for batch in loader:

            smiles = batch.smile
            targets = batch.y
            batch = batch.to(self.device)

            preds_models = []

            for path in self.model_paths:
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.to(self.device)
                model.eval()

                with torch.no_grad():
                    preds = model(batch)
                    preds = self.scaler.inverse_transform(preds.cpu())
                    preds = torch.from_numpy(preds).float()

                preds_models.append(preds)

            preds_models = torch.stack(preds_models)      # [n_models, batch, n_prop]
            preds_mean = preds_models.mean(dim=0)         # [batch, n_prop]

            ensemble_preds.append(preds_mean)
            all_targets.append(targets)
            all_smiles.extend(smiles)

        all_preds = torch.cat(ensemble_preds, dim=0) # [N, n_prop]
        all_targets = torch.cat(all_targets, dim=0)

        return all_smiles, all_targets, all_preds
        

    def __call__(self, smiles, target_properties, features):
        
        loader = create_dataset(
            generated_smiles=smiles,
            target_properties=target_properties,
            features = features
        )
        all_smi, all_targets, all_preds = self.predict(self.model, loader)
        return all_smi, all_targets, all_preds




    