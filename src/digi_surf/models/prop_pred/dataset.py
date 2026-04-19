import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"{x} not in allowable set: {allowable_set}")
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps input not in set to the last element"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set] # one hot encoding returned


def atom_features(atom, explicit_H=False, use_chirality=True):
    features = (
        one_of_k_encoding_unk(atom.GetSymbol(), [
            "B", "C", "N", "O", "F", "Si", "P", "S", "Cl",
            "As", "Se", "Br", "Te", "I", "At", "other"
        ])
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            "other"
        ])
        + [atom.GetIsAromatic()]
    )
    if not explicit_H:
        features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            features += one_of_k_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"])
            features += [atom.HasProp("_ChiralityPossible")]
        except:
            features += [False, False]
            features += [atom.HasProp("_ChiralityPossible")]
    return np.array(features, dtype=np.float32)


def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        feats += one_of_k_encoding_unk(str(bond.GetStereo()), [
            "STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"
        ])
    return np.array(feats, dtype=np.float32)

class SMILESDataset(Dataset):
    def __init__(
            self, smiles, index, mode='train',
            feat=['pCMC', 'AW_ST_CMC', 'Gamma_max'],
            y_val=None, fold_values=None,
            scaler=None, scale_data=False,
            source_data = None, transform=None):
        
        super().__init__(None, transform=transform)

        self.mode = mode
        self.feat = feat
        self.scale_data = scale_data
        self.smiles = smiles
        self.index = index
        self.fold_values = fold_values
        self.source_data = source_data

        if y_val is not None:
            assert y_val.shape[1]==len(feat)
            self.y_val = y_val
        else:
            self.y_val = np.zeros((len(smiles), len(feat)))

        if self.mode == 'train' and scale_data:
            self.scaler = RobustScaler()
            self.scaler.fit(self.y_val)
        else:
            self.scaler = scaler

        self.masks = np.where(np.isnan(self.y_val), 0, 1)
        if self.scale_data:
            scaled = self.scaler.transform(self.y_val)
            self.scaled_labels = np.nan_to_num(scaled, nan=0.0)
        else:
            self.scaled_labels = np.nan_to_num(self.y_val, nan=0.0)

        self.data_list = []
        # print('Converting smiles to graphs')
        for idx in tqdm(range(len(self.smiles))):
            
            data = self.smile_to_graph(
                smile=self.smiles[idx], 
                y_val= self.scaled_labels[idx],
                mask = self.masks[idx]
            )
            if data is None:
                # print("Continuing")
                continue
            if data is not None:
                data.fold = self.fold_values[idx] if self.fold_values is not None else -1
                data.index = idx

            # optional
            if self.source_data is not None:
                data.source_data_exists = True
                for key in self.source_data.keys():
                    data[key] = self.source_data[key][idx]
                data.source_data_keys = list(self.source_data.keys())
            else:
                data.source_data_exists = torch.tensor(0.0)

            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def smile_to_graph(self, smile, y_val, mask, add_self_loops=True):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        x = torch.tensor(
            np.array([atom_features(atom) for atom in mol.GetAtoms()])
        )
        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            feat = bond_features(bond)
            edge_index += [[i,j], [j,i]]
            edge_attr +=[feat, feat]
        if edge_attr == []:
            # print(smile)
            return None
        if add_self_loops:
            num_atoms = mol.GetNumAtoms()
            for i in range(num_atoms):
                edge_index.append([i,i])
                edge_attr.append(np.zeros(len(edge_attr[0]), dtype=np.float32))
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

            # mark self loops
            is_self = (edge_index[0] == edge_index[1]).float().unsqueeze(1)
            edge_attr = torch.cat([edge_attr, is_self], dim=1)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        y = torch.tensor(y_val, dtype=torch.float).unsqueeze(0)
        mask = torch.tensor(mask, dtype = torch.bool).unsqueeze(0)

        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, y=y
        )
        data.smile = smile
        data.mask = mask
        return data


if __name__=="__main__":

    train_df = pd.read_csv('data/surfpro_train.csv')
    test_df = pd.read_csv('data/surfpro_test.csv')

    train_smiles = train_df['SMILES'].to_numpy()
    train_pcmc = train_df['pCMC'].to_numpy()
    train_pcmc = np.reshape(train_pcmc, (len(train_pcmc),1))

    dataset = SMILESDataset(
        smiles = train_smiles, index = np.arange(len(train_smiles)),
        mode = 'train', feat=['pCMC'], y_val = train_pcmc,
        fold_values = train_df['fold'].to_numpy(),
        scaler=None, scale_data=True
    )

    # print(dataset[0])


