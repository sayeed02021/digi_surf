import os
import torch
import rdkit.Chem as Chem
from rdkit.Contrib.SA_Score import sascorer
from pathlib import Path
import time
import gdown
import numpy as np


from .models.diff_generator import DiffGen
from .models.trfm_model import TrfmGenerator
from .models.predict_model import PredictionModel
from .models.scscore import SCScorer

def download_gdrive_file(url, output_path):
    gdown.download(
        gdown.download(url, output_path, quiet=False)
    )

BASE_DIR = Path(__file__).resolve().parent

class SurfGen(object):
    """
    diff_single: https://drive.google.com/file/d/1qifIjpvwlSi_4q41_sWyjzN970L4w3Mf/view?usp=drive_link
    diff_multi: https://drive.google.com/file/d/13QAitIb2_T7QWL_mVuoDe6vsmAsBWgSm/view?usp=drive_link
    scscore: https://drive.google.com/file/d/1G9ZtPWVZvuKd2HB5uvgZlyY03DF78_tz/view?usp=drive_link
    trfm_single: https://drive.google.com/file/d/1ANC4hKCeOLwaQWnJyiCoO5q7Mj98vV78/view?usp=drive_link
    trfm_multi: https://drive.google.com/file/d/1gWXNuisfL9KyGQRE6ZOfEwYV5wILmvKU/view?usp=drive_link
    prop_pred: https://drive.google.com/drive/folders/1z4VUHu6-yFeeMGNiBYH-nv6h7vpnAkvs?usp=drive_link
    
    """


    def __init__(self, device):
        self.device = torch.device(device)

        self.diff_single_id = '1qifIjpvwlSi_4q41_sWyjzN970L4w3Mf'
        self.diff_multi_id = '13QAitIb2_T7QWL_mVuoDe6vsmAsBWgSm'
        self.trfm_single_id = '1ANC4hKCeOLwaQWnJyiCoO5q7Mj98vV78'
        self.trfm_multi_id = '1gWXNuisfL9KyGQRE6ZOfEwYV5wILmvKU'
        self.prop_folder_id = '1z4VUHu6-yFeeMGNiBYH-nv6h7vpnAkvs'
        self.scscore_id = '1G9ZtPWVZvuKd2HB5uvgZlyY03DF78_tz'


        
    
    def load_models(self, mode):
        

        if mode=='multi':
            
            diff_out_path = str(BASE_DIR / os.path.join('models', 'model_paths', 'diff_multi.pt'))
            trfm_out_path = str(BASE_DIR / os.path.join('models', 'model_paths', 'trfm_multi.pt'))
            if not os.path.exists(diff_out_path):
                print("Downloading diffusion model weights from drive")
                gdown.download(
                    url=f"https://drive.google.com/uc?id={self.diff_multi_id}",
                    output=diff_out_path,
                    quiet=False
                )
            
            self.diff_model = DiffGen(
                model_path=diff_out_path,
                properties_type=3*['regression'],
                device=self.device
            )

            if not os.path.exists(trfm_out_path):
                print("Downloading transformer model weights from drive")
                gdown.download(
                    url=f"https://drive.google.com/uc?id={self.trfm_multi_id}",
                    output=trfm_out_path,
                    quiet=False
                )
            
            self.trfm_model = TrfmGenerator(
                mode='multi', device=self.device
            )

        else:
            diff_out_path = str(BASE_DIR / os.path.join('models', 'model_paths', 'diff_single.pt'))
            trfm_out_path = str(BASE_DIR / os.path.join('models', 'model_paths', 'trfm_single.pt'))
            

            if not os.path.exists(diff_out_path):
                print("Downloading diffusion model weights from drive")
                gdown.download(
                    url=f"https://drive.google.com/uc?id={self.diff_single_id}",
                    output=diff_out_path,
                    quiet=True
                )
            self.diff_model = DiffGen(
                model_path=diff_out_path,
                properties_type=['regression'],
                device=self.device
            )
            
            
            if not os.path.exists(trfm_out_path):
                print("Downloading transformer model weights from drive")
                gdown.download(
                    url=f"https://drive.google.com/uc?id={self.trfm_single_id}",
                    output=trfm_out_path,
                    quiet=True
                )
            self.trfm_model = TrfmGenerator(
                mode='single', device=self.device
            )

        scscore_model_weight_path = str(BASE_DIR / os.path.join('models', 'model_paths', 'model.ckpt-10654.as_numpy.json.gz'))
        if not os.path.exists(scscore_model_weight_path):
            print("Downloading model weights for scscore")
            gdown.download(
                url=f"https://drive.google.com/uc?id={self.scscore_id}",
                output=scscore_model_weight_path,
                quiet=True
            )
        self.scscore_model = SCScorer()
        self.scscore_model.restore(
            weight_path=scscore_model_weight_path
        )


        prediction_folder = str(BASE_DIR / os.path.join('models', 'model_paths', 'prop_pred'))
        if not os.path.exists(prediction_folder):
            os.makedirs(prediction_folder, exist_ok=True)
            print("Downloading property prediction models")
            gdown.download_folder(
                url = f'https://drive.google.com/drive/folders/{self.prop_folder_id}',
                output= prediction_folder,
                quiet=True
            )
            


        self.prediction_model = PredictionModel(
            device=self.device, n_device=1
        )
        

    def diff_gen(self, constraints, n_gen):
        """
        Uses constraints to generate new molecules using diffusion
        constraints: (N,n_prop)
        n_gen: int
        """

        num_samples, n_prop = constraints.shape
        if n_prop!=1 and n_prop!=3:
            raise ValueError("Diffusion models only trained for either pCMC or pCMC, AW_ST_CMC, Area_min")
        
        
        all_smi, all_p = self.diff_model(constraints, n_gen)

        return all_smi, all_p
            

    
    def trfm_gen(self, src_smi, src_p, tgt_p, n_gen):
        N,n_prop = src_p.shape
        
        if not isinstance(src_p, torch.Tensor):
            src_p = torch.tensor(src_p)


        if not isinstance(tgt_p, torch.Tensor):
            tgt_p = torch.tensor(tgt_p)

        
        if n_prop!=1 and n_prop!=3:
            raise ValueError("Diffusion models only trained for either pCMC or pCMC, AW_ST_CMC, Area_min")
        
        
        all_src, all_gen, all_src_p, all_tgt_p = self.trfm_model(
                src_smi = src_smi,
                src_p = src_p,
                tgt_p = tgt_p,
                n_gen = n_gen
        )

        return all_src, all_gen, all_src_p, all_tgt_p
    

    def __call__(self, pcmc, 
                 aw_st_cmc=None,
                 area_min = None,
                 trfm_cycles = 2):
        """
        Combines the entire pipeline together, from generating using 
        diffusion to optimizing using transformer
        
        constraints: np.arrary: (N,num_prop)
        """
        if aw_st_cmc is None and area_min is not None:
            raise ValueError('Both surface tension and area min have to be specified along with pCMC')
        elif aw_st_cmc is not None and area_min is None:
            raise ValueError('Both surface tension and area min have to be specified along with pCMC')
        elif aw_st_cmc is not None and area_min is not None:
            n_prop = 3
            features = ['pCMC', 'AW_ST_CMC', 'Area_min']
            pcmc = torch.tensor(pcmc)
            aw_st_cmc = torch.tensor(aw_st_cmc)
            area_min = torch.tensor(area_min)
            constraints = torch.stack((pcmc, aw_st_cmc, area_min), dim=-1)
            
        elif aw_st_cmc is None and area_min is None:
            n_prop = 1
            features = ['pCMC']
            pcmc = torch.tensor(pcmc)
            if pcmc.ndim==1:
                constraints = pcmc.unsqueeze(-1)
            else:
                constraints=pcmc

        if n_prop==1:
            self.load_models(mode='single')
        else:
            self.load_models(mode='multi')
        
        print("Generating molecules using inverse design")
        start = time.time()
        diff_smiles, all_tgt_prop = self.diff_gen(
            constraints=constraints, n_gen=2
        )

        if len(diff_smiles)==0:
            print("No valid smiles were generated, trying to generate more smiles")
            while len(diff_smiles)==0:
                diff_smiles, all_tgt_prop = self.diff_gen(
                constraints=constraints, n_gen=5
        )

        all_gen_diff, all_tgt_prop, all_src_prop = self.prediction_model(diff_smiles, target_properties=all_tgt_prop, features = features)
        diff_src_p = all_src_prop
        diff_tgt_p = all_tgt_prop
        if n_prop==1:
            src_p = all_src_prop[:,0].unsqueeze(-1)
        else:
            src_p = all_src_prop



        print("Optimizing generated molecules using transformer")
        current_gen = all_gen_diff
        current_src_p = src_p
        current_tgt_p  = all_tgt_prop
        for cycle in range(trfm_cycles):
            
            all_src, all_gen_trfm, all_src_p, all_tgt_p = self.trfm_gen(
                src_smi=current_gen, 
                src_p = current_src_p,
                tgt_p = current_tgt_p,
                n_gen=5
            )
            
            all_gen_trfm = np.array(all_gen_trfm)
            unique_smi = np.unique(all_gen_trfm)
            unique_tgt_p = torch.zeros((len(unique_smi), n_prop), dtype=torch.float32)
            
            for smi_idx,smi in enumerate(unique_smi):
                idx = np.where(all_gen_trfm==smi)[0][0]
                unique_tgt_p[smi_idx, :] = all_tgt_p[idx, :]
            all_tgt_p = unique_tgt_p
            all_gen_trfm = unique_smi
    
        
            all_gen_trfm, all_tgt_trfm, all_preds_trfm = self.prediction_model(
                all_gen_trfm, all_tgt_p, features=features
            )

            if n_prop==1:
                current_src_p = all_preds_trfm[:,0].unsqueeze(-1)
            else:
                current_src_p = all_preds_trfm
            
            current_tgt_p = all_tgt_trfm
            current_gen = all_gen_trfm



        total_time = time.time()-start




        print(f'Generated {len(all_gen_trfm)} molecules in {total_time:0.3f} seconds')


        all_gen_diff, all_tgt_diff, all_preds_diff = self.sort_data(all_gen_diff, diff_tgt_p, diff_src_p)
        all_gen_trfm, all_tgt_trfm, all_preds_trfm = self.sort_data(all_gen_trfm, all_tgt_trfm, all_preds_trfm)


        diff_sa_score = self.compute_SA_score(all_gen_diff)
        trfm_sa_score = self.compute_SA_score(all_gen_trfm)
        
        
        diff_sc_score = self.compute_SC_score(all_gen_diff)
        trfm_sc_score = self.compute_SC_score(all_gen_trfm)
        


        diff_data_dict = self.arrange_data(all_gen_diff, all_tgt_diff, all_preds_diff, diff_sa_score, diff_sc_score)
        trfm_data_dict = self.arrange_data(all_gen_trfm, all_tgt_trfm, all_preds_trfm, trfm_sa_score, trfm_sc_score)
        
        
        return trfm_data_dict, diff_data_dict
    

    def compute_SA_score(self, smiles):
        all_scores = torch.zeros(len(smiles),dtype=torch.float32)
        for idx, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            all_scores[idx] = sascorer.calculateScore(mol)
        return all_scores
    

    def compute_SC_score(self, smiles):
        all_scores = torch.zeros(len(smiles), dtype=torch.float32)

        for idx,smi in enumerate(smiles):
            (smi, score) = self.scscore_model.get_score_from_smi(smi)
            all_scores[idx] = score

        return all_scores
    
    
    
    
    def arrange_data(self, smiles, targets, preds, sa_score, sc_score):

        data_dict = {}
        

        for tgt_row, pred_row, smile, sa, sc in zip(targets, preds, smiles, sa_score, sc_score):
            key = tuple(tgt_row.tolist())
            if key not in data_dict.keys():
                data_dict[key] = {}
                data_dict[key]['smiles'] = []
                data_dict[key]['preds'] = []
                data_dict[key]['sa_score'] = []
                data_dict[key]['sc_score'] = []
            
            data_dict[key]['smiles'].append(smile)
            data_dict[key]['preds'].append(pred_row.tolist())
            data_dict[key]['sa_score'].append(sa.item())
            data_dict[key]['sc_score'].append(sc.item())

        return data_dict
    
    def sort_data(self, smiles, targets, preds):
        """
        Sorts data based on pCMC values
        """
        tgt_pcmc = targets[:,0]
        pred_pcmc = preds[:,0]
        sorted_idx = torch.argsort(torch.abs(tgt_pcmc-pred_pcmc))
        smiles = np.array(smiles)
        sorted_smiles = smiles[sorted_idx]
        sorted_targets = targets[sorted_idx, :]
        sorted_preds = preds[sorted_idx, :]

        return sorted_smiles.tolist(), sorted_targets, sorted_preds




                