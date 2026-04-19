import torch
import numpy as np
import rdkit.Chem as Chem
import torch_molecule
from tqdm import tqdm

def load_diff_model(model_path, device, task, batch_size=1):
    model = torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator(
        device=device, batch_size=batch_size,
        task_type=task,
        epochs=10000, verbose = "none"
    )

    model.load_from_local(model_path)
    return model

class DiffGen(object):
    def __init__(
            self, 
            model_path, 
            properties_type,
            device='cpu'
    ):
        """
        Class for generating molecules through diffusion
        
        model_path: path to saved model parameters

        properties_type: Shape: (prop_num) list of property strings, indicating if they are regression or categorical. Ex.: ['regression']

        device: device on which inference will be done
        """
        self.device = device
        self.model = load_diff_model(
            model_path=model_path,
            device = device,
            task = properties_type
        )
        
    
    def generate_per_property(self, property, n_gen):
        """
        property: Shape (num_prop,)
        """
        property_set = np.tile(property, (n_gen,1)) # (n_gen,num_prop)

        smi_gen = self.model.generate(labels=property_set)

        ## check which molecules are valid
        final_smi = []
        final_target_prop = []
        for smi in smi_gen:
            if smi is not None:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    final_smi.append(smi)
        
        
        return final_smi, torch.tensor(property_set[:len(final_smi),:], dtype=torch.float32) # (n_gen_actual, num_prop)


    def __call__(self, constraints, gen_per_mol):
        """
        constraints: Shape: (N,prop_num), property constraints on which the model will generate new molecules
        gen_per_mol: int, number of molecules to generate per property set
        """
        all_smi = []
        total = 0
        valid = 0
        all_target_prop = []
        for property in tqdm(constraints, dynamic_ncols=True, leave=False):
            total+=gen_per_mol
            gen_smiles, target_prop = self.generate_per_property(property, gen_per_mol)
            all_smi+=gen_smiles
            valid +=len(gen_smiles)
            all_target_prop.append(target_prop)

        # print(f'{valid}/{total} valid smiles generated. Validity={valid*100/total:0.2f}%')
        all_target_prop = torch.cat(all_target_prop, dim=0)
        return all_smi, all_target_prop
    


if __name__=="__main__":

    diff_model = DiffGen(
        model_path = "model_paths/diff_single.pt",
        properties_type=['regression'],
        device='cpu'
    )
    
        