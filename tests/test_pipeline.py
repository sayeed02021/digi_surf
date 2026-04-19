import numpy as np
from digi_surf import SurfGen

def test_pipeline():
    model = SurfGen(device="cpu")

    constraints = np.array([[6.0]])
    out = model(pcmc=constraints)

    assert "gen_smiles" in out