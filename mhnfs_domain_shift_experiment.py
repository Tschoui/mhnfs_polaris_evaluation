"""
This evaluation script tests the performance of MHNfs (trained on the FS-Mol
benchmark) for polaris downstream tasks.
Notably, no re-training or fine-tuning is done.
The support set is built from the training set.
"""

#---------------------------------------------------------------------------------------
# User choices
MHNFS_PATH = "/system/user/publicwork/schimunek/mhnfs_huggingface/"
POLARIS_BENCHMARK = "polaris/pkis2-ret-wt-cls-v2"
SEED = 1019
#---------------------------------------------------------------------------------------
# Dependencies
import sys
sys.path.append(MHNFS_PATH)
from src.prediction_pipeline import ActivityPredictor

import polaris as po
import random
import copy
#---------------------------------------------------------------------------------------
# Functions needed

def create_support_set(train_set, nbr_actives=8, nbr_inactives=8):
    molecules = train_set.X
    labels = train_set.y
    
    active_molecules = molecules[labels == 1]
    inactive_molecules = molecules[labels == 0]
    
    assert active_molecules.shape[0] >= 8
    assert inactive_molecules.shape[0] >= 8
    
    active_idx = list(range(active_molecules.shape[0]))
    random.shuffle(active_idx)
    #active_idx = active_idx[:nbr_actives]
    
    inactive_idx = list(range(inactive_molecules.shape[0]))
    random.shuffle(inactive_idx)
    #inactive_idx = inactive_idx[:nbr_inactives]
    
    support_actives_smiles = list(active_molecules[active_idx])
    support_inactives_smiles = list(inactive_molecules[inactive_idx])
    
    return support_actives_smiles, support_inactives_smiles

def mhnfs_polaris_domain_shift_experiment(benchmark=POLARIS_BENCHMARK, seed=SEED):
    random.seed(seed)
    
    # Load MHNfs
    print('Load model ...')
    predictor = ActivityPredictor()
    
    # Load polaris task
    print('Load polaris task and transform it into a few-shot setting ...')
    benchmark = po.load_benchmark(benchmark)
    train, test = benchmark.get_train_test_split()
    
    support_actives_smiles, support_inactives_smiles = create_support_set(train)
    query_smiles = list(test.X)
    
    
    
    # Make predictions
    print('Use MHNfs to predict activities ...')
    predictions = predictor.predict(query_smiles,
                                    support_actives_smiles,
                                    support_inactives_smiles)
    binary_preds = copy.deepcopy(predictions)
    binary_preds[binary_preds > 0.5] = 1.
    binary_preds[binary_preds <= 0.5] = 0.
    
    # Evaluate predictions
    print('Evaluate results ...')
    results = benchmark.evaluate(binary_preds, predictions)
    
    # Fill meta-data and submit results
    print('Submit to leaderboard ...')
    results.name = "MHNfs"
    results.description = "MHNfs is a few-shot method trained on FS-Mol. The trained model is applied to the benchmark by using the training data as a support set."

    results.upload_to_hub(owner="tschoui")

def run_experiment(benchmarks)
#---------------------------------------------------------------------------------------
# Execute script
if __name__ == "__main__":
    mhnfs_polaris_domain_shift_experiment()
