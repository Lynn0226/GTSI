"""
    File to load dataset based on user control from main file
"""
from data.PROP import PROPDataset

def LoadData(DATASET_NAME):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """    
    PROP_DATASETS = ['PROP_KARATE_inf0.2', 'PROP_JAZZ_inf0.2', 'PROP_FACEBOOK_inf0.2', 'PROP_KARATE_inf0.2_hand', 'PROP_JAZZ_inf0.2_hand','PROP_FACEBOOK_inf0.2_hand',
                     'PROP_KARATE_dyninf_hand', 'PROP_JAZZ_dyninf_hand', 'PROP_DOLPHIN_dyninf_hand', 'PROP_FACEBOOK_dyninf_hand', 'PROP_WIKIVOTE_dyninf_hand','PROP_PAGELARGE_dyninf_hand', 'PROP_TWITCHES_dyninf_hand',
                     'PROP_TWITCHES_KARATE_hand', 'PROP_TWITCHES_JAZZ_hand', 'PROP_TWITCHES_FACEBOOK_hand', 'PROP_TWITCHES_WIKIVOTE_hand', 'PROP_TWITCHES_PAGELARGE_hand',
                     'PROP_TWITCHES_SIHOMO_hand', 'PROP_TWITCHES_SIRHOMO_hand', 'PROP_TWITCHES_SIHETE_hand', 'PROP_TWITCHES_SIRHETE_hand'
                     ]
    if DATASET_NAME in PROP_DATASETS:
        return PROPDataset(DATASET_NAME)


    # if DATASET_NAME == 'karate':
    #     return KarateDatasete(DATASET_NAME)
    