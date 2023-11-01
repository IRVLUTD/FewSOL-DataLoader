from .helpers import load_fewsol_dataloader as lfd

def load_fewsol_dataloader(dataset_root_dir, split='real_objects'):
    return lfd(dataset_root_dir, split)