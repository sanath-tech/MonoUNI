import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.rope3d import Rope3D

# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg):
    # --------------  build rope3d dataset -----------------
    if cfg['type'] == 'rope3d':
        train_set = Rope3D(root_dir=cfg['root_dir'], split='train', cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=2,
                                  worker_init_fn=my_worker_init_fn,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=False)
        val_set = Rope3D(root_dir=cfg['root_dir'], split='val', cfg=cfg)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=2,
                                 worker_init_fn=my_worker_init_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        return train_loader, val_loader,val_loader

    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

