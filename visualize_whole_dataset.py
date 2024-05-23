from lib.models.monouni import MonoUNI
import numpy as np 
import yaml
import torch
from lib.helpers.save_helper import load_checkpoint
from lib.train_val import create_logger
import os
from lib.datasets.rope3d_utils import Calibration,Denorm
import cv2
from lib.helpers.tester_helper import Tester
from torch.utils.data import DataLoader
from lib.datasets.rope3d import Rope3D

cfg = yaml.load(open('lib/config.yaml', 'r'), Loader=yaml.Loader)

logger = create_logger(os.path.join(cfg['trainer']['log_dir'],'train.log')) 

cls_mean_size = np.array([[1.288762253204939, 1.6939648801353426, 4.25589251897889],
                        [1.7199308570318539, 1.7356837654961508, 4.641152817981265],
                        [2.682263889273618, 2.3482764551684268, 6.940250839428722],
                        [2.9588510594399073, 2.5199248789610693, 10.542197736838778]])

model = MonoUNI(backbone=cfg['model']['backbone'], neck=cfg['model']['neck'], mean_size=cls_mean_size,cfg=cfg['model'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

load_checkpoint(model = model,
                        optimizer = None,
                        filename = cfg['tester']['resume_model'],
                        logger = logger,
                        map_location=device)
model.to(device)
val_set = Rope3D(root_dir=cfg['dataset']['root_dir'], split='val', cfg=cfg['dataset'])
val_loader = DataLoader(dataset=val_set,
                                batch_size=cfg['dataset']['batch_size']*4,
                                num_workers=2,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)
tester = Tester(cfg, model, val_loader, logger)
tester.test() 