from lib.models.monouni import MonoUNI
import numpy as np 
import yaml
import torch
from lib.helpers.save_helper import load_checkpoint
from lib.train_val import create_logger
import os
from lib.datasets.rope3d_utils import Calibration,Denorm
import cv2
from help import read_kitti_ext
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections_one

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

#####input
dataset_dir = '/home/workstation/Documents/MonoUNI/Rope3D_data'
instance = '1632_fa2sd4a11North_420_1612431546_1612432197_1_obstacle'
calib = Calibration(f"{dataset_dir}/calib/{instance}.txt")
calib_p = torch.tensor(calib.P2).unsqueeze(0).to(device)
denorm = Denorm(f"{dataset_dir}/denorm/{instance}.txt")
pitch_sin=denorm.pitch_sin
calib_pitch_sin=torch.tensor(pitch_sin).unsqueeze(0).to(device)
pitch_cos=denorm.pitch_cos
calib_pitch_cos=torch.tensor(pitch_cos).unsqueeze(0).to(device)
img = cv2.imread(f"{dataset_dir}/image_2/{instance}.jpg")
resolution = [960 ,512]
img = cv2.resize(img,(resolution[0],resolution[1]))
img = img.astype(np.float32)  / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img = (img - mean) / std
img = img.transpose(2, 0, 1) 
img_tensor = torch.tensor(img).unsqueeze(0).to(device)
inputs=img_tensor
coord_ranges = [[[   0.,    0.],
         [1920., 1080.]]]
coord_ranges=torch.tensor(coord_ranges).to(device)
extrinsic_file = f"{dataset_dir}/extrinsics/{instance}.yaml"
world2camera = read_kitti_ext(extrinsic_file).reshape((4, 4))


outputs = model(inputs,coord_ranges,calib_p,K=50,mode='val', calib_pitch_sin=calib_pitch_sin, calib_pitch_cos=calib_pitch_cos)

dets = extract_dets_from_outputs(outputs,calib, K=50)
dets = dets.detach().cpu().numpy()
info ={'img_id':instance,
       'bbox_downsample_ratio':[8.    , 8.4375],
       'img_size':[1920,1080],

}
dets = decode_detections_one(dets = dets,
                        info = info,
                        calibs = calib,
                        denorms = denorm,
                        cls_mean_size=cls_mean_size,
                        threshold = cfg['tester']['threshold']) 
print(dets)