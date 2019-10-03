import os
from opt import opt
args = opt
import json
import yaml
from tqdm import tqdm
import numpy as np
import cv2
from utils.utils import draw_detections_3D
from utils.metrics import projection_error_2d
from betapose_evaluate import load_sixd_models

from IPython import embed

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

BBOX_3D = np.array(
    [[  0.5, 0.5, 0.5  ],
     [  0.5,-0.5, 0.5  ],
     [ -0.5,-0.5, 0.5  ],
     [ -0.5, 0.5, 0.5  ],
     [  0.5, 0.5,-0.5  ],
     [  0.5,-0.5,-0.5  ],
     [ -0.5,-0.5,-0.5  ],
     [ -0.5, 0.5,-0.5  ]]
)

def load_yaml(path):
    with open(path, 'r') as f:
        content = yaml.full_load(f)
        return content

def load_results(path):
    with open(path) as f:
        data = json.load(f)
    
    return data

def load_image(image_path):
    return cv2.imread(image_path)

def get_pose(cam_R, cam_t):
        pose = np.identity(4)
        pose[:3, :3] = cam_R
        pose[:3, 3 ] = cam_t
        return pose

def project(pose, model, cam):
    pose = pose[:3]
    model = np.concatenate((model, np.ones((model.shape[0], 1))), axis=1)
    p = np.matmul(np.matmul(cam, pose), model.T)
    p /= p[2, :]
    p = p[:2,:].T
    return p

def draw_3d_bbox(img, pose, cam, scale = 1, color=GREEN):
    pose = pose[:3]
    box = np.concatenate((BBOX_3D / scale * 10, np.ones((BBOX_3D.shape[0], 1))), axis=1)
    p = np.matmul(np.matmul(cam, pose), box.T)
    p /= p[2, :]
    p = p[:2,:].T
    for vertex in p:
        cv2.circle(img, (int(vertex[0]), int(vertex[1])), 1, color, 2)
    return img

if __name__ == "__main__":
    obj_id = args.object_id
    id = args.image_id
    result_path = "examples/seq{}_dpg/Betapose-results.json".format(obj_id)
    image_root_path = "../LineMod/test/{:02d}/rgb".format(obj_id)
    sixd_base = "../LineMod"
    print("Loading SIXD...")
    sixd_bench = load_sixd_models(sixd_base, obj_id)
    cam_K = sixd_bench.cam
    models = sixd_bench.models
    model = models['{:02d}'.format(obj_id)].vertices
    diameter = models['{:02d}'.format(obj_id)].diameter
    print("Loading gt...")
    gt_data = load_yaml("../LineMod/test/{:02d}/gt.yml".format(obj_id))
    print("Loading results...")
    result = load_results(result_path)
    # for f in tqdm(result, ncols=80, ascii=True):
    for f in [result[id]]:
        imgname = f['image_id']
        img = load_image(os.path.join(image_root_path, imgname))
        bg_gt = img.copy()
        bg_est = img.copy()
        # Ground truth
        gt_dets = gt_data[id]
        for gt in gt_dets:
            if gt['obj_id'] == obj_id:
                gt_cam_R = np.reshape(gt['cam_R_m2c'], [3, 3])
                gt_cam_t = [x / 1000 for x in gt['cam_t_m2c']]
                gt_pose = get_pose(gt_cam_R, gt_cam_t)

                gt_proj_2d = project(gt_pose, model, cam_K)

                for vertex in gt_proj_2d:
                    cv2.circle(bg_gt, (int(vertex[0]), int(vertex[1])), 1, GREEN)
                draw_3d_bbox(bg_gt, gt_pose, cam_K, scale = diameter, color = GREEN)

        # Prediction        
        cam_R = np.reshape(f['cam_R'], [3, 3])
        cam_t = f['cam_t']
        #keypoints = f['keypoints']
        #score = f['score']
        est_pose = get_pose(cam_R, cam_t)

        proj_2d = project(est_pose, model, cam_K)

        for vertex in proj_2d:
            cv2.circle(bg_est, (int(vertex[0]), int(vertex[1])), 1, RED)
        draw_3d_bbox(bg_est, est_pose, cam_K, scale = diameter, color=RED)

        bg = cv2.addWeighted(bg_est, 0.5, bg_gt, 0.5, 0)
        img = cv2.addWeighted(img, 0.2, bg, 0.8, 0)
        cv2.imwrite('est.png', img)