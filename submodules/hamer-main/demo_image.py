from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import math
from tqdm import tqdm
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from decord import VideoReader
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

# from vitpose_model import ViTPoseModel
from DWPose.ControlNet.annotator.dwpose import DWposeDetector

import json
from typing import Dict, Optional

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    # from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    # from detectron2.config import LazyConfig
    # import hamer
    # cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    # detectron2_cfg = LazyConfig.load(str(cfg_path))
    # detectron2_cfg.train.init_checkpoint = "/mnt/workspace/workgroup/wangbenzhi.wbz/RealisHuman/submodules/hamer-main/model_final_f05665.pkl"
    # for i in range(3):
    #     detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    # detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    # cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)
    input_path = args.img_folder
    if not os.path.exists(input_path):
        print('path not exists: {}'.format(input_path))
        return
    pose = DWposeDetector()
    for i, image_name in tqdm(enumerate(os.listdir(input_path))):
        image_path = os.path.join(input_path, image_name)
        img_cv2 = cv2.imread(image_path)
        vitposes_out = pose(img_cv2)
        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes[-42:-21]
            right_hand_keyp = vitposes[-21:]
            # Rejecting not confident detections

            keyp = left_hand_keyp
            if (keyp!=-1).sum()>6: 
                valid_indices = np.where(keyp != -1)
                valid_keyp_x = keyp[valid_indices[0], 0]
                valid_keyp_y = keyp[valid_indices[0], 1]
                bbox = [valid_keyp_x.min(), valid_keyp_y.min(), valid_keyp_x.max(), valid_keyp_y.max()]
                bboxes.append(bbox)
                is_right.append(0)
        
            keyp = right_hand_keyp
            if (keyp!=-1).sum()>6: 
                valid_indices = np.where(keyp != -1)
                valid_keyp_x = keyp[valid_indices[0], 0]
                valid_keyp_y = keyp[valid_indices[0], 1]
                bbox = [valid_keyp_x.min(), valid_keyp_y.min(), valid_keyp_x.max(), valid_keyp_y.max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                # img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(0, 0, 0),
                                        )

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(0, 0, 0),
                                            side_view=True)
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    hand_mask_img = regression_img
                    hand_img = input_patch

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

        #         # Save all meshes to disk
        #         if args.save_mesh:
        #             camera_translation = cam_t.copy()
        #             tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
        #             tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view_diff = renderer.render_rgba_multiple_diff(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            fullview_out_image_diff = os.path.join(args.out_folder, image_name)
            cv2.imwrite(fullview_out_image_diff, 255*cam_view_diff[:, :, ::-1])

if __name__ == '__main__':
    main()
