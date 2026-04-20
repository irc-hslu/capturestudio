from typing import List

import numpy as np
import cv2
from tqdm import tqdm

from reconstruction.data.thuman import MultiSessionDataLoader
from reconstruction.eval.metrics import MetricsAggregator
from reconstruction.primitive.pcd import PixelPoints, RGBDImage
from reconstruction.primitive.splat import GSImage, PixelGSPoints
from reconstruction.primitive.stereo import StereoImage

# ---------------------------------------------------------------------------
CALIBRATION_SESSION_NAME = 'Thanos_2_Calib_1'
CAMERAS = [4, 5, 7, 8, 9]
IMAGE_SIZE_HW = (1024, 1024)
GPS_CHECKPOINT = 'neptune://154/best'
# FRAME_RANGE = [1200, 1500]
FRAME_RANGE = [0, 526]
# ---------------------------------------------------------------------------

print('##########################################################')
print(f"THuman 2.0 ({CALIBRATION_SESSION_NAME}) - {IMAGE_SIZE_HW[0]}x{IMAGE_SIZE_HW[1]} - {FRAME_RANGE[0]:04d}-{FRAME_RANGE[1]:04d}")
print('##########################################################')
for RECON_TYPE in ['gs']:
    for DEPTH_SOURCE in ['aligned']:
        # Load the dataset
        data_loader_ = MultiSessionDataLoader.for_eval(
            calibration_session_name=CALIBRATION_SESSION_NAME,
            calibration_method='MultiCamCalib',
            cam_indices=CAMERAS,
            n_cams_per_sample=-1,
            use_stereo=False,
            depth_filter='aligned',
            target_image_size_hw=IMAGE_SIZE_HW,
            # data-loading
            batch_size=4,
            num_workers=8,
        )

        # Evaluations
        evaluator_ = MetricsAggregator.full()  # SSIM, PSNR, LPIPS
        ## Point cloud
        pbar_ = tqdm(data_loader_, desc=f'Evaluating THuman on {RECON_TYPE.upper()}')
        for b_, batch_ in enumerate(pbar_):
            t_images_: List[RGBDImage]
            for bt_, t_images_ in enumerate(batch_):
                t_ = data_loader_.batch_size * b_ + bt_  # global time step
                if not FRAME_RANGE[0] <= t_ < FRAME_RANGE[1]:
                    continue

                for evaluator_idx_, (rgbd_l_, rgbd_middle_, rgbd_r_) in enumerate(zip(t_images_[:-2], t_images_[1:-1], t_images_[2:])):

                    if 'stereo' in DEPTH_SOURCE:
                        # (rgbd_l_, rgbd_r_) is a rectified image pair with depth computed from estimated disparity
                        rgbd_l_, rgbd_r_ = StereoImage.from_rgb_images(rgbd_l_, rgbd_r_, disparity_estimator_model=DEPTH_SOURCE, disparity_estimator_checkpoint=GPS_CHECKPOINT if DEPTH_SOURCE == 'raftstereo' else 'vitl').split_lr()

                    # out_grid_ = [
                    #     rgbd_l_.save_png(out_path=None),
                    #     rgbd_r_.save_png(out_path=None),
                    #     rgbd_middle_.save_png(out_path=None),
                    # ]

                    # unproject left and right and stitch
                    if RECON_TYPE == 'gs':
                        gs_l_, gs_r_ = GSImage.from_rgbd_image(rgbd_l_, gs_regressor_model='gps', gs_regressor_checkpoint=GPS_CHECKPOINT), GSImage.from_rgbd_image(rgbd_r_, gs_regressor_model='gps', gs_regressor_checkpoint=GPS_CHECKPOINT)
                        gs_pcd_l_ = PixelGSPoints.from_rgbd_image(gs_l_)
                        gs_pcd_r_ = PixelGSPoints.from_rgbd_image(gs_r_)
                        pcd_lr_ = PixelGSPoints.from_partials(gs_pcd_l_, gs_pcd_r_)
                    else:
                        pcd_l_ = PixelPoints.from_rgbd_image(rgbd_l_)
                        pcd_r_ = PixelPoints.from_rgbd_image(rgbd_r_)
                        pcd_lr_ = PixelPoints.from_partials(pcd_l_, pcd_r_)

                    # project to middle
                    rgbd_middle_proj_ = pcd_lr_.project(
                        target_intrinsic=rgbd_middle_.intrinsic,
                        target_extrinsic=rgbd_middle_.extrinsic_w2c,
                        target_image_size_hw=(rgbd_middle_.rgb.shape[0], rgbd_middle_.rgb.shape[1]),
                        is_c2w=False,
                        use_cache=True
                    )
                    # import numpy as np
                    # import cv2
                    # out_grid_.append(rgbd_middle_proj_.save_png(out_path=None))
                    # cv2.imwrite(f'thuman_grid_rs_{t_:04d}_{CAMERAS[evaluator_idx_]}_{CAMERAS[evaluator_idx_ + 1]}_{CAMERAS[evaluator_idx_ + 2]}.jpg', np.concatenate(out_grid_, axis=0))
                    # exit(0)

                    # evaluate the reprojection
                    evaluator_(rgbd_middle_, rgbd_middle_proj_, align=True, group=evaluator_idx_)
                    mean_metrics_t_ = evaluator_.gather()
                    pbar_.set_postfix({'t': t_, **mean_metrics_t_})

        print('')
        print('----------------------------------------------------------')
        print(f"{RECON_TYPE} / {DEPTH_SOURCE}")
        mean_metrics_ = evaluator_.gather()
        for metric_name_, metric_value_ in mean_metrics_.items():
            print(f"\t{metric_name_}: {metric_value_:.4f}")
        print(f"\tPer camera triplet metrics:")
        group_metrics_ = evaluator_.gather_grouped()
        for group_name_, group_metrics_ in group_metrics_.items():
            print(f"\t\tcams [{CAMERAS[int(group_name_)]},{CAMERAS[int(group_name_) + 1]},{CAMERAS[int(group_name_) + 2]}:")
            for metric_name_, metric_value_ in group_metrics_.items():
                print(f"\t\t\t{metric_name_}: {metric_value_:.4f}")
        print('----------------------------------------------------------')
        print('')
print('##########################################################')
