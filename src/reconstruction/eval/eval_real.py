from typing import List

from tqdm import tqdm

from reconstruction.data.capturestudio import MultiSessionDataLoader
from reconstruction.eval.metrics import MetricsAggregator
from reconstruction.primitive.pcd import PixelPoints, RGBDImage
from reconstruction.primitive.splat import GSImage, PixelGSPoints
from reconstruction.primitive.stereo import StereoImage

# ---------------------------------------------------------------------------
# SESSION_NAME = 'Thanos_2_Perf_1'
SESSION_NAME = 'Philipp_1_Perf_6'
# CALIBRATION_SESSION_NAME = 'Thanos_2_Calib_1'
CALIBRATION_SESSION_NAME = 'Philipp_1_Calib_1'
CAMERAS = [4, 5, 7, 8, 9]
IMAGE_SIZE_HW = (1024, 1024)
GPS_CHECKPOINT = 'neptune://154/best'
# FRAME_RANGE = [1200, 1500]
FRAME_RANGE = [400, 700]
# ---------------------------------------------------------------------------

print('##########################################################')
print(f"{SESSION_NAME} ({CALIBRATION_SESSION_NAME}) - {IMAGE_SIZE_HW[0]}x{IMAGE_SIZE_HW[1]} - {FRAME_RANGE[0]:04d}-{FRAME_RANGE[1]:04d}")
print('##########################################################')
for RECON_TYPE in ['pcd', 'gs']:
    for DEPTH_SOURCE in ['bilateral_temporal', 'raftstereo', 'foundationstereo']:
        # Load the dataset
        data_loader_ = MultiSessionDataLoader.for_eval(
            calibration_session_name=CALIBRATION_SESSION_NAME,
            calibration_method='MultiCamCalib',
            session_names=[SESSION_NAME],
            cam_indices=CAMERAS,
            n_cams_per_sample=-1,
            use_stereo=False,
            depth_filter=None if 'stereo' in DEPTH_SOURCE else DEPTH_SOURCE,
            target_image_size_hw=IMAGE_SIZE_HW,
            apply_intrinsics_fix='stereo' in DEPTH_SOURCE,
            # data-loading
            batch_size=4,
            num_workers=8,
        )

        # Evaluations
        evaluator_ = MetricsAggregator.full()  # SSIM, PSNR, LPIPS
        ## Point cloud
        pbar_ = tqdm(data_loader_, desc='Evaluating point clouds')
        for b_, batch_ in enumerate(pbar_):
            t_images_: List[RGBDImage]
            for bt_, t_images_ in enumerate(batch_):
                t_ = data_loader_.batch_size * b_ + bt_  # global time step
                if not FRAME_RANGE[0] <= t_ < FRAME_RANGE[1]:
                    continue

                for evaluator_idx_, (rgbd_l_, rgbd_middle_, rgbd_r_) in enumerate(zip(t_images_[:-2], t_images_[1:-1], t_images_[2:])):
                    # out_grid_ = [
                    #     rgbd_l_.save_png(out_path=None),
                    #     rgbd_r_.save_png(out_path=None),
                    #     rgbd_middle_.save_png(out_path=None),
                    # ]

                    if 'stereo' in DEPTH_SOURCE:
                        # (rgbd_l_, rgbd_r_) is a rectified image pair with depth computed from estimated disparity
                        rgbd_l_, rgbd_r_ = StereoImage.from_rgb_images(rgbd_l_, rgbd_r_, disparity_estimator_model=DEPTH_SOURCE, disparity_estimator_checkpoint=GPS_CHECKPOINT if DEPTH_SOURCE == 'raftstereo' else 'vitl').split_lr()

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
                    # out_grid_.append(rgbd_middle_proj_.save_png(out_path=None))
                    # cv2.imwrite(f'grid_{t_:04d}_{CAMERAS[evaluator_idx_]}_{CAMERAS[evaluator_idx_ + 1]}_{CAMERAS[evaluator_idx_ + 2]}.jpg', np.concatenate(out_grid_, axis=0))

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


"""
##########################################################
Thanos_2_Perf_1 (Thanos_2_Calib_1) - 1024x1280 - 1200-1500
##########################################################
----------------------------------------------------------
pcd / bilateral_temporal
	PSNR: 14.7931
	SSIM: 1.9591 / 3 = 0.6530
	LPIPS: 0.4063
	Per camera triplet metrics:
		cams [4,5,7:
			PSNR: 15.2596
			SSIM: 2.0128
			LPIPS: 0.4080
		cams [5,7,8:
			PSNR: 14.3448
			SSIM: 1.9292
			LPIPS: 0.3854
		cams [7,8,9:
			PSNR: 14.7748
			SSIM: 1.9353
			LPIPS: 0.4254
----------------------------------------------------------

----------------------------------------------------------
pcd / raftstereo
	PSNR: 14.4637
	SSIM: 1.8667 / 3 = 0.6222
	LPIPS: 0.2191
	Per camera triplet metrics:
		cams [4,5,7:
			PSNR: 14.7387
			SSIM: 1.8773
			LPIPS: 0.2199
		cams [5,7,8:
			PSNR: 14.3268
			SSIM: 1.8507
			LPIPS: 0.2101
		cams [7,8,9:
			PSNR: 14.3257
			SSIM: 1.8722
			LPIPS: 0.2272
----------------------------------------------------------

----------------------------------------------------------
pcd / foundationstereo
	PSNR: 12.5632
	SSIM: 1.6314 / 3 = 0.5438
	LPIPS: 0.2193
	Per camera triplet metrics:
		cams [4,5,7:
			PSNR: 12.2034
			SSIM: 1.6021
			LPIPS: 0.2204
		cams [5,7,8:
			PSNR: 13.1019
			SSIM: 1.6917
			LPIPS: 0.2100
		cams [7,8,9:
			PSNR: 12.3843
			SSIM: 1.6005
			LPIPS: 0.2275
----------------------------------------------------------

----------------------------------------------------------
gs / bilateral_temporal
	PSNR: 14.9778
	SSIM: 2.0646 / 3 = 0.6882
	LPIPS: 0.4120
	Per camera triplet metrics:
		cams [4,5,7:
			PSNR: 15.2751
			SSIM: 2.1149
			LPIPS: 0.4138
		cams [5,7,8:
			PSNR: 14.6479
			SSIM: 2.0335
			LPIPS: 0.3915
		cams [7,8,9:
			PSNR: 15.0104
			SSIM: 2.0454
			LPIPS: 0.4306
----------------------------------------------------------

----------------------------------------------------------
gs / raftstereo
	PSNR: 14.3962
	SSIM: 2.1343 / 3 = 0.7114
	LPIPS: 0.2194
	Per camera triplet metrics:
		cams [4,5,7:
			PSNR: 14.1828
			SSIM: 2.1288
			LPIPS: 0.2203
		cams [5,7,8:
			PSNR: 14.4995
			SSIM: 2.1314
			LPIPS: 0.2103
		cams [7,8,9:
			PSNR: 14.5064
			SSIM: 2.1426
			LPIPS: 0.2274
----------------------------------------------------------

----------------------------------------------------------
gs / foundationstereo
	PSNR: 11.3375
	SSIM: 1.6869 / 3 = 0.5623   
	LPIPS: 0.2210
	Per camera triplet metrics:
		cams [4,5,7:
			PSNR: 10.8892
			SSIM: 1.6398
			LPIPS: 0.2221
		cams [5,7,8:
			PSNR: 11.8578
			SSIM: 1.7553
			LPIPS: 0.2117
		cams [7,8,9:
			PSNR: 11.2653
			SSIM: 1.6654
			LPIPS: 0.2290
----------------------------------------------------------
##########################################################


##########################################################
Philipp_1_Perf_6 (Philipp_1_Calib_1) - 1024x1280 - 0400-0700
##########################################################
----------------------------------------------------------
pcd / bilateral_temporal
	PSNR: 10.0080
	SSIM: 0.4189
	LPIPS: 0.2613
	Per camera triplet metrics:
		cams [4,5,7:
			PSNR: 10.2553
			SSIM: 0.4425
			LPIPS: 0.2649
		cams [5,7,8:
			PSNR: 9.8974
			SSIM: 0.4129
			LPIPS: 0.2581
		cams [7,8,9:
			PSNR: 9.8713
			SSIM: 0.4015
			LPIPS: 0.2607
----------------------------------------------------------

----------------------------------------------------------
pcd / raftstereo
	PSNR: 9.5117
	SSIM: 0.4034
	LPIPS: 0.1581
	Per camera triplet metrics:
		cams [4,5,7:
			PSNR: 8.8237
			SSIM: 0.3683
			LPIPS: 0.1607
		cams [5,7,8:
			PSNR: 9.8944
			SSIM: 0.4170
			LPIPS: 0.1543
		cams [7,8,9:
			PSNR: 9.8170
			SSIM: 0.4248
			LPIPS: 0.1592
----------------------------------------------------------

----------------------------------------------------------
pcd / foundationstereo
	PSNR: 9.4150
	SSIM: 0.3651
	LPIPS: 0.1570
	Per camera triplet metrics:
		cams [4,5,7:
			PSNR: 9.4922
			SSIM: 0.3688
			LPIPS: 0.1575
		cams [5,7,8:
			PSNR: 9.4351
			SSIM: 0.3655
			LPIPS: 0.1541
		cams [7,8,9:
			PSNR: 9.3177
			SSIM: 0.3611
			LPIPS: 0.1593
----------------------------------------------------------

----------------------------------------------------------
gs / bilateral_temporal
	PSNR: 10.2960
	SSIM: 0.4243
	LPIPS: 0.2742
	Per camera triplet metrics:
		cams [4,5,7:
			PSNR: 10.5517
			SSIM: 0.4479
			LPIPS: 0.2774
		cams [5,7,8:
			PSNR: 10.1449
			SSIM: 0.4138
			LPIPS: 0.2727
		cams [7,8,9:
			PSNR: 10.1915
			SSIM: 0.4112
			LPIPS: 0.2725
----------------------------------------------------------

----------------------------------------------------------
gs / raftstereo
	PSNR: 9.6763
	SSIM: 0.4332
	LPIPS: 0.1617
	Per camera triplet metrics:
		cams [4,5,7:
			PSNR: 9.0202
			SSIM: 0.4024
			LPIPS: 0.1639
		cams [5,7,8:
			PSNR: 10.0171
			SSIM: 0.4458
			LPIPS: 0.1585
		cams [7,8,9:
			PSNR: 9.9915
			SSIM: 0.4515
			LPIPS: 0.1626
----------------------------------------------------------

----------------------------------------------------------
gs / foundationstereo
	PSNR: 9.7554
	SSIM: 0.3922
	LPIPS: 0.1611
	Per camera triplet metrics:
		cams [4,5,7:
			PSNR: 9.8249
			SSIM: 0.3962
			LPIPS: 0.1617
		cams [5,7,8:
			PSNR: 9.7948
			SSIM: 0.3905
			LPIPS: 0.1586
		cams [7,8,9:
			PSNR: 9.6466
			SSIM: 0.3899
			LPIPS: 0.1629
----------------------------------------------------------
##########################################################
"""