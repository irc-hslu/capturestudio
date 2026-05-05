from typing import Union, Literal, Optional, List, Tuple, Mapping, Any
import easydict
import numpy
import torch
from pytorch3d.transforms import (
    matrix_to_quaternion,  # p3d quat format: wxyz
    quaternion_multiply,
    quaternion_invert
)
from torch import nn
from torch.ao.nn.quantized import InstanceNorm2d

from reconstruction.arch.gpsgaussian.extractor import UnetExtractor
from reconstruction.arch.gpsgaussian.gs_parm_network import GSRegresser
from reconstruction.arch.gpsgaussian.raft_stereo_human import RAFTStereoHuman
from reconstruction.primitive.pcd import PCDUtils
from reconstruction.primitive.splat import GSUtils
from utils.misc import PathUtils, log


class GPSRaftStereo(nn.Module):
    def __init__(self, frozen: bool = False, mixed_precision: bool = False):
        super().__init__()
        raft_cfg = easydict.EasyDict(
            train_iters=3,
            val_iters=3,
            encoder_dims=[32, 48, 96],
            hidden_dims=[96, 96, 96],
            corr_implementation='reg_cuda',  # or 'reg'
            corr_levels=4,
            corr_radius=4,
            n_downsample=3,
            n_gru_layers=1,
            slow_fast_gru=None,
            mixed_precision=mixed_precision,
        )
        self.raft_cfg = raft_cfg
        # noinspection PyTypeChecker
        self.raft = RAFTStereoHuman(raft_cfg)
        self.frozen = frozen
        if self.frozen:
            for param in self.raft.parameters():
                param.requires_grad = False

    def forward(self, image_feat_lr: torch.Tensor):
        if self.frozen:
            with torch.no_grad():
                # noinspection PyUnresolvedReferences
                flow_pred_lr = self.raft.forward(image_feat_lr, iters=self.raft_cfg.val_iters, test_mode=True)
            return flow_pred_lr
        # noinspection PyUnresolvedReferences
        flow_pred_left, flow_pred_right = self.raft(image_feat_lr, iters=self.raft_cfg.train_iters, test_mode=False)
        return flow_pred_left, flow_pred_right


class GPSFoundationStereo(nn.Module):
    MODEL_PATHS = dict(
        checkpoint=PathUtils.checkpoints_path() / 'foundation_stereo' / 'bp2_{which_vit}.pth',
        config=PathUtils.checkpoints_path() / 'foundation_stereo' / 'bp2_{which_vit}.yaml'
    )

    def __init__(self, which_vit: Union[Literal['vits'], Literal['vitl']] = 'vits', hierarchical: bool = False):
        super().__init__()
        from foundationstereo.core.foundation_stereo import FoundationStereo
        from omegaconf import OmegaConf

        # Load config
        cli_args = dict(
            scale=0.375,  # downsize the image by scale, must be <=1
            hiera=int(hierarchical),  # hierarchical inference (only needed for high-resolution images (>1K))
            z_far=10.0,  # max depth to clip in point cloud
            valid_iters=32,  # number of flow-field updates during forward pass
            get_pc=1,  # save point cloud output
            remove_invisible=1,  # remove non-overlapping observations between left and right images from point cloud
            denoise_cloud=1,  # whether to denoise the point cloud
            denoise_nb_points=30,  # number of points to consider for radius outlier removal
            denoise_radius=0.03  # radius to use for outlier removal
        )
        cfg = OmegaConf.load(str(self.__class__.MODEL_PATHS['config']).format(which_vit=which_vit))
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        for k in cli_args:
            cfg[k] = cli_args[k]
        args = OmegaConf.create(cfg)
        # Load the model
        self.model = FoundationStereo(args)
        torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
        torch.serialization.add_safe_globals([numpy.dtype])
        ckpt_path = str(self.__class__.MODEL_PATHS['checkpoint']).format(which_vit=which_vit)
        ckpt = torch.load(ckpt_path, weights_only=False)
        self.model.load_state_dict(ckpt['model'], strict=True)
        log(f'[{self.__class__.__name__}::__init__] Loaded FoundationStereo model from {ckpt_path}', 'debug')

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def run_hierarchical(self, *args, **kwargs):
        """
        Run hierarchical inference on the model (for high-resolution images).
        This method is a wrapper around the model's `run_hierachical` method (note the typo in the original method name).
        """
        return self.model.run_hierachical(*args, **kwargs)


class GPSGaussianRegressor(nn.Module):
    def __init__(self,
                 rot_head_frozen: bool = True,
                 opacity_head_frozen: bool = True,
                 scale_head_frozen: bool = False,
                 do_render: bool = True,
                 scale_head_tanh: bool = False,
                 scale_head_norm: bool = False,
                 rot_bug_fix_version: Literal['none', 'correct', 'simone', 'thanos'] = 'none'):  # 'none': no bug fix, 'correct': correct, 'simone': Simone's version, 'thanos': Thanos' version
        assert rot_bug_fix_version in ['none', 'correct', 'simone', 'thanos']
        super().__init__()
        cfg = easydict.EasyDict(dict(
            raft=dict(
                encoder_dims=[32, 48, 96],
                hidden_dims=[96, 96, 96]
            ),
            gsnet=dict(
                encoder_dims=[32, 48, 96],
                decoder_dims=[48, 64, 96],
                parm_head_dim=32
            )
        ))
        self.gs_parm_regresser = GSRegresser(cfg, rgb_dim=3, depth_dim=1, scale_tanh=scale_head_tanh, scale_norm=scale_head_norm)
        self.rot_frozen = rot_head_frozen
        if self.rot_frozen:
            for param in self.gs_parm_regresser.rot_head.parameters():
                param.requires_grad = False
        self.opacity_frozen = opacity_head_frozen
        if self.opacity_frozen:
            for param in self.gs_parm_regresser.opacity_head.parameters():
                param.requires_grad = False
        self.scale_frozen = scale_head_frozen
        if self.scale_frozen:
            for param in self.gs_parm_regresser.scale_head.parameters():
                param.requires_grad = False
        self.do_render = do_render

        self.scale_alpha = nn.Parameter(torch.tensor([0.0025]), requires_grad=True)
        self.scale_beta = nn.Parameter(torch.tensor([0.0026]), requires_grad=True)
        self.rot_bug_fix_version = rot_bug_fix_version

    @staticmethod
    def render(xyz: torch.Tensor, colors: torch.Tensor, gs_rot: torch.Tensor, gs_scale: torch.Tensor, gs_opacity: torch.Tensor, rasterizer=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # render
        screenspace_points = torch.zeros_like(xyz, requires_grad=True)
        screenspace_points.retain_grad()
        # rendered_image, rendered_depth, _, rendered_alpha, radii, _ = rasterizer(
        rendered_image, rendered_radii, rendered_invdepth = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            colors_precomp=colors,
            opacities=gs_opacity,
            scales=gs_scale,
            rotations=gs_rot,
        )
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        # rendered_depth = 1 / (rendered_invdepth + 1e-6)
        rendered_depth = rendered_invdepth
        return rendered_image, rendered_depth

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        super().load_state_dict(state_dict['network'] if 'network' in state_dict else state_dict, **kwargs)

    def forward(self,
                image: torch.Tensor,  # Bx3xHxW
                image_feat: List[torch.Tensor],  # list of BxC_feat
                depth: torch.Tensor,  # Bx1xHxW
                intrinsic: torch.Tensor,  # Bx3x3
                extrinsic: torch.Tensor,  # Bx4x4
                camera_idx: torch.Tensor,  # B
                stitch_n_cams: Optional[int] = None,
                rasterizers: Optional[list] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # get point cloud
        with torch.no_grad():
            xyz, xyz_color = PCDUtils.unproject_depth_maps_torch(
                depth=depth,
                color=image,
                intrinsics=intrinsic,
                rotmats=extrinsic[..., :3, :3],
                tvecs=extrinsic[..., :3, 3]
            )
            # from process.reconstruction.pcd import PointClouds
            # PointClouds.from_rgbd(depth, intrinsics=intrinsic, rotmats=extrinsic[..., :3, :3], tvecs=extrinsic[..., :3, 3], images=image).stitch().save_ply('out__forward__.ply')
            pts_valid = (depth > 0.5).logical_and_(depth < 5.0).reshape_as(xyz[..., [-1]]).logical_and_(~torch.isnan(xyz[..., [-1]])).logical_and_(~torch.isinf(xyz[..., [-1]]))
            xyz[~pts_valid.expand_as(xyz)] = 0.0
        # image_left_feat, image_right_feat = torch.split(image_feat, [image_left.shape[0], image_right.shape[0]], dim=0)
        inv_depth_lr = 1.0 / (depth + 1e-6)
        gs_rot, gs_scale, gs_opacity = self.gs_parm_regresser(image, inv_depth_lr, image_feat, self.scale_alpha, self.scale_beta)
        if self.rot_frozen:
            gs_rot = gs_rot.detach()
        if self.opacity_frozen:
            gs_opacity = gs_opacity.detach()
        if self.scale_frozen:
            gs_scale = gs_scale.detach()

        # FIX: Rotations from cam -> world
        if self.rot_bug_fix_version == 'none':
            pass
        elif self.rot_bug_fix_version == 'correct':
            ## Correct: R <- r_c2w (pose matrix of camera) * r_g2c (pose matrix of gaussian) [NO INVERSION]
            r_c2w = matrix_to_quaternion(extrinsic[..., :3, :3])[..., None, None, :]  # (..., 1, 1, 4)
            r_g2c = gs_rot.permute(0, 2, 3, 1)  # (B, h, w, 4)
            gs_rot = quaternion_multiply(r_c2w, r_g2c).permute(0, 3, 1, 2)  # (B, 4, h, w)
        elif self.rot_bug_fix_version == 'simone':
            ## Simone: R <- r_w2c (viewmatrix of camera) * r_g2c (pose matrix of gaussian)
            r_c2w = matrix_to_quaternion(extrinsic[..., :3, :3])[..., None, None, :]  # (..., 1, 1, 4)
            r_w2c = quaternion_invert(r_c2w)  # (B, 1, 1, 4)
            r_g2c = gs_rot.permute(0, 2, 3, 1)  # (B, h, w, 4)
            gs_rot = quaternion_multiply(r_w2c, r_g2c).permute(0, 3, 1, 2)  # (B, 4, h, w)
        elif self.rot_bug_fix_version == 'thanos':
            ## Thanos: R <- r_c2w (pose matrix of camera) * r_c2g (viewmatrix of gaussian)
            r_c2w = matrix_to_quaternion(extrinsic[..., :3, :3])[..., None, None, :]  # (..., 1, 1, 4)
            r_g2c = gs_rot.permute(0, 2, 3, 1)  # (B, h, w, 4)
            r_g2w = quaternion_multiply(r_c2w, quaternion_invert(r_g2c))  # (B, h, w, 4) # ATTN: this is still wrong but it was how the model was trained on
            gs_rot = r_g2w.permute(0, 3, 1, 2)  # (B, 4, h, w)
        else:
            raise ValueError(f"Unknown rot_bug_fix_version: {self.rot_bug_fix_version}. Must be one of ['none', 'correct', 'simone', 'thanos'].")

        # render if needed
        if self.do_render:
            assert rasterizers is not None, "Rasterizers must be provided when rendering is enabled."

            if stitch_n_cams is None:
                stitch_n_cams = 1
            # When stitching is enabled, it is assumed that the cam data to be stitched, have been concatenated along the batch dimension
            #   i.e. if we have B left and B right cams that need to be stitched the input batch size should have been 2*B, in the form LL...LRR...R
            # The batch is unpacked and the corresponding cam data are stitched BEFORE inputting to the rasterizer.
            b_stitch_interleave = image.shape[0] // stitch_n_cams

            rendered_images, rendered_depths = [], []
            gs_rots, gs_scales, gs_opacities = [], [], []
            for b_outer in range(b_stitch_interleave):
                xyz_stitched, xyz_color_stitched, gs_rot_stitched, gs_scale_stitched, gs_opacity_stitched = [], [], [], [], []
                for b_inner in range(stitch_n_cams):
                    b = b_outer + b_inner * b_stitch_interleave
                    # get all the parameters for all cameras
                    xyz_c, xyz_color_c = xyz[b], xyz_color[b]
                    gs_rot_c, gs_scale_c, gs_opacity_c = gs_rot[b].view(4, -1).transpose(-1, -2), gs_scale[b].view(3, -1).transpose(-1, -2), gs_opacity[b].view(-1, 1)
                    valid_c = pts_valid[b]
                    valid_c3, valid_c4 = valid_c.expand_as(xyz_c), valid_c.expand_as(gs_rot_c)
                    # remove invalid points
                    xyz_c_valid = xyz_c[valid_c3].reshape(-1, 3)
                    xyz_color_c_valid = xyz_color_c[valid_c3].reshape(-1, 3)
                    gs_rot_c_valid = gs_rot_c[valid_c4].reshape(-1, 4)
                    gs_scale_c_valid = gs_scale_c[valid_c3].reshape(-1, 3)
                    gs_opacity_c_valid = gs_opacity_c[valid_c].reshape(-1, 1)
                    # append to lists
                    xyz_stitched.append(xyz_c_valid)
                    xyz_color_stitched.append(xyz_color_c_valid)
                    gs_rot_stitched.append(gs_rot_c_valid)
                    gs_scale_stitched.append(gs_scale_c_valid)
                    gs_opacity_stitched.append(gs_opacity_c_valid)
                # concatenate
                xyz_stitched = torch.cat(xyz_stitched, dim=0)
                xyz_color_stitched = torch.cat(xyz_color_stitched, dim=0)
                gs_rot_stitched = torch.cat(gs_rot_stitched, dim=0)
                gs_scale_stitched = torch.cat(gs_scale_stitched, dim=0)
                gs_opacity_stitched = torch.cat(gs_opacity_stitched, dim=0)
                # append
                gs_rots.append(gs_rot_stitched)
                gs_scales.append(gs_scale_stitched)
                gs_opacities.append(gs_opacity_stitched)
                # rasterize the point cloud in both cameras
                for b_inner in range(stitch_n_cams):
                    b = b_outer + b_inner * b_stitch_interleave
                    rendered_image, rendered_depth = self.__class__.render(
                        xyz=xyz_stitched,
                        colors=xyz_color_stitched,
                        gs_rot=gs_rot_stitched,
                        gs_scale=gs_scale_stitched,
                        gs_opacity=gs_opacity_stitched,
                        rasterizer=rasterizers[camera_idx[b].item()]
                    )
                    rendered_images.append(rendered_image)
                    rendered_depths.append(rendered_depth)

            rendered_image = torch.stack([elem for i in range(stitch_n_cams) for elem in rendered_images[i::stitch_n_cams]], dim=0)
            rendered_depth = torch.stack([elem for i in range(stitch_n_cams) for elem in rendered_depths[i::stitch_n_cams]], dim=0)

            gs_rots = torch.cat(gs_rots, dim=0)
            gs_scales = torch.cat(gs_scales, dim=0)
            gs_opacities = torch.cat(gs_opacities, dim=0)

            # import matplotlib.pyplot as plt
            # plt.imshow(torch.cat(rendered_image.unbind(0), dim=-1).mul(255).add(0.5).byte().cpu().permute(1, 2, 0))
            # plt.show()

            return rendered_image, rendered_depth, gs_rots, gs_scales, gs_opacities

        # if not rendering, just return the parameters as a tuple (left params, right params)
        return xyz, xyz_color, gs_rot, gs_scale, gs_opacity


class GPS(nn.Module):
    cameras: List[easydict.EasyDict] = []
    rasterizers: list = []

    def __init__(self,
                 finetune_feat_enc: bool = False,
                 use_raft_stereo: bool = False,
                 use_foundation_stereo: bool = False,
                 finetune_raft_stereo: bool = False,
                 finetune_gaussian_rotations: bool = False,
                 finetune_gaussian_opacities: bool = False,
                 finetune_gaussian_scales: bool = True,
                 gps_chkpt_path: Optional[str] = None,
                 scale_head_tanh: bool = False,
                 scale_head_norm: bool = False,
                 rot_bug_fix_version: Literal['none', 'correct', 'simone', 'thanos'] = 'none',
                 do_render: bool = True):
        super().__init__()
        self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=[32, 48, 96])
        self.feat_enc_frozen = not finetune_feat_enc
        if self.feat_enc_frozen:
            for param in self.img_encoder.parameters():
                param.requires_grad = False
        self.use_raft_stereo = use_raft_stereo
        self.mixed_precision = not (finetune_raft_stereo or finetune_feat_enc)
        assert (use_raft_stereo ^ use_foundation_stereo) or (not use_raft_stereo and not use_foundation_stereo), "Cannot have both use_raft_stereo and use_foundation_stereo"
        if use_raft_stereo:
            self.raft_stereo_frozen = not finetune_raft_stereo
            self.raft_stereo = GPSRaftStereo(
                frozen=self.raft_stereo_frozen,
                mixed_precision=self.mixed_precision
            )
            self.use_foundation_stereo = False
        elif use_foundation_stereo:
            assert not finetune_raft_stereo, 'When using FoundationStereo finetuning is prohibitively expensive :('
            self.raft_stereo_frozen = True
            self.use_foundation_stereo = True
            self.raft_stereo = GPSFoundationStereo(
                which_vit='vits'
            )
        else:
            self.raft_stereo = None
            self.raft_stereo_frozen = True  # No RAFT stereo, so it is considered frozen
            self.use_foundation_stereo = False
        self.gs_regressor_frozen = not (finetune_gaussian_rotations or finetune_gaussian_opacities or finetune_gaussian_scales)
        self.gs_regressor = GPSGaussianRegressor(
            rot_head_frozen=not finetune_gaussian_rotations,
            opacity_head_frozen=not finetune_gaussian_opacities,
            scale_head_frozen=not finetune_gaussian_scales,
            do_render=do_render,
            scale_head_tanh=scale_head_tanh,
            scale_head_norm=scale_head_norm,
            rot_bug_fix_version=rot_bug_fix_version,
        )
        # placeholders
        self.image_size_hw = (0, 0)

        # load pretrained weights if provided
        assert gps_chkpt_path is None, "Checkpoint loading has been moved to load_state_dict method. Please use that instead."
        # if gps_chkpt_path is not None:
        #     if not Path(gps_chkpt_path).exists():
        #         gps_chkpt_path = str(PathUtils.checkpoints_path() / str(gps_chkpt_path).split('TORCH_HOME/checkpoints')[-1].lstrip('/'))
        #     self.load_state_dict(torch.load(gps_chkpt_path, map_location=torch.device('cpu')))
        #     log(f"[{self.__class__.__name__}::__init__] Loaded GPS model from {gps_chkpt_path}", 'info')

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        if 'network' in state_dict:
            state_dict = state_dict['network']
        state_dict_fixed = {}
        for k, v in state_dict.items():
            if k.startswith('gs_parm_regresser.img_encoder.'):
                k = k[len('gs_parm_regresser.'):]
            elif k.startswith('raft_stereo.') and not k.startswith('raft_stereo.raft.'):
                if not self.use_raft_stereo or self.use_foundation_stereo:
                    continue
                k = k.replace('raft_stereo.', 'raft_stereo.raft.')
            elif k.startswith('gs_parm_regresser.'):
                k = k.replace('gs_parm_regresser.', 'gs_regressor.gs_parm_regresser.')
            state_dict_fixed[k] = v
            # new changes related to the scale head
            if hasattr(self.gs_regressor, 'scale_alpha') and 'gs_regressor.scale_alpha' not in state_dict_fixed:
                state_dict_fixed['gs_regressor.scale_alpha'] = torch.tensor([1.0], dtype=torch.float32)
            if hasattr(self.gs_regressor, 'scale_beta') and 'gs_regressor.scale_beta' not in state_dict_fixed:
                state_dict_fixed['gs_regressor.scale_beta'] = torch.tensor([0.0], dtype=torch.float32)
            if isinstance(list(self.gs_regressor.gs_parm_regresser.scale_head.modules())[-1], InstanceNorm2d) and 'gs_regressor.gs_parm_regresser.scale_head.4.weight' not in state_dict_fixed:
                state_dict_fixed['gs_regressor.gs_parm_regresser.scale_head.4.weight'] = torch.tensor([0.0025, 0.0025, 0.0025])
                state_dict_fixed['gs_regressor.gs_parm_regresser.scale_head.4.bias'] = torch.tensor([0.005, 0.005, 0.005])
        missing, unused = super().load_state_dict(state_dict_fixed, strict=False)
        assert len(missing) == 0, missing

    def set_camera_parameters(self, intrinsics: torch.Tensor, extrinsics: torch.Tensor, image_size_hw: Tuple[int, int] = (2160, 3840)):
        from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

        # create cameras
        self_device = 'cuda:0'
        self.cameras = GSUtils.create_cameras(
            intrinsics=intrinsics.to(self_device, dtype=torch.float32),
            rotmats=extrinsics[..., :3, :3].to(self_device, dtype=torch.float32),
            tvecs=extrinsics[..., :3, 3].to(self_device, dtype=torch.float32),
            image_size=image_size_hw,
        )
        self.image_size_hw = image_size_hw
        # noinspection PyUnresolvedReferences
        self.rasterizers = [
            GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
                image_height=camera.H,
                image_width=camera.W,
                tanfovx=math.tan(camera.fov_x * 0.5),
                tanfovy=math.tan(camera.fov_y * 0.5),
                bg=torch.tensor((0.0, 0.0, 0.0), device=self_device, dtype=torch.float32),
                scale_modifier=1.0,
                viewmatrix=camera.world_view_transform,
                projmatrix=camera.full_proj_transform,
                sh_degree=0,
                campos=camera.camera_center,
                prefiltered=False,
                debug=False
            ))
            for camera in self.cameras
        ]

    @staticmethod
    def flow2depth(flow: torch.Tensor, mask: torch.Tensor, intrinsics: torch.Tensor, intrinsics_ref: torch.Tensor, tf_x: torch.Tensor) -> torch.Tensor:
        offset = intrinsics[..., 0, 2] - intrinsics_ref[..., 0, 2]
        offset = torch.broadcast_to(offset[:, None, None, None], flow.shape)
        disparity = offset - flow
        inv_depth = -disparity / tf_x[:, None, None, None]
        inv_depth *= mask[:, :1, :, :]
        depth = 1.0 / (inv_depth + 1e-6)
        return depth

    @staticmethod
    def batch_lr_to_llrr(lr: torch.Tensor) -> torch.Tensor:
        return lr.swapaxes(0, 1).flatten(0, 1)

    @staticmethod
    def llrr_to_batch_lr(lr: torch.Tensor, n_cams_per_batch_sample: int) -> torch.Tensor:
        return lr.unflatten(0, (n_cams_per_batch_sample, -1)).swapaxes(0, 1)

    def forward(self,
                image: torch.Tensor,  # Bx2x3xHxW
                mask: torch.Tensor,  # Bx2x1xHxW
                intrinsic: torch.Tensor,  # Bx2x3x3
                extrinsic: torch.Tensor,  # Bx2x4x4
                camera_idx: torch.Tensor,  # Bx2
                # frame_idx: torch.Tensor,  # Bx2
                tf_x: Optional[torch.Tensor] = None,  # Bx2
                baseline: Optional[torch.Tensor] = None,  # B
                depth: Optional[torch.Tensor] = None,  # Bx2x1xHxW, only needed if `use_raft_stereo` is False
                stitch_partials: bool = False  # if True, it stitches the partials prior to rasterization
                ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # flatten the batch dimension for left and right images
        b, n_cams_per_sample, c, *self.image_size_hw = image.shape  # H, W
        image = self.batch_lr_to_llrr(image)  # (Bx2)x3xHxW -> (2B)x3xHxW
        mask = self.batch_lr_to_llrr(mask)  # (Bx2)x1xHxW -> (2B)x1xHxW
        intrinsic = self.batch_lr_to_llrr(intrinsic)  # (Bx2)x3x3 -> (2B)x3x3
        extrinsic = self.batch_lr_to_llrr(extrinsic)  # (Bx2)x4x4 -> (2B)x4x4
        camera_idx = self.batch_lr_to_llrr(camera_idx)  # (Bx2) -> (2B)
        # frame_idx = self.batch_lr_to_llrr(frame_idx)  # (Bx2) -> (2B)
        if depth is not None:
            depth = self.batch_lr_to_llrr(depth)  # (Bx2)x1xHxW -> (2B)x1xHxW

        # get image features
        if self.feat_enc_frozen:
            with torch.no_grad(), torch.autocast(enabled=self.mixed_precision, device_type='cuda'):
                img_feat_lr = self.img_encoder(image)
        else:
            with torch.autocast(enabled=self.mixed_precision, device_type='cuda'):
                img_feat_lr = self.img_encoder(image)

        # compute depth
        # depth_gt = depth.clone()
        if not self.use_raft_stereo and not self.use_foundation_stereo:
            assert depth is not None, "Depth must be provided when RAFT stereo is not used."
        else:
            assert False, 'Please compute stereo data using StereoUtils / StereoImage classes and pass the stereo depth as input to the GPS model.'
            # assert n_cams_per_sample == 2, "RAFT stereo and FoundationStereo only support stereo pairs (2 cameras per sample)."
            # intrinsic_ref = torch.cat([
            #     intrinsic[intrinsic.shape[0] // 2:],
            #     intrinsic[:intrinsic.shape[0] // 2]
            # ])
            # cx_disparity = intrinsic[..., 0, 2] - intrinsic_ref[..., 0, 2]
            # if self.use_foundation_stereo:
            #     # print('in_fs')
            #     img_l, img_r = torch.split(torch.nn.functional.interpolate(image, scale_factor=0.25, align_corners=True, mode='bilinear'), image.shape[0] // 2, dim=0)
            #     with torch.cuda.amp.autocast(True):
            #         disparity_l = self.raft_stereo(img_l, img_r, iters=16, test_mode=True)
            #         disparity_r = self.raft_stereo(img_r, img_l, iters=16, test_mode=True)
            #     disparity_lr = torch.nn.functional.interpolate(torch.cat([disparity_l, disparity_r], dim=0), scale_factor=4.0, align_corners=True, mode='bilinear') * 4.0  # - cx_disparity[:, None, None, None]
            #     # TODO: check me
            #     # disparity_lr -= cx_disparity.flatten()[:, None, None, None]
            #     disparity_lr[mask < 0.3] = torch.inf
            #     depth = (intrinsic[..., [0], [0]] * baseline)[..., None, None] / disparity_lr
            # else:
            #     # print('in_rs')
            #     # Use RAFT stereo to compute depth
            #     disparity_lr = self.raft_stereo.forward(img_feat_lr[-1])
            #     if isinstance(disparity_lr, (tuple, list)):
            #         disparity_lr = torch.cat(disparity_lr, dim=0)  # Concatenate left and right flows
            #     disparity_lr[mask < 0.3] = torch.inf
            #     depth = tf_x.flatten()[:, None, None, None] / (disparity_lr - cx_disparity.flatten()[:, None, None, None])
            #
            # if depth.shape[-3] != 1:
            #     depth = depth.unsqueeze(-3)
            # if depth[mask > 0.0].max() > 100:
            #     depth /= 1000  # to meters

        # compute gaussians
        if self.gs_regressor_frozen:
            with torch.no_grad():
                gs_regressor_out = self.gs_regressor.forward(
                    image=image,
                    image_feat=img_feat_lr,
                    depth=depth,
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    camera_idx=camera_idx,
                    rasterizers=self.rasterizers,
                    stitch_n_cams=1 if not stitch_partials else n_cams_per_sample,
                )
        else:
            gs_regressor_out = self.gs_regressor(
                image=image,
                image_feat=img_feat_lr,
                depth=depth,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                camera_idx=camera_idx,
                rasterizers=self.rasterizers,
                stitch_n_cams=1 if not stitch_partials else n_cams_per_sample,
            )

        # reshape
        out = []
        for tensor in gs_regressor_out:
            if tensor.shape[0] == b * n_cams_per_sample:
                tensor = self.llrr_to_batch_lr(tensor, n_cams_per_sample)
            out.append(tensor)
        return tuple(out)
