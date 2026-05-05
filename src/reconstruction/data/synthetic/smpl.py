import dataclasses
import inspect
import pickle
from functools import partial
from pathlib import Path
from typing import Optional, Union, Dict as DictT, Sequence, Tuple, List

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import numpy as np
import torch
from smplx.lbs import find_dynamic_lmk_idx_and_bcoords, vertices2landmarks, blend_shapes, lbs
from smplx.vertex_ids import vertex_ids as SMPL_VERTEX_IDS
from smplx.vertex_joint_selector import VertexJointSelector
from torch import nn, Tensor

from utils.misc import Gender, log, PathUtils
from utils.smpl import cam_crop_to_full, convert_yup, Bbox, Dict, full_perspective_projection, Nutrients, Pose2dFormat, Reflector

# FIX: Numpy types error when np version is greater than 1.23.0
np.bool = np.bool_
np.int = np.int_
np.float = np.float64
np.complex = np.complex128
np.object = np.object_
np.unicode = np.str_
np.str = np.str_


@dataclasses.dataclass(kw_only=True)
class SmplOutput:
    # inputs
    model_name: Optional[List[str]] = None
    gender: Optional[List[Gender]] = None
    age: Optional[List[Tensor]] = None
    betas: Optional[Tensor] = None
    expression: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    thetas: Optional[Tensor] = None
    left_hand_pose: Optional[Tensor] = None
    right_hand_pose: Optional[Tensor] = None
    jaw_pose: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    global_scale: Optional[Tensor] = None
    global_transl: Optional[Tensor] = None
    global_feet_transl: Optional[Tensor] = None
    uv_image: Optional[Tensor] = None
    uv_vector: Optional[Tensor] = None
    features: Optional[Tensor] = None
    # output
    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    pelvis_offset: Optional[Tensor] = None
    v_shaped: Optional[Tensor] = None

    @property
    def joints14(self) -> Tensor:
        return self.joints[..., -17:, :][..., [*Pose2dFormat('mpii14').to('h36m').values()], :]

    @property
    def vertices_yup(self) -> Tensor:
        return convert_yup(self.vertices)

    @property
    def vertices_cv2gl(self) -> Tensor:
        return self.vertices_yup

    def get_idx_for2d(self, pose2d_fmt: Union[Pose2dFormat, str]) -> Optional[List[int]]:
        if self.joints is None:
            return None
        pose2d_fmt = Pose2dFormat(pose2d_fmt)
        return list(pose2d_fmt.to_smpl(self.model_name[0] if self.model_name is not None else 'smpl').values())

    def get_joints_offset(self, pose2d_fmt: Union[Pose2dFormat, str]) -> Optional[Tensor]:
        if self.joints is None:
            return None
        pose2d_fmt = Pose2dFormat(pose2d_fmt)
        joint_idx = self.get_idx_for2d(pose2d_fmt)
        joint_root_idx = sorted(list(pose2d_fmt.hip_indices))
        joints2d_offset = self.joints[..., [joint_idx[_] for _ in joint_root_idx], :].mean(-2).unsqueeze(-2)
        return joints2d_offset

    def get_joints_centered(self, target_2d_fmt: Union[Pose2dFormat, str]) -> Optional[Tensor]:
        if self.joints is None:
            return None
        joints2d_offset = self.get_joints_offset(target_2d_fmt)
        joints2d_idx = self.get_idx_for2d(target_2d_fmt)
        return self.joints[..., joints2d_idx, :] - joints2d_offset

    def get_vertices_centered(self, target_2d_fmt: Union[Pose2dFormat, str] = 'pelvis') -> Optional[Tensor]:
        if self.joints is None:
            return self.vertices
        if target_2d_fmt == 'pelvis':
            joints_offset = self.joints[..., [0], :]
        else:
            joints_offset = self.get_joints_offset(target_2d_fmt)
        return self.vertices - joints_offset


@dataclasses.dataclass(kw_only=True)
class SmplProjectedOutput:
    patch_bbox_xyxy: Optional[Tensor] = None
    tight_bbox_xyxy: Optional[Tensor] = None
    frame_bbox_xyxy: Optional[Tensor] = None
    patch_to_bbox_center_offset: Optional[Tensor] = None
    input_size_wh: Optional[Tensor] = None
    patch_size_wh: Optional[Tensor] = None
    frame_size_wh: Optional[Tensor] = None
    focal_length: Optional[Tensor] = None
    focal_length_patch: Optional[Tensor] = None
    focal_length_full_frame: Optional[Tensor] = None
    cam: Optional[Tensor] = None
    weak_cam_t: Optional[Tensor] = None  # wrt resized patch (i.e. the input to the HMR model)
    patch_cam_t: Optional[Tensor] = None  # wrt original patch
    full_frame_cam_t: Optional[Tensor] = None  # wrt to frame (original, no resize)
    weak_joints2d: Optional[Tensor] = None
    weak_joints2d17: Optional[Tensor] = None
    patch_joints2d: Optional[Tensor] = None
    patch_joints2d17: Optional[Tensor] = None
    full_frame_joints2d: Optional[Tensor] = None
    full_frame_joints2d17: Optional[Tensor] = None

    def compute_patch_to_bbox_center_offset_(self) -> None:
        if self.tight_bbox_xyxy is None or self.patch_bbox_xyxy is None:
            return None
        self.patch_to_bbox_center_offset = (
                (self.patch_bbox_xyxy[..., 2:] - self.patch_bbox_xyxy[..., :2]) / 2
                - (self.tight_bbox_xyxy[..., 2:] - self.tight_bbox_xyxy[..., :2]) / 2)


@dataclasses.dataclass(kw_only=True)
class SmplWithCamOutput:
    smpl: SmplOutput
    projection: SmplProjectedOutput

    def asdict(self, flatten: bool = False, flatten_delimiter: Optional[str] = ':') -> Dict:
        dict_out = Dict(smpl=self.smpl.__dict__, projection=self.projection.__dict__)
        if flatten:
            return dict_out.flatten(flatten_delimiter).filter(dtype=(Tensor, Sequence))
        return dict_out


# TODO: Compare with WHAM
# noinspection PyTypeChecker
class Smpl(nn.Module):
    NUM_JOINTS = 23  # excluding root/pelvis joint
    NUM_JOINTS_AUGMENTED = 45  # excluding root/pelvis joint
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    PROPERTY_DIMS = {
        'betas': NUM_BETAS,
        'thetas': NUM_JOINTS * 3,
        'global_orientation': 3,
        'global_translation': 3,
    }
    __TEMPLATES__: DictT[str, DictT[Gender, DictT[str, Tensor]]] = {
        k: {g: None for g in Gender}
        for k in ['smpl', 'smplh', 'smplx']
    }
    __VERTEX_REGRESSORS__: DictT[str, Union[Tensor, nn.Module]] = {}

    def __init__(self,
                 model_name: str = 'smpl',
                 vertex_selector: Optional[DictT[str, int]] = None,
                 vertex_regressors: Sequence[Union[str, Path, np.ndarray]] = (),
                 joint_mapper: Optional[nn.Module] = None,
                 **kwargs):
        """
        SMPL model constructor.

        Parameters
        ----------
        model_name: str
            One of 'smpl', 'smplh', 'smplx'. Currently, 'smplh' has not been implemented, and therefore it will behave
            as if it was 'smpl'.
        vertex_selector: dict, optional
            A dictionary containing the indices of the extra vertices that will be selected from the SMPL model
            to match the vertex indices of the dataset. SMPL and SMPL-H share the same topology, so any extra joints
            can be drawn from the same place. (default = None)
        vertex_regressors: Sequence[Path, np.ndarray], optional
            If present, extra joints will be regressed from vertices and will be concatenated to the SMPL joints.
            (default = empty sequence)
        joint_mapper: nn.Module or None
            An object that re-maps the joints. Useful if one wants to re-order the SMPL joints to some other
            convention (e.g. MS-COCO). (default = None)
        """
        super(Smpl, self).__init__()

        # get path to gender files
        assert model_name in ['smpl', 'smplh', 'smplx', 'star', 'supr'], f'model_name not recognized: `{model_name}`'
        self.model_name = model_name
        self.model_version = '1.0' if model_name == 'supr' else '1.1'
        self.model_path_fn = partial(PathUtils.smpl_path, model=model_name, model_version=self.model_version)

        # Add vertex selector (for extra joints derived from vertices)
        if vertex_selector is None and model_name == 'smpl':
            vertex_selector = SMPL_VERTEX_IDS['smplh']
        if vertex_selector is not None:
            self.register_module('vertex_selector', VertexJointSelector(vertex_ids=vertex_selector, **kwargs))
            self.vertex_selector_joint_names = list(vertex_selector.keys())
        else:
            self.vertex_selector_joint_names = []

        # Add joint mapper
        if joint_mapper is not None:
            self.register_module('joint_mapper', joint_mapper)

        # Add vertex regressor (for extra joints derived from a weight combination of vertices)
        _vertex_regressors = []
        for vr in vertex_regressors:
            vr_key = None
            if isinstance(vr, (str, Path)):
                vr_key = str(Path(vr).stem)
            if vr_key is not None and vr_key in self.__VERTEX_REGRESSORS__.keys():
                vr = self.__VERTEX_REGRESSORS__[vr_key]
            else:
                if isinstance(vr, (str, Path)):
                    vr = Path(vr)
                    if not vr.exists():
                        vr = PathUtils.checkpoints_path() / 'smpl' / 'joint_regressors' / str(vr)
                    if vr.suffix in ['.npy', '.npz']:
                        vr = np.load(vr)
                    elif vr.suffix in ['.pkl', '.pickle']:
                        vr = pickle.load(open(vr, 'rb'), encoding='latin1')
                    else:
                        raise ValueError(f'[Smpl::__init__] Joint regressor file type not recognized: {vr}')
                if not isinstance(vr, Tensor):
                    vr = torch.tensor(vr)
                if vr_key is not None:
                    self.__VERTEX_REGRESSORS__[vr_key] = vr.squeeze()
            _vertex_regressors.append(vr.to(dtype=torch.float))
        if len(_vertex_regressors) > 0:
            self.register_buffer('vertex_regressor', torch.cat(_vertex_regressors, dim=0))

    @property
    def _template_keys(self) -> DictT[str, str]:
        return {
            'f': 'faces_tensor',
            'shapedirs': 'shapedirs',
            'posedirs': 'posedirs',
            'v_template': 'v_template',
            'J_regressor': 'J_regressor',
            'weights': 'lbs_weights',
            'kintree_table': 'parents',
        }

    def _template(self, gender: Gender, device: Union[str, torch.device] = 'cpu'):
        if gender not in self.__TEMPLATES__[self.model_name] or self.__TEMPLATES__[self.model_name][gender] is None:
            # load template
            log(f'[Smpl::_template] Allocating template for {self.model_name}-{gender.full}')
            template_data = self.load_file(self.model_path_fn(gender))
            # pick certain keys and remap them
            remap_keys = self._template_keys
            self.__TEMPLATES__[self.model_name][gender] = {}
            for key, tensor in Nutrients.tensorize(template_data).items():
                if key not in remap_keys:
                    continue
                if key == 'posedirs':
                    # Pose blend shape basis: 6890 x 3 x 207 --> reshaped to 6890*3 x 207 --> and to 207 x 20670
                    tensor = torch.reshape(tensor, [-1, tensor.shape[-1]]).T.contiguous()
                elif key == 'kintree_table':
                    tensor = tensor[0]
                    tensor[0] = -1
                elif key == 'shapedirs' and self.NUM_BETAS is not None:
                    tensor = tensor[:, :, :self.NUM_BETAS]
                self.__TEMPLATES__[self.model_name][gender][remap_keys[key]] = \
                    (tensor.float() if tensor.dtype == torch.double else tensor)
        return Dict(self.__TEMPLATES__[self.model_name][gender]).tensorify(device=device)

    def get_faces(self, gender: Gender) -> Tensor:
        return self._template(gender)['faces_tensor']

    # noinspection DuplicatedCode
    def forward(self,
                betas: Tensor,  # shape (batch_size, NUM_BETAS)
                thetas: Tensor,  # shape (batch_size, NUM_SMPL_JOINTS, 3, 3) or (batch_size, NUM_SMPL_JOINTS, 3)
                gender: Gender = Gender.NEUTRAL,
                transl: Optional[Tensor] = None,  # shape (batch_size, 3)
                global_orient: Optional[Tensor] = None,  # shape (batch_size, 3, 3) or (batch_size, 3)
                global_transl: Optional[Tensor] = None,  # shape (batch_size, 3)
                global_scale: Optional[Tensor] = None,  # shape (batch_size, 3)
                return_verts: bool = True,
                return_full_pose: bool = False,  # if True, the first part of the joints will be the global orientation
                return_lbs: bool = False,
                center_on_pelvis: bool = False,
                convert_to_yup: bool = False) -> Union[SmplOutput, Tuple[Tensor, Tensor]]:
        """
        Forward pass for the SMPL model

        Returns
        -------
        namedtuple
            A named tuple containing the following items:
            - vertices: The output vertices of the SMPL model. Shape is (batch_size x 6890 x 3).
            - joints: The joint locations after forward kinematics on the given poses. Shape is (batch_size x 23 x 3).
            - full_pose: The axis-angle representation of the joint rotations. Shape is (batch_size x 72).
            - betas: The shape parameters of the model. Shape is (batch_size x 10).
            - global_orient: The global orientation of the body. Shape is (batch_size x 3).
            - thetas: The pose of the body. Shape is (batch_size x 63).
            - expression: The expression of the body. Shape is (batch_size x 10).
            - left_hand_pose: The pose of the left hand. Shape is (batch_size x 15).
            - right_hand_pose: The pose of the right hand. Shape is (batch_size x 15).
            - jaw_pose: The pose of the jaw. Shape is (batch_size x 3).
        """
        # Get SMPL template
        template = self._template(gender, betas.device)
        # Prepare input
        batch_size = max(betas.shape[0], thetas.shape[0])
        if global_orient is not None:
            batch_size = max(batch_size, global_orient.shape[0])
        if global_orient is not None and global_orient.shape[0] != batch_size:
            num_repeats = int(batch_size / global_orient.shape[0])
            global_orient = global_orient.repeat(num_repeats, 1)
        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.repeat(num_repeats, 1)
        if thetas.shape[0] != batch_size:
            num_repeats = int(batch_size / thetas.shape[0])
            thetas = thetas.repeat(num_repeats, 1)
        betas = betas[..., :self.NUM_BETAS]
        # Run SMPL
        if global_orient is None:
            split_dim = -3 if list(thetas.shape[-2:]) == [3, 3] else -2
            n_joints = thetas.shape[split_dim]
            global_orient, thetas = torch.split(thetas, [1, n_joints - 1], dim=split_dim)
        if global_orient.ndim != thetas.ndim:
            assert global_orient.ndim == thetas.ndim - 1
            global_orient = global_orient.unsqueeze(-3 if list(thetas.shape[-2:]) == [3, 3] else -2)
        full_pose = torch.cat([global_orient, thetas], dim=1) if global_orient is not None else thetas
        if 'pose_mean' in self._parameters.keys():
            # Add the mean pose of the model. Does not affect the body, only the
            # hands when flat_hand_mean == False
            full_pose += self.pose_mean
        vertices, joints = lbs(betas,
                               full_pose,
                               **Reflector.collect_args(lbs, template),
                               pose2rot=list(full_pose.shape[-2:]) != [3, 3])
        # Center mesh so that pelvis is at (0,0,0)
        pelvis_offset = joints[..., [0], :] if center_on_pelvis else torch.zeros_like(joints[..., [0], :])
        joints = joints - pelvis_offset
        vertices = vertices - pelvis_offset
        if transl is not None:
            vertices = vertices + transl.unsqueeze(1)
            joints = joints + transl.unsqueeze(1)
        if return_lbs:
            if convert_to_yup:
                vertices = convert_yup(vertices)
                joints = convert_yup(joints)
            return vertices, joints, full_pose
        # Add selected vertices to joints
        if 'vertex_selector' in self._modules.keys():
            joints = self.vertex_selector(vertices, joints)
        if 'vertex_regressor' in self._buffers.keys():
            extra_joints = self.vertex_regressor @ vertices
            joints = torch.cat((joints, extra_joints), dim=1)
        # Map the joints to a new data format (e.g. from SMPL -> COCO)
        if 'joint_mapper' in self._modules.keys():
            joints = self.joint_mapper(joints)
        # Move pelvis to requested global_transl (after this, the pelvis WILL BE EXACTLY AT global_transl)
        if global_scale is not None:
            joints = joints * global_scale.mean(0, keepdim=True)
            vertices = vertices * global_scale.mean(0, keepdim=True)
        if global_transl is not None:
            if global_transl.ndim == 2:
                global_transl = global_transl.unsqueeze(1)
            joints = joints + global_transl
            vertices = vertices + global_transl
        # Return results
        return SmplOutput(
            # inputs
            model_name=[self.model_name] * batch_size,
            gender=[gender] * batch_size,
            full_pose=full_pose if return_full_pose else None,
            betas=betas,
            thetas=thetas,
            global_orient=global_orient,
            global_scale=torch.ones(batch_size, 1, 1, dtype=vertices.dtype, device=vertices.device) if global_scale is None else global_scale,
            transl=torch.zeros(batch_size, 3, device=thetas.device, dtype=thetas.dtype, requires_grad=True) if transl is None else transl,
            global_transl=global_transl,
            global_feet_transl=vertices.reshape(batch_size, -1, 3)[:, [3216, 3387, 6617, 6787], :],
            # outputs
            vertices=vertices if return_verts else None,
            joints=joints,
            pelvis_offset=pelvis_offset,
        )

    # noinspection DuplicatedCode
    def forward_with_cam(self,
                         gender: Gender,
                         patch_bbox: Bbox,  # this is the TIGHT bbox resized to aspect ratio 196/256
                         frame_bbox: Optional[Bbox] = None,
                         patch_resized_wh: Tuple[int, int] = (192, 256),
                         pred_cam: Optional[Tensor] = None,  # (B, 3)
                         weak_cam_t: Optional[Tensor] = None,  # (B, 3)
                         patch_cam_t: Optional[Tensor] = None,  # (B, 3)
                         full_frame_cam_t: Optional[Tensor] = None,  # (B, 3)
                         focal_length: Union[float, Tensor] = 5000.,  # float or (B, 2)
                         pose2d_fmt: Union[Pose2dFormat, str] = Pose2dFormat.COCO17,
                         *forward_args,
                         **forward_kwargs) -> SmplWithCamOutput:
        # SMPL
        smpl_output = self.forward(gender=gender, *forward_args, **forward_kwargs)
        smpl_output.vertices = smpl_output.get_vertices_centered(pose2d_fmt)
        # # Cam
        # if isinstance(focal_length, (int, float)) and any(_ is not None for _ in [pred_cam, weak_cam_t, patch_cam_t]):
        #     if pred_cam is not None:
        #         B = pred_cam.shape[0]
        #         device = pred_cam.device
        #         dtype = pred_cam.dtype
        #     elif weak_cam_t is not None:
        #         B = weak_cam_t.shape[0]
        #         device = weak_cam_t.device
        #         dtype = weak_cam_t.dtype
        #     else:
        #         B = patch_cam_t.shape[0]
        #         device = patch_cam_t.device
        #         dtype = patch_cam_t.dtype
        #     focal_length = focal_length * torch.ones(B, 2).to(dtype=dtype, device=device)
        # compensate for the input/patch/frame size by adjusting focal plane
        focal_length_smpl, size_input = focal_length, 256
        if weak_cam_t is None and pred_cam is not None:
            #   - weak perspective projection --> perspective projection
            weak_cam_t = torch.stack([
                pred_cam[:, 1],
                pred_cam[:, 2],
                2 * focal_length_smpl / (size_input * pred_cam[:, 0] + 1e-9)
            ], dim=-1)
            weak_cam_t += smpl_output.pelvis_offset.squeeze(1)  # + smpl_output.get_joints_offset(pose2d_fmt).squeeze(1)
        #   - patch-resized cam ("weak") --> patch-original cam ("full")
        focal_length_patch = focal_length_smpl / (size_input / patch_bbox.frame_wh.max(dim=-1)[0])
        if patch_cam_t is None and pred_cam is not None:
            patch_cam_t = cam_crop_to_full(pred_cam.clone(),
                                           patch_bbox.cxys[..., :2],
                                           patch_bbox.cxys[..., 2],
                                           patch_bbox.frame_wh,
                                           focal_length_patch)
            patch_cam_t += smpl_output.pelvis_offset.squeeze(1) + smpl_output.get_joints_offset(pose2d_fmt).squeeze(1)
        #   - patch-resized cam ("weak") --> full frame cam ("full_frame")
        focal_length_full_frame = focal_length_smpl / size_input * frame_bbox.frame_wh.max(dim=-1)[0]
        if pred_cam is not None and frame_bbox is not None and full_frame_cam_t is None:
            full_frame_cam_t = cam_crop_to_full(pred_cam,
                                                frame_bbox.cxys[..., :2],
                                                frame_bbox.cxys[..., 2],
                                                frame_bbox.frame_wh,
                                                focal_length_full_frame)
            full_frame_cam_t += (smpl_output.pelvis_offset + smpl_output.get_joints_offset(pose2d_fmt)).squeeze(1)
        # create output dict
        projection_output = SmplProjectedOutput(
            cam=pred_cam if pred_cam is not None else None,
            # focal lengths
            focal_length=torch.tensor(focal_length_smpl, device=focal_length_patch.device).expand_as(focal_length_patch),
            focal_length_patch=focal_length_patch,
            focal_length_full_frame=focal_length_full_frame,
            # cam translations
            # weak_cam_t=weak_cam_t,
            # patch_cam_t=patch_cam_t,
            full_frame_cam_t=full_frame_cam_t,
            # bbox
            patch_bbox_xyxy=patch_bbox.xyxy,
            patch_size_wh=patch_bbox.frame_wh,
            frame_bbox_xyxy=frame_bbox.xyxy,
            frame_size_wh=frame_bbox.frame_wh,
        )
        # get full projection
        smpl_joints2d17 = smpl_output.get_joints_centered(pose2d_fmt)
        # # keypoints to 2D (normalized to patch center)
        # if weak_cam_t is not None:
        #     projection_output.weak_joints2d = perspective_projection(
        #         smpl_output.joints,
        #         translation=projection_output.weak_cam_t - smpl_output.get_joints_offset(pose2d_fmt).squeeze(1),
        #         focal_length=projection_output.focal_length.repeat(1, 2),
        #     ).reshape(projection_output.weak_cam_t.shape[0], -1, 2) + float(size_input // 2)
        #     projection_output.weak_joints2d[..., 0] /= (patch_resized_wh[0] / 2.)
        #     projection_output.weak_joints2d[..., 1] /= (patch_resized_wh[1] / 2.)
        #     projection_output.weak_joints2d17 = perspective_projection(
        #         smpl_joints2d17,
        #         translation=projection_output.weak_cam_t,
        #         focal_length=projection_output.focal_length.repeat(1, 2),
        #     ).reshape(projection_output.weak_cam_t.shape[0], -1, 2) + float(size_input // 2)
        #     projection_output.weak_joints2d17[..., 0] /= (patch_resized_wh[0] / 2.)
        #     projection_output.weak_joints2d17[..., 1] /= (patch_resized_wh[1] / 2.)
        # # keypoints to 2D (wrt to patch coordinates)
        # if patch_cam_t is not None:
        #     patch_cam_intrinsics = torch.zeros(patch_cam_t.shape[0], 3, 3, dtype=patch_cam_t.dtype,
        #                                        device=patch_cam_t.device)
        #     patch_cam_intrinsics[:, 0, 0] = focal_length_patch
        #     patch_cam_intrinsics[:, 1, 1] = focal_length_patch
        #     patch_cam_intrinsics[:, 2, 2] = 1.
        #     patch_cam_intrinsics[:, :-1, -1] = patch_bbox.frame_wh / 2.0
        #     projection_output.patch_joints2d = full_perspective_projection(
        #         smpl_output.joints,
        #         translation=projection_output.patch_cam_t - smpl_output.get_joints_offset(pose2d_fmt).squeeze(1),
        #         cam_intrinsics=patch_cam_intrinsics,
        #     )  # / (patch_bbox.frame_wh / 2.).unsqueeze(1)
        #     projection_output.patch_joints2d17 = full_perspective_projection(
        #         smpl_joints2d17,
        #         translation=projection_output.patch_cam_t,
        #         cam_intrinsics=patch_cam_intrinsics,
        #     )  # / (patch_bbox.frame_wh / 2.).unsqueeze(1)
        # # keypoints to 2D (wrt to original frame coordinates)
        if full_frame_cam_t is not None:
            full_frame_cam_intrinsics = torch.zeros(full_frame_cam_t.shape[0], 3, 3, dtype=full_frame_cam_t.dtype,
                                                    device=full_frame_cam_t.device)
            full_frame_cam_intrinsics[:, 0, 0] = focal_length_full_frame
            full_frame_cam_intrinsics[:, 1, 1] = focal_length_full_frame
            full_frame_cam_intrinsics[:, 2, 2] = 1.
            full_frame_cam_intrinsics[:, :2, -1] = frame_bbox.frame_wh / 2.0
            projection_output.full_frame_joints2d = full_perspective_projection(
                smpl_output.joints,
                translation=projection_output.full_frame_cam_t,
                cam_intrinsics=full_frame_cam_intrinsics,
            )
            projection_output.full_frame_joints2d17 = full_perspective_projection(
                smpl_joints2d17,
                translation=projection_output.full_frame_cam_t,
                cam_intrinsics=full_frame_cam_intrinsics,
            )
        # combine and return
        return SmplWithCamOutput(smpl=smpl_output, projection=projection_output)

    @staticmethod
    def load_file(path: Path) -> DictT[str, np.ndarray]:
        assert path.exists(), f'[Smpl::load_from_file] Path {path} does not exist!'
        if path.suffix in ['.npz', '.npy']:
            data_struct = np.load(path, allow_pickle=True)
            if isinstance(data_struct, np.lib.npyio.NpzFile):
                npz = data_struct
                data_struct = {npz.files[i]: npz[npz.files[i]] for i in range(len(npz.files))}
                npz.close()
            elif path.suffix == '.npy' and hasattr(data_struct, 'item') and type(data_struct.item()) == dict:
                data_struct = data_struct.item()
        else:
            with open(path, 'rb') as smpl_file:
                data_struct = pickle.load(smpl_file, encoding='latin1')
        return data_struct


class Smplx(Smpl):
    NUM_JOINTS = 54  # excluding root/pelvis joint
    NUM_JOINTS_AUGMENTED = 55
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    NUM_BODY_JOINTS = NUM_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
    NUM_EXPR_COEFS = 10
    PROPERTY_DIMS = {
        'betas': Smpl.NUM_BETAS,
        'expr': NUM_EXPR_COEFS,
        'thetas': NUM_JOINTS * 3,
        'global_orientation': 3,
        'global_translation': 3,
    }

    def __init__(self, use_face_contour: bool = False, *args, **kwargs):
        super().__init__(model_name='smplx', *args, **kwargs)
        self.use_face_contour = use_face_contour

    @property
    def _template_keys(self) -> DictT[str, str]:
        return super()._template_keys | {
            'lmk_faces_idx': 'lmk_faces_idx',
            'lmk_bary_coords': 'lmk_bary_coords',
        } | ({
                 'dynamic_lmk_faces_idx': 'dynamic_lmk_faces_idx',
                 'dynamic_lmk_bary_coords': 'dynamic_lmk_bary_coords',
             } if self.use_face_contour else {})

    # noinspection PyMethodOverriding,PyUnresolvedReferences,PyTypeChecker,DuplicatedCode
    def forward(self,
                expression: Tensor,  # shape (B, NUM_EXPR_COEFS)
                left_hand_thetas: Tensor,  # shape (B, NUM_HAND_JOINTS, 3), in angle-axis format
                right_hand_thetas: Tensor,  # shape (B, NUM_HAND_JOINTS, 3)
                jaw_pose: Tensor,  # shape (B, 3)
                leye_pose: Tensor,  # shape (B, 3)
                reye_pose: Tensor,  # shape (B, 3)
                return_shaped: bool = False,
                *args,
                **kwargs) -> SmplOutput:
        B = jaw_pose.shape[0]
        # Do LBS on the augmented body pose and shape/expression blend shapes
        kwargs['thetas'] = torch.cat([
            kwargs['thetas'],
            jaw_pose.reshape(-1, 1, 3),
            leye_pose.reshape(-1, 1, 3),
            reye_pose.reshape(-1, 1, 3),
            left_hand_thetas.reshape(-1, self.NUM_HAND_JOINTS, 3),
            right_hand_thetas.reshape(-1, self.NUM_HAND_JOINTS, 3)
        ], dim=1).flatten(1)
        vertices, joints, full_pose = super().forward(return_lbs=True,
                                                      return_full_pose=self.use_face_contour,
                                                      *args, **kwargs)
        # add face face_landmarks
        template = self._template(kwargs['gender'], device=vertices.device)
        lmk_faces_idx = template['lmk_faces_idx'].unsqueeze(dim=0).expand(B, -1).contiguous()
        lmk_bary_coords = template['lmk_bary_coords'].unsqueeze(dim=0).repeat(B, 1, 1)
        if self.use_face_contour:
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose,
                template['dynamic_lmk_faces_idx'],
                template['dynamic_lmk_bary_coords'],
                self.neck_kin_chain,
                pose2rot=True,
            )
            lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat([lmk_bary_coords.expand(B, -1, -1), dyn_lmk_bary_coords], 1)
        face_landmarks = vertices2landmarks(vertices, template['faces_tensor'], lmk_faces_idx, lmk_bary_coords)
        # Add selected vertices to joints
        if 'vertex_selector' in self._modules.keys():
            joints = self.vertex_selector(vertices, joints)
        # Add the face landmarks to the joints
        joints = torch.cat([joints, face_landmarks], dim=1)
        # Regress more joints from vertices
        if 'vertex_regressor' in self._buffers.keys():
            extra_joints = self.vertex_regressor @ vertices
            joints = torch.cat((joints, extra_joints), dim=1)
        # Map the joints to a new data format (e.g. from SMPL -> COCO)
        if 'joint_mapper' in self._modules.keys():
            joints = self.joint_mapper(joints)
        # Move pelvis to requested global_transl (after this, the pelvis WILL BE EXACTLY AT global_transl)
        global_transl = kwargs.pop('global_transl', None)
        global_scale = kwargs.pop('global_scale', None)
        if global_transl is not None:
            if global_scale is None:
                global_scale = torch.ones(B, 1, 1, dtype=vertices.dtype, device=vertices.device)
            if global_transl.ndim == 2:
                global_transl = global_transl.unsqueeze(1)
            cur_global_trans = joints[:, [0]]
            vertices = (vertices - cur_global_trans) * global_scale + global_transl
            joints = (joints - joints) * global_scale + global_transl
        # Map the joints to the current dataset
        v_shaped = None
        if return_shaped:
            v_shaped = template['v_template'] + \
                       blend_shapes(kwargs['betas'], template['shapedirs'][..., :self.NUM_BETAS])

        return SmplOutput(
            # inputs
            full_pose=full_pose if kwargs.get('return_full_pose', False) else None,
            betas=kwargs['betas'],
            expression=expression,
            thetas=kwargs['thetas'],
            left_hand_pose=left_hand_thetas,
            right_hand_pose=right_hand_thetas,
            jaw_pose=jaw_pose,
            global_orient=kwargs.get('global_orient', None),
            global_transl=kwargs.get('global_transl', None),
            # outputs
            vertices=vertices if kwargs.get('return_verts', False) else None,
            joints=joints,
            v_shaped=v_shaped
        )


if __name__ == '__main__':
    # # SMPL
    # smpl_ = Smpl()
    # betas_ = torch.zeros(1, 10)
    # thetas_ = torch.rand(1, 23, 3)
    # global_orient_ = torch.rand(1, 3)
    # out_ = smpl_.forward(
    #     gender=Gender.FEMALE,
    #     betas=betas_,
    #     thetas=thetas_,
    #     global_orient=global_orient_,
    #     return_verts=True,
    # )
    # print(out_)

    # SMPLX
    smplx_ = Smplx()
    betas_ = torch.zeros(1, 10)
    expression_ = torch.zeros(1, 10)
    thetas_ = torch.rand(1, 21, 3)
    global_orient_ = torch.rand(1, 3)
    left_hand_thetas_ = torch.rand(1, 15, 3)
    right_hand_thetas_ = torch.rand(1, 15, 3)
    jaw_pose_ = torch.rand(1, 3)
    leye_pose_ = torch.rand(1, 3)
    reye_pose_ = torch.rand(1, 3)
    out_ = smplx_.forward(
        gender=Gender.FEMALE,
        betas=betas_,
        expression=expression_,
        thetas=thetas_,
        global_orient=global_orient_,
        left_hand_thetas=left_hand_thetas_,
        right_hand_thetas=right_hand_thetas_,
        jaw_pose=jaw_pose_,
        leye_pose=leye_pose_,
        reye_pose=reye_pose_,
        return_verts=True,
        return_shaped=True,
    )
    print(out_.vertices.shape, out_.joints.shape)  # (1, 10475, 3) (1, 106, 3)
