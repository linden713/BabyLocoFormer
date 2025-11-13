from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Mapping, Sequence

import isaaclab.utils.math as math_utils
import omni.usd
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from pxr import UsdGeom

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class compute_nominal_heights(ManagerTermBase):
    """Cache nominal base heights by reading link geometry directly from the USD stage."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._stage = omni.usd.get_context().get_stage()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | Sequence[int] | None,
        asset_cfg: SceneEntityCfg,
        segments: Sequence[Mapping[str, object]] | None = None,
        base_offset: float | int | None = 0.0,
    ):
        asset: Articulation = env.scene[asset_cfg.name]
        indices = self._normalize_env_ids(env, env_ids)
        buffer = self._ensure_buffer(env, asset)

        if not segments:
            default_root_state = asset.data.default_root_state
            if default_root_state is None:
                raise RuntimeError(
                    "Unable to compute nominal heights because the articulation does not expose a default root state."
                )
            buffer[indices] = default_root_state[indices, 2]
            return

        stage = self._stage
        link_paths = asset.root_physx_view.link_paths
        body_to_index = {name: idx for idx, name in enumerate(asset.data.body_names)}
        offset_value = float(base_offset) if base_offset is not None else 0.0

        for env_id in indices.tolist():
            nominal_height = offset_value
            for spec in segments:
                body_name = str(spec["body"])
                rel_path = str(spec.get("geometry_path", ""))
                multiplier = float(spec.get("multiplier", 1.0))

                body_idx = body_to_index[body_name]
                link_path = self._resolve_link_path(link_paths, env_id, body_idx)
                prim_path = self._resolve_prim_path(stage, link_path, rel_path)
                nominal_height += multiplier * self._read_cylinder_height(stage, prim_path)

            buffer[env_id] = nominal_height

        # Also update the default root state's z so generic reset uses it.
        if asset.data.default_root_state is not None:
            # clone-assign only the z column
            default_rs = asset.data.default_root_state
            default_rs[indices, 2] = buffer[indices]

        # Debug print (once at startup) to verify heights are applied.
        # try:
        #     bh = buffer[indices].detach().cpu()
        #     dz = (
        #         default_rs[indices, 2].detach().cpu() - bh
        #         if asset.data.default_root_state is not None
        #         else torch.zeros_like(bh)
        #     )
        #     msg = (
        #         f"[babylocoformer] nominal_heights per-env: min={bh.min().item():.3f}, max={bh.max().item():.3f}; "
        #         f"max|default_z - nominal|={dz.abs().max().item():.6f}. Samples: "
        #         f"{bh[:8].tolist()}"
        #     )
        #     print(msg)
        # except Exception:
        #     pass

    @staticmethod
    def _normalize_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | Sequence[int] | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=env.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)

    @staticmethod
    def _resolve_link_path(link_paths: Sequence, env_id: int, body_idx: int) -> str:
        try:
            env_paths = link_paths[env_id]  # type: ignore[index]
        except (IndexError, TypeError):
            env_paths = link_paths[0]  # type: ignore[index]
        return str(env_paths[body_idx])

    def _resolve_prim_path(self, stage, link_path: str, relative_path: str) -> str:
        if not relative_path:
            return link_path.rstrip("/")

        candidate = f"{link_path.rstrip('/')}/{relative_path.lstrip('/')}"
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid():
            return candidate

        mesh_candidate = f"{link_path.rstrip('/')}/mesh_0/{relative_path.lstrip('/')}"
        prim = stage.GetPrimAtPath(mesh_candidate)
        if prim and prim.IsValid():
            return mesh_candidate

        raise RuntimeError(f"Prim '{candidate}' not found while computing nominal height.")

    def _read_cylinder_height(self, stage, prim_path: str) -> float:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim '{prim_path}' not found while computing nominal height.")
        if not prim.IsA(UsdGeom.Cylinder):
            raise RuntimeError(f"Prim '{prim_path}' is not a UsdGeom.Cylinder.")
        height = UsdGeom.Cylinder(prim).GetHeightAttr().Get()
        if height is None:
            raise RuntimeError(f"Cylinder prim '{prim_path}' does not define a height attribute.")
        return float(height)

    @staticmethod
    def _ensure_buffer(env: ManagerBasedEnv, asset: Articulation) -> torch.Tensor:
        dtype = asset.data.default_root_state.dtype if asset.data.default_root_state is not None else torch.float32
        if not hasattr(env, "_nominal_heights") or env._nominal_heights.shape[0] != env.num_envs:
            env._nominal_heights = torch.zeros(env.num_envs, dtype=dtype, device=env.device)
        else:
            env._nominal_heights = env._nominal_heights.to(device=env.device, dtype=dtype)
        return env._nominal_heights


# def nominal_height_obs(env: ManagerBasedEnv) -> torch.Tensor:
#     """Return cached nominal heights as a single-scalar observation per env.

#     Assumes :class:`compute_nominal_heights` was executed at startup. When the
#     buffer is not available, returns zeros to avoid breaking the pipeline.
#     """
#     if not hasattr(env, "_nominal_heights") or env._nominal_heights is None:
#         return torch.zeros((env.num_envs, 1), device=env.device)
#     return env._nominal_heights.reshape(-1, 1)


# def set_root_z_to_nominal(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor | Sequence[int] | None,
#     z_offset: float = 0.0,
#     z_noise_range: tuple[float, float] = (0.0, 0.0),
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):
#     """Set base height to nominal (+ offset + noise) per environment.

#     Use as a reset event after position/orientation randomization to ensure each
#     morphology starts at a consistent, feasible height.
#     """
#     asset = env.scene[asset_cfg.name]
#     # normalize env ids
#     if env_ids is None:
#         env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
#     elif not isinstance(env_ids, torch.Tensor):
#         env_ids = torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)

#     pos = asset.data.root_pos_w[env_ids].clone()
#     quat = asset.data.root_quat_w[env_ids].clone()

#     if hasattr(env, "_nominal_heights") and env._nominal_heights is not None:
#         nominal = env._nominal_heights[env_ids]
#     else:
#         nominal = pos[:, 2]

#     lo, hi = z_noise_range
#     if hi != 0.0 or lo != 0.0:
#         noise = torch.empty_like(nominal).uniform_(lo, hi)
#     else:
#         noise = torch.zeros_like(nominal)

#     pos[:, 2] = nominal + float(z_offset) + noise
#     asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)


def randomize_joint_pd_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int] | None,
    stiffness_scale_range: tuple[float, float] = (0.8, 1.2),
    damping_scale_range: tuple[float, float] = (0.8, 1.2),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*"),
):
    """Randomly scale joint PD gains at reset.

    Multiplies current PhysX DOF stiffness/damping by a per-env scalar sampled
    uniformly from the given ranges. Works regardless of actuator type.
    """
    asset = env.scene[asset_cfg.name]

    # normalize env ids
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)

    joint_ids = asset_cfg.joint_ids if getattr(asset_cfg, "joint_ids", None) else slice(None)

    k = asset.data.joint_stiffness[env_ids][:, joint_ids]
    d = asset.data.joint_damping[env_ids][:, joint_ids]

    ks = torch.empty((len(env_ids), 1), device=asset.device).uniform_(*stiffness_scale_range)
    ds = torch.empty((len(env_ids), 1), device=asset.device).uniform_(*damping_scale_range)

    asset.write_joint_stiffness_to_sim(k * ks, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_damping_to_sim(d * ds, joint_ids=joint_ids, env_ids=env_ids)


def set_joint_position_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int] | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*"),
    lower: float = 0.0,
    upper: float = 0.0,
):
    """Override specified joints' PhysX limits. Passing lower==upper effectively locks the DOF."""
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device

    if upper < lower:
        raise ValueError(f"upper limit {upper} smaller than lower limit {lower}.")

    if env_ids is None:
        env_ids_t = torch.arange(env.num_envs, device=device, dtype=torch.long)
    elif isinstance(env_ids, torch.Tensor):
        env_ids_t = env_ids.to(device=device, dtype=torch.long)
    else:
        env_ids_t = torch.as_tensor(list(env_ids), device=device, dtype=torch.long)

    joint_ids = getattr(asset_cfg, "joint_ids", None)
    if joint_ids is None:
        joint_ids_t = torch.arange(asset.num_joints, device=device, dtype=torch.long)
    elif isinstance(joint_ids, slice):
        joint_ids_t = torch.arange(asset.num_joints, device=device, dtype=torch.long)[joint_ids]
    else:
        joint_ids_t = torch.as_tensor(joint_ids, dtype=torch.long, device=device)

    if env_ids_t.numel() == 0 or joint_ids_t.numel() == 0:
        return

    limits = torch.empty((env_ids_t.numel(), joint_ids_t.numel(), 2), device=device, dtype=asset.data.joint_pos.dtype)
    limits[..., 0] = lower
    limits[..., 1] = upper

    asset.write_joint_position_limit_to_sim(limits, joint_ids=joint_ids_t.tolist(), env_ids=env_ids_t)


def randomize_rigid_body_com_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[env_ids[:, None], body_ids, :3] = rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)
