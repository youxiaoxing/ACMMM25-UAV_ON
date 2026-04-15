import argparse
import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import airsim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool


StateKey = Tuple[int, int, int, int]  # x_idx, y_idx, z_idx, yaw_bin
WorldPoint3D = Tuple[float, float, float]
Action = Tuple[str, float]


@dataclass
class Episode:
    episode_id: str
    map_name: str
    start_position: WorldPoint3D
    start_quaternionr: Sequence[float]
    goal_position: WorldPoint3D
    object_name: str


@dataclass
class PlannerConfig:
    horizontal_step: float
    vertical_step: float
    yaw_step_deg: float
    state_xy_resolution: float
    state_z_resolution: float
    search_margin_xy_m: float
    search_margin_z_m: float
    edge_check_step: float
    max_expansions: int
    goal_tolerance_xy: float
    goal_tolerance_z: float


def load_episodes(gt_json_path: str) -> List[Episode]:
    with open(gt_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw if isinstance(raw, list) else [raw]
    episodes: List[Episode] = []

    for item in items:
        goal_pos = item["pose"][0] if isinstance(item.get("pose", []), list) and len(item["pose"]) > 0 else item["pose"]
        episodes.append(
            Episode(
                episode_id=str(item["episode_id"]),
                map_name=item["map_name"].replace("_test", ""),
                start_position=tuple(item["start_pose"]["start_position"]),
                start_quaternionr=item["start_pose"]["start_quaternionr"],
                goal_position=tuple(goal_pos),
                object_name=item.get("object_name", "unknown"),
            )
        )

    return episodes


def _normalize_deg(deg: float) -> float:
    v = deg % 360.0
    if v < 0:
        v += 360.0
    return v


def quaternion_to_yaw_deg(quat_xyzw: Sequence[float]) -> float:
    q = airsim.Quaternionr(quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3])
    _, _, yaw = airsim.to_eularian_angles(q)
    return _normalize_deg(math.degrees(yaw))


def yaw_bin_from_deg(yaw_deg: float, yaw_step_deg: float) -> int:
    num_bins = int(round(360.0 / yaw_step_deg))
    return int(round(_normalize_deg(yaw_deg) / yaw_step_deg)) % num_bins


def yaw_deg_from_bin(yaw_bin: int, yaw_step_deg: float) -> float:
    num_bins = int(round(360.0 / yaw_step_deg))
    return (yaw_bin % num_bins) * yaw_step_deg


def world_to_idx(x: float, resolution: float) -> int:
    return int(round(x / resolution))


def idx_to_world(idx: int, resolution: float) -> float:
    return float(idx * resolution)


def state_to_world(state: StateKey, cfg: PlannerConfig) -> WorldPoint3D:
    x, y, z, _ = state
    return (
        idx_to_world(x, cfg.state_xy_resolution),
        idx_to_world(y, cfg.state_xy_resolution),
        idx_to_world(z, cfg.state_z_resolution),
    )


def yaw_from_state(state: StateKey, cfg: PlannerConfig) -> float:
    return yaw_deg_from_bin(state[3], cfg.yaw_step_deg)


def _probe_collision(sim_tool: AirVLNSimulatorClientTool, point: WorldPoint3D, yaw_deg: float) -> bool:
    yaw_rad = math.radians(yaw_deg)
    q = airsim.to_quaternion(0.0, 0.0, yaw_rad)
    pose = airsim.Pose(
        position_val=airsim.Vector3r(point[0], point[1], point[2]),
        orientation_val=airsim.Quaternionr(q.x_val, q.y_val, q.z_val, q.w_val),
    )
    ok = sim_tool.setPoses([[pose]])
    if not ok:
        return True

    sensor_info = sim_tool.getSensorInfo()
    if not sensor_info:
        return True

    return bool(sensor_info[0][0]["sensors"]["state"]["collision"]["has_collided"])


def _edge_points(a: WorldPoint3D, b: WorldPoint3D, sample_step: float) -> List[WorldPoint3D]:
    dist = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    n = max(1, int(math.ceil(dist / max(sample_step, 1e-6))))
    pts: List[WorldPoint3D] = []
    for i in range(n + 1):
        t = i / n
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        z = a[2] + t * (b[2] - a[2])
        pts.append((float(x), float(y), float(z)))
    return pts


def goal_reached(state: StateKey, goal: WorldPoint3D, cfg: PlannerConfig) -> bool:
    x, y, z = state_to_world(state, cfg)
    dxy = math.hypot(x - goal[0], y - goal[1])
    dz = abs(z - goal[2])
    return dxy <= cfg.goal_tolerance_xy and dz <= cfg.goal_tolerance_z


def heuristic(state: StateKey, goal: WorldPoint3D, cfg: PlannerConfig) -> float:
    x, y, z = state_to_world(state, cfg)
    return math.sqrt((x - goal[0]) ** 2 + (y - goal[1]) ** 2 + (z - goal[2]) ** 2)


def expand_actions(state: StateKey, cfg: PlannerConfig) -> List[Tuple[StateKey, Action, float]]:
    x_idx, y_idx, z_idx, yaw_bin = state
    yaw_deg = yaw_deg_from_bin(yaw_bin, cfg.yaw_step_deg)

    out: List[Tuple[StateKey, Action, float]] = []

    def _move_xy(move_heading_deg: float, action_name: str) -> None:
        rad = math.radians(move_heading_deg)
        x = idx_to_world(x_idx, cfg.state_xy_resolution) + cfg.horizontal_step * math.cos(rad)
        y = idx_to_world(y_idx, cfg.state_xy_resolution) + cfg.horizontal_step * math.sin(rad)
        nx = world_to_idx(x, cfg.state_xy_resolution)
        ny = world_to_idx(y, cfg.state_xy_resolution)
        out.append(((nx, ny, z_idx, yaw_bin), (action_name, cfg.horizontal_step), cfg.horizontal_step))

    _move_xy(yaw_deg, "forward")
    _move_xy(yaw_deg - 90.0, "left")
    _move_xy(yaw_deg + 90.0, "right")

    out.append(((x_idx, y_idx, world_to_idx(idx_to_world(z_idx, cfg.state_z_resolution) - cfg.vertical_step, cfg.state_z_resolution), yaw_bin), ("ascend", cfg.vertical_step), cfg.vertical_step))
    out.append(((x_idx, y_idx, world_to_idx(idx_to_world(z_idx, cfg.state_z_resolution) + cfg.vertical_step, cfg.state_z_resolution), yaw_bin), ("descend", cfg.vertical_step), cfg.vertical_step))

    num_bins = int(round(360.0 / cfg.yaw_step_deg))
    out.append(((x_idx, y_idx, z_idx, (yaw_bin - 1) % num_bins), ("rotl", cfg.yaw_step_deg), 0.5))
    out.append(((x_idx, y_idx, z_idx, (yaw_bin + 1) % num_bins), ("rotr", cfg.yaw_step_deg), 0.5))

    return out


def plan_action_consistent_astar(
    sim_tool: AirVLNSimulatorClientTool,
    episode: Episode,
    cfg: PlannerConfig,
) -> Tuple[List[StateKey], List[Action]]:
    start_yaw_deg = quaternion_to_yaw_deg(episode.start_quaternionr)
    start_state: StateKey = (
        world_to_idx(episode.start_position[0], cfg.state_xy_resolution),
        world_to_idx(episode.start_position[1], cfg.state_xy_resolution),
        world_to_idx(episode.start_position[2], cfg.state_z_resolution),
        yaw_bin_from_deg(start_yaw_deg, cfg.yaw_step_deg),
    )

    goal = episode.goal_position

    x_min = world_to_idx(min(episode.start_position[0], goal[0]) - cfg.search_margin_xy_m, cfg.state_xy_resolution)
    x_max = world_to_idx(max(episode.start_position[0], goal[0]) + cfg.search_margin_xy_m, cfg.state_xy_resolution)
    y_min = world_to_idx(min(episode.start_position[1], goal[1]) - cfg.search_margin_xy_m, cfg.state_xy_resolution)
    y_max = world_to_idx(max(episode.start_position[1], goal[1]) + cfg.search_margin_xy_m, cfg.state_xy_resolution)
    z_min = world_to_idx(min(episode.start_position[2], goal[2]) - cfg.search_margin_z_m, cfg.state_z_resolution)
    z_max = world_to_idx(max(episode.start_position[2], goal[2]) + cfg.search_margin_z_m, cfg.state_z_resolution)

    def in_bounds(s: StateKey) -> bool:
        x, y, z, _ = s
        return x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max

    node_free_cache: Dict[Tuple[int, int, int], bool] = {}
    edge_free_cache: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], bool] = {}

    def pos_key(s: StateKey) -> Tuple[int, int, int]:
        return s[0], s[1], s[2]

    def is_node_free(s: StateKey) -> bool:
        key = pos_key(s)
        if key in node_free_cache:
            return node_free_cache[key]
        p = state_to_world(s, cfg)
        y = yaw_from_state(s, cfg)
        free = not _probe_collision(sim_tool, p, y)
        node_free_cache[key] = free
        return free

    def is_edge_free(s1: StateKey, s2: StateKey) -> bool:
        k1, k2 = pos_key(s1), pos_key(s2)
        if k1 <= k2:
            key = (k1, k2)
        else:
            key = (k2, k1)
        if key in edge_free_cache:
            return edge_free_cache[key]

        p1 = state_to_world(s1, cfg)
        p2 = state_to_world(s2, cfg)
        yaw = yaw_from_state(s2, cfg)
        ok = True
        for p in _edge_points(p1, p2, cfg.edge_check_step):
            if _probe_collision(sim_tool, p, yaw):
                ok = False
                break
        edge_free_cache[key] = ok
        return ok

    if not is_node_free(start_state):
        raise RuntimeError("Start position is in collision.")

    open_heap: List[Tuple[float, StateKey]] = [(heuristic(start_state, goal, cfg), start_state)]
    came_from: Dict[StateKey, StateKey] = {}
    action_from: Dict[StateKey, Action] = {}
    g_cost: Dict[StateKey, float] = {start_state: 0.0}
    visited = set()

    expansions = 0
    goal_state: Optional[StateKey] = None

    while open_heap:
        _, curr = heapq.heappop(open_heap)
        if curr in visited:
            continue
        visited.add(curr)

        expansions += 1
        if expansions > cfg.max_expansions:
            raise RuntimeError("A* exceeded max expansions.")

        if goal_reached(curr, goal, cfg):
            goal_state = curr
            break

        for nxt, action, step_cost in expand_actions(curr, cfg):
            if not in_bounds(nxt):
                continue
            if not is_node_free(nxt):
                continue
            if not is_edge_free(curr, nxt):
                continue

            new_g = g_cost[curr] + step_cost
            if nxt not in g_cost or new_g < g_cost[nxt]:
                g_cost[nxt] = new_g
                came_from[nxt] = curr
                action_from[nxt] = action
                f = new_g + heuristic(nxt, goal, cfg)
                heapq.heappush(open_heap, (f, nxt))

    if goal_state is None:
        raise RuntimeError("Failed to find action-consistent collision-free path.")

    states = [goal_state]
    actions_rev: List[Action] = []
    cur = goal_state
    while cur in came_from:
        actions_rev.append(action_from[cur])
        cur = came_from[cur]
        states.append(cur)

    states.reverse()
    actions_rev.reverse()
    return states, actions_rev


def write_rgb_bytes(path: str, image_bytes: bytes) -> None:
    with open(path, "wb") as f:
        f.write(image_bytes)


def capture_episode_path_images(
    sim_tool: AirVLNSimulatorClientTool,
    episode: Episode,
    states: List[StateKey],
    actions: List[Action],
    cfg: PlannerConfig,
    output_dir: Path,
    cameras: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    poses_for_meta = []

    for idx, state in enumerate(states):
        curr = state_to_world(state, cfg)
        yaw_deg = yaw_from_state(state, cfg)
        yaw_rad = math.radians(yaw_deg)
        q = airsim.to_quaternion(0.0, 0.0, yaw_rad)
        pose = airsim.Pose(
            position_val=airsim.Vector3r(curr[0], curr[1], curr[2]),
            orientation_val=airsim.Quaternionr(q.x_val, q.y_val, q.z_val, q.w_val),
        )

        ok = sim_tool.setPoses([[pose]])
        if not ok:
            raise RuntimeError(f"Failed to set pose at step={idx}.")

        responses = sim_tool.getImageResponses(cameras=list(cameras))
        if responses is None:
            raise RuntimeError(f"Failed to get images at step={idx}.")

        rgb_list, depth_list = responses[0][0]

        step_dir = output_dir / f"step_{idx:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        cam_meta = []
        for cam_idx, cam_name in enumerate(cameras):
            rgb_path = step_dir / f"cam_{cam_name}_rgb.png"
            depth_path = step_dir / f"cam_{cam_name}_depth.png"

            write_rgb_bytes(str(rgb_path), rgb_list[cam_idx])
            cv2.imwrite(str(depth_path), depth_list[cam_idx])

            cam_meta.append(
                {
                    "camera": cam_name,
                    "rgb": str(rgb_path.name),
                    "depth": str(depth_path.name),
                }
            )

        next_action = None
        if idx < len(actions):
            next_action = {"name": actions[idx][0], "value": actions[idx][1]}

        poses_for_meta.append(
            {
                "step": idx,
                "position": [curr[0], curr[1], curr[2]],
                "yaw_deg": yaw_deg,
                "quaternion": [q.x_val, q.y_val, q.z_val, q.w_val],
                "next_action": next_action,
                "cameras": cam_meta,
            }
        )

    meta_path = output_dir / "path_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "episode_id": episode.episode_id,
                "map_name": episode.map_name,
                "object_name": episode.object_name,
                "num_steps": len(states),
                "num_actions": len(actions),
                "planner": "action-consistent-a-star",
                "actions": [{"name": a[0], "value": a[1]} for a in actions],
                "path": poses_for_meta,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def build_machine_info(simulator_port: int, map_name: str, gpu_id: int) -> List[dict]:
    return [
        {
            "MACHINE_IP": "127.0.0.1",
            "SOCKET_PORT": simulator_port,
            "MAX_SCENE_NUM": 1,
            "open_scenes": [map_name],
            "gpus": [gpu_id],
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Action-space-consistent A* path planning + image collection.")
    parser.add_argument("--gt_json", type=str, required=True, help="Ground-truth json (single episode object or list).")
    parser.add_argument("--output_dir", type=str, default="./logs/astar_path_images", help="Output root directory.")
    parser.add_argument("--simulator_port", type=int, default=31000, help="AirVLNSimulatorServerTool socket port.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id passed to server when opening scene.")

    parser.add_argument("--horizontal_step", type=float, default=5.0, help="Forward/left/right step size.")
    parser.add_argument("--vertical_step", type=float, default=2.0, help="Ascend/descend step size.")
    parser.add_argument("--yaw_step_deg", type=float, default=15.0, help="rotl/rotr rotation angle in degrees.")

    parser.add_argument("--state_xy_resolution", type=float, default=1.0, help="State discretization for x/y.")
    parser.add_argument("--state_z_resolution", type=float, default=1.0, help="State discretization for z.")
    parser.add_argument("--search_margin_xy_m", type=float, default=60.0, help="XY search range margin in meters.")
    parser.add_argument("--search_margin_z_m", type=float, default=20.0, help="Z search range margin in meters.")

    parser.add_argument("--edge_check_step", type=float, default=1.0, help="Collision probe step along transitions.")
    parser.add_argument("--max_expansions", type=int, default=50000, help="Maximum A* node expansions.")
    parser.add_argument("--goal_tolerance_xy", type=float, default=5.0, help="Goal XY tolerance in meters.")
    parser.add_argument("--goal_tolerance_z", type=float, default=2.0, help="Goal Z tolerance in meters.")

    parser.add_argument("--cameras", type=str, default="0,1,2,3", help="Comma-separated AirSim camera names.")
    parser.add_argument("--max_episodes", type=int, default=0, help="If >0, only process first N episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    cfg = PlannerConfig(
        horizontal_step=args.horizontal_step,
        vertical_step=args.vertical_step,
        yaw_step_deg=args.yaw_step_deg,
        state_xy_resolution=args.state_xy_resolution,
        state_z_resolution=args.state_z_resolution,
        search_margin_xy_m=args.search_margin_xy_m,
        search_margin_z_m=args.search_margin_z_m,
        edge_check_step=args.edge_check_step,
        max_expansions=args.max_expansions,
        goal_tolerance_xy=args.goal_tolerance_xy,
        goal_tolerance_z=args.goal_tolerance_z,
    )

    episodes = load_episodes(args.gt_json)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    by_map: Dict[str, List[Episode]] = {}
    for e in episodes:
        by_map.setdefault(e.map_name, []).append(e)

    for map_name, eps in by_map.items():
        machines_info = build_machine_info(args.simulator_port, map_name, args.gpu_id)
        sim_tool = AirVLNSimulatorClientTool(machines_info=machines_info)
        print(f"[INFO] Opening scene: {map_name}")
        sim_tool.run_call()

        try:
            for episode in eps:
                print(f"[INFO] episode_id={episode.episode_id}, object={episode.object_name}")
                states, actions = plan_action_consistent_astar(sim_tool=sim_tool, episode=episode, cfg=cfg)

                ep_dir = output_root / map_name / f"episode_{episode.episode_id}"
                capture_episode_path_images(
                    sim_tool=sim_tool,
                    episode=episode,
                    states=states,
                    actions=actions,
                    cfg=cfg,
                    output_dir=ep_dir,
                    cameras=cameras,
                )

                print(f"[INFO] Saved {len(states)} states / {len(actions)} actions to: {ep_dir}")
        finally:
            print(f"[INFO] Closing scene: {map_name}")
            sim_tool.closeScenes()


if __name__ == "__main__":
    main()
