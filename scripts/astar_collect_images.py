import argparse
import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import airsim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool


GridCoord3D = Tuple[int, int, int]
WorldPoint3D = Tuple[float, float, float]


@dataclass
class Episode:
    episode_id: str
    map_name: str
    start_position: WorldPoint3D
    start_quaternionr: Sequence[float]
    goal_position: WorldPoint3D
    object_name: str


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


def heuristic_3d(a: GridCoord3D, b: GridCoord3D) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def neighbors_26(c: GridCoord3D) -> Iterable[Tuple[GridCoord3D, float]]:
    x, y, z = c
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nxt = (x + dx, y + dy, z + dz)
                move_cost = math.sqrt(dx * dx + dy * dy + dz * dz)
                yield nxt, move_cost


def world_to_grid_3d(
    x: float,
    y: float,
    z: float,
    origin_xyz: Tuple[float, float, float],
    xy_resolution: float,
    z_resolution: float,
) -> GridCoord3D:
    ox, oy, oz = origin_xyz
    gx = int(round((x - ox) / xy_resolution))
    gy = int(round((y - oy) / xy_resolution))
    gz = int(round((z - oz) / z_resolution))
    return gx, gy, gz


def grid_to_world_3d(
    gx: int,
    gy: int,
    gz: int,
    origin_xyz: Tuple[float, float, float],
    xy_resolution: float,
    z_resolution: float,
) -> WorldPoint3D:
    ox, oy, oz = origin_xyz
    x = ox + gx * xy_resolution
    y = oy + gy * xy_resolution
    z = oz + gz * z_resolution
    return float(x), float(y), float(z)


def yaw_from_points(curr: WorldPoint3D, nxt: Optional[WorldPoint3D]) -> float:
    if nxt is None:
        return 0.0
    dx = nxt[0] - curr[0]
    dy = nxt[1] - curr[1]
    return math.atan2(dy, dx)


def _probe_collision(sim_tool: AirVLNSimulatorClientTool, point: WorldPoint3D, yaw: float = 0.0) -> bool:
    q = airsim.to_quaternion(0.0, 0.0, yaw)
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

    has_collided = bool(sensor_info[0][0]["sensors"]["state"]["collision"]["has_collided"])
    return has_collided


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


def astar_3d(
    start: GridCoord3D,
    goal: GridCoord3D,
    in_bounds: Callable[[GridCoord3D], bool],
    is_free: Callable[[GridCoord3D], bool],
    edge_is_free: Callable[[GridCoord3D, GridCoord3D], bool],
    max_expansions: int,
) -> List[GridCoord3D]:
    open_heap: List[Tuple[float, GridCoord3D]] = [(heuristic_3d(start, goal), start)]
    came_from: Dict[GridCoord3D, GridCoord3D] = {}
    g_cost: Dict[GridCoord3D, float] = {start: 0.0}
    visited = set()
    expansions = 0

    while open_heap:
        _, curr = heapq.heappop(open_heap)
        if curr in visited:
            continue

        visited.add(curr)
        expansions += 1

        if expansions > max_expansions:
            raise RuntimeError("A* exceeded max expansions. Increase bounds/resolution or max_expansions.")

        if curr == goal:
            path = [curr]
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            path.reverse()
            return path

        for nxt, move_cost in neighbors_26(curr):
            if not in_bounds(nxt):
                continue
            if not is_free(nxt):
                continue
            if not edge_is_free(curr, nxt):
                continue

            new_g = g_cost[curr] + move_cost
            if nxt not in g_cost or new_g < g_cost[nxt]:
                g_cost[nxt] = new_g
                came_from[nxt] = curr
                f = new_g + heuristic_3d(nxt, goal)
                heapq.heappush(open_heap, (f, nxt))

    raise RuntimeError("A* failed to find a collision-free 3D path.")


def build_world_path_3d(
    sim_tool: AirVLNSimulatorClientTool,
    start: WorldPoint3D,
    goal: WorldPoint3D,
    xy_resolution: float,
    z_resolution: float,
    search_margin_xy: int,
    search_margin_z: int,
    max_expansions: int,
    edge_check_step: float,
) -> List[WorldPoint3D]:
    min_x = min(start[0], goal[0]) - search_margin_xy * xy_resolution
    min_y = min(start[1], goal[1]) - search_margin_xy * xy_resolution
    min_z = min(start[2], goal[2]) - search_margin_z * z_resolution

    origin_xyz = (min_x, min_y, min_z)

    s = world_to_grid_3d(start[0], start[1], start[2], origin_xyz, xy_resolution, z_resolution)
    g = world_to_grid_3d(goal[0], goal[1], goal[2], origin_xyz, xy_resolution, z_resolution)

    x_min = min(s[0], g[0]) - search_margin_xy
    x_max = max(s[0], g[0]) + search_margin_xy
    y_min = min(s[1], g[1]) - search_margin_xy
    y_max = max(s[1], g[1]) + search_margin_xy
    z_min = min(s[2], g[2]) - search_margin_z
    z_max = max(s[2], g[2]) + search_margin_z

    def in_bounds(node: GridCoord3D) -> bool:
        x, y, z = node
        return x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max

    free_cache: Dict[GridCoord3D, bool] = {}
    edge_cache: Dict[Tuple[GridCoord3D, GridCoord3D], bool] = {}

    def node_to_world(node: GridCoord3D) -> WorldPoint3D:
        return grid_to_world_3d(node[0], node[1], node[2], origin_xyz, xy_resolution, z_resolution)

    def is_free(node: GridCoord3D) -> bool:
        if node in free_cache:
            return free_cache[node]
        point = node_to_world(node)
        occupied = _probe_collision(sim_tool, point)
        free_cache[node] = (not occupied)
        return free_cache[node]

    def edge_is_free(a: GridCoord3D, b: GridCoord3D) -> bool:
        key = (a, b) if a <= b else (b, a)
        if key in edge_cache:
            return edge_cache[key]

        pa = node_to_world(a)
        pb = node_to_world(b)
        pts = _edge_points(pa, pb, sample_step=edge_check_step)

        ok = True
        for p in pts:
            if _probe_collision(sim_tool, p):
                ok = False
                break

        edge_cache[key] = ok
        return ok

    if not is_free(s):
        raise RuntimeError("Start position is in collision. Cannot run 3D A*.")
    if not is_free(g):
        raise RuntimeError("Goal position is in collision. Cannot run 3D A*.")

    grid_path = astar_3d(
        start=s,
        goal=g,
        in_bounds=in_bounds,
        is_free=is_free,
        edge_is_free=edge_is_free,
        max_expansions=max_expansions,
    )

    world_path = [node_to_world(n) for n in grid_path]
    return world_path


def write_rgb_bytes(path: str, image_bytes: bytes) -> None:
    with open(path, "wb") as f:
        f.write(image_bytes)


def capture_episode_path_images(
    sim_tool: AirVLNSimulatorClientTool,
    episode: Episode,
    world_path: List[WorldPoint3D],
    output_dir: Path,
    cameras: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    poses_for_meta = []

    for idx, curr in enumerate(world_path):
        nxt = world_path[idx + 1] if idx + 1 < len(world_path) else None
        yaw = yaw_from_points(curr, nxt)
        q = airsim.to_quaternion(0.0, 0.0, yaw)
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

        poses_for_meta.append(
            {
                "step": idx,
                "position": [curr[0], curr[1], curr[2]],
                "yaw_rad": yaw,
                "quaternion": [q.x_val, q.y_val, q.z_val, q.w_val],
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
                "num_steps": len(world_path),
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
    parser = argparse.ArgumentParser(description="Plan collision-free 3D A* path and capture RGB/Depth images.")
    parser.add_argument("--gt_json", type=str, required=True, help="Ground-truth json (single episode object or list).")
    parser.add_argument("--output_dir", type=str, default="./logs/astar_path_images", help="Output root directory.")
    parser.add_argument("--simulator_port", type=int, default=31000, help="AirVLNSimulatorServerTool socket port.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id passed to server when opening scene.")
    parser.add_argument("--xy_resolution", type=float, default=2.0, help="3D A* XY grid resolution.")
    parser.add_argument("--z_resolution", type=float, default=1.0, help="3D A* Z grid resolution.")
    parser.add_argument("--search_margin_xy", type=int, default=20, help="Extra XY grid cells around start-goal.")
    parser.add_argument("--search_margin_z", type=int, default=6, help="Extra Z grid cells around start-goal.")
    parser.add_argument("--edge_check_step", type=float, default=1.0, help="Collision probe step along edges.")
    parser.add_argument("--max_expansions", type=int, default=30000, help="Maximum A* node expansions.")
    parser.add_argument("--cameras", type=str, default="0,1,2,3", help="Comma-separated AirSim camera names.")
    parser.add_argument("--max_episodes", type=int, default=0, help="If >0, only process first N episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

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
                world_path = build_world_path_3d(
                    sim_tool=sim_tool,
                    start=episode.start_position,
                    goal=episode.goal_position,
                    xy_resolution=args.xy_resolution,
                    z_resolution=args.z_resolution,
                    search_margin_xy=args.search_margin_xy,
                    search_margin_z=args.search_margin_z,
                    max_expansions=args.max_expansions,
                    edge_check_step=args.edge_check_step,
                )

                ep_dir = output_root / map_name / f"episode_{episode.episode_id}"
                capture_episode_path_images(
                    sim_tool=sim_tool,
                    episode=episode,
                    world_path=world_path,
                    output_dir=ep_dir,
                    cameras=cameras,
                )

                print(f"[INFO] Saved {len(world_path)} steps to: {ep_dir}")
        finally:
            print(f"[INFO] Closing scene: {map_name}")
            sim_tool.closeScenes()


if __name__ == "__main__":
    main()
