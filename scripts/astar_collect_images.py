import argparse
import heapq
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import airsim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool


GridCoord = Tuple[int, int]
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


def heuristic(a: GridCoord, b: GridCoord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def neighbors_8(c: GridCoord) -> Iterable[Tuple[GridCoord, float]]:
    x, y = c
    steps = [
        ((x + 1, y), 1.0),
        ((x - 1, y), 1.0),
        ((x, y + 1), 1.0),
        ((x, y - 1), 1.0),
        ((x + 1, y + 1), math.sqrt(2.0)),
        ((x + 1, y - 1), math.sqrt(2.0)),
        ((x - 1, y + 1), math.sqrt(2.0)),
        ((x - 1, y - 1), math.sqrt(2.0)),
    ]
    return steps


def astar(
    start: GridCoord,
    goal: GridCoord,
    blocked: Optional[set],
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> List[GridCoord]:
    if blocked is None:
        blocked = set()

    open_heap: List[Tuple[float, GridCoord]] = [(0.0, start)]
    came_from: Dict[GridCoord, GridCoord] = {}
    g_cost: Dict[GridCoord, float] = {start: 0.0}
    visited = set()

    while open_heap:
        _, curr = heapq.heappop(open_heap)
        if curr in visited:
            continue
        visited.add(curr)

        if curr == goal:
            path = [curr]
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            path.reverse()
            return path

        for nxt, move_cost in neighbors_8(curr):
            nx, ny = nxt
            if nx < x_min or nx > x_max or ny < y_min or ny > y_max:
                continue
            if nxt in blocked:
                continue

            new_g = g_cost[curr] + move_cost
            if nxt not in g_cost or new_g < g_cost[nxt]:
                g_cost[nxt] = new_g
                f = new_g + heuristic(nxt, goal)
                came_from[nxt] = curr
                heapq.heappush(open_heap, (f, nxt))

    raise RuntimeError("A* failed to find a path. Try larger --search_margin.")


def world_to_grid(x: float, y: float, origin_xy: Tuple[float, float], resolution: float) -> GridCoord:
    ox, oy = origin_xy
    gx = int(round((x - ox) / resolution))
    gy = int(round((y - oy) / resolution))
    return gx, gy


def grid_to_world(gx: int, gy: int, origin_xy: Tuple[float, float], resolution: float) -> Tuple[float, float]:
    ox, oy = origin_xy
    x = ox + gx * resolution
    y = oy + gy * resolution
    return x, y


def build_world_path(
    start: WorldPoint3D,
    goal: WorldPoint3D,
    resolution: float,
    search_margin: int,
    keep_z_constant: bool,
) -> List[WorldPoint3D]:
    min_x = min(start[0], goal[0]) - search_margin * resolution
    min_y = min(start[1], goal[1]) - search_margin * resolution

    origin_xy = (min_x, min_y)

    s = world_to_grid(start[0], start[1], origin_xy, resolution)
    g = world_to_grid(goal[0], goal[1], origin_xy, resolution)

    x_min = min(s[0], g[0]) - search_margin
    x_max = max(s[0], g[0]) + search_margin
    y_min = min(s[1], g[1]) - search_margin
    y_max = max(s[1], g[1]) + search_margin

    grid_path = astar(s, g, blocked=None, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    world_path: List[WorldPoint3D] = []
    total = max(1, len(grid_path) - 1)
    for i, (gx, gy) in enumerate(grid_path):
        x, y = grid_to_world(gx, gy, origin_xy, resolution)
        if keep_z_constant:
            z = float(start[2])
        else:
            alpha = i / total
            z = float(start[2] + alpha * (goal[2] - start[2]))
        world_path.append((float(x), float(y), float(z)))

    return world_path


def yaw_from_points(curr: WorldPoint3D, nxt: Optional[WorldPoint3D]) -> float:
    if nxt is None:
        return 0.0
    dx = nxt[0] - curr[0]
    dy = nxt[1] - curr[1]
    return math.atan2(dy, dx)


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
    parser = argparse.ArgumentParser(description="Plan A* path and capture RGB/Depth images along the whole path.")
    parser.add_argument("--gt_json", type=str, required=True, help="Ground-truth json (single episode object or list).")
    parser.add_argument("--output_dir", type=str, default="./logs/astar_path_images", help="Output root directory.")
    parser.add_argument("--simulator_port", type=int, default=31000, help="AirVLNSimulatorServerTool socket port.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id passed to server when opening scene.")
    parser.add_argument("--resolution", type=float, default=2.0, help="A* grid resolution in world units.")
    parser.add_argument("--search_margin", type=int, default=20, help="Extra grid cells around start-goal bounding box.")
    parser.add_argument("--keep_z_constant", action="store_true", help="Keep z fixed at start z across path.")
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
                world_path = build_world_path(
                    start=episode.start_position,
                    goal=episode.goal_position,
                    resolution=args.resolution,
                    search_margin=args.search_margin,
                    keep_z_constant=args.keep_z_constant,
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
