"""
Physics engine: batted ball trajectory from launch speed, launch angle, spray angle.

Converts launch parameters to 3D velocity, computes max height, time of flight,
and horizontal distance. Optional drag for realistic distance (reduces carry by ~20–30% vs vacuum).
"""

from __future__ import annotations

import math
from typing import Any


# ft/s^2
G = 32.2
# mph -> ft/s
MPH_TO_FPS = 1.467
DEG_TO_RAD = math.pi / 180.0


def compute_trajectory(
    launch_speed_mph: float,
    launch_angle_deg: float,
    spray_angle_deg: float,
    use_drag: bool = False,
    drag_coeff: float = 0.35,
    dt: float = 0.01,
) -> dict[str, Any]:
    """
    Compute batted ball flight from launch parameters.

    Args:
        launch_speed_mph: Exit velocity in mph.
        launch_angle_deg: Vertical launch angle in degrees (positive = up).
        spray_angle_deg: Horizontal angle in degrees (e.g. negative = left, positive = right).
        use_drag: If True, integrate with drag for ~20-30% shorter distance.
        drag_coeff: Drag coefficient when use_drag=True (reduces velocity over time).
        dt: Time step in seconds for drag integration.

    Returns:
        Dict with max_height_ft, time_of_flight_s, distance_ft, and optionally vx, vy, vz (ft/s).
    """
    v0_fps = launch_speed_mph * MPH_TO_FPS
    theta = launch_angle_deg * DEG_TO_RAD
    phi = spray_angle_deg * DEG_TO_RAD

    vx = v0_fps * math.cos(theta) * math.sin(phi)
    vy = v0_fps * math.cos(theta) * math.cos(phi)
    vz = v0_fps * math.sin(theta)

    if not use_drag:
        # Vacuum: closed-form
        h_max = (vz * vz) / (2 * G)
        t_flight = 2 * vz / G
        distance_ft = vy * t_flight
        return {
            "max_height_ft": h_max,
            "time_of_flight_s": t_flight,
            "distance_ft": distance_ft,
            "vx": vx,
            "vy": vy,
            "vz": vz,
        }

    # With drag: integrate until z <= 0
    x, y, z = 0.0, 0.0, 0.0
    vx_i, vy_i, vz_i = vx, vy, vz
    h_max = 0.0
    t = 0.0
    while z >= 0 or t == 0:
        speed = math.sqrt(vx_i * vx_i + vy_i * vy_i + vz_i * vz_i)
        if speed < 1e-6:
            break
        # Drag force proportional to v^2; F_drag = -k * v * |v| -> a = -k * v * speed
        k = drag_coeff * 0.001
        ax = -k * vx_i * speed
        ay = -k * vy_i * speed
        az = -G - k * vz_i * speed
        vx_i += ax * dt
        vy_i += ay * dt
        vz_i += az * dt
        x += vx_i * dt
        y += vy_i * dt
        z += vz_i * dt
        t += dt
        if z > h_max:
            h_max = z
        if t > 30.0:
            break
    distance_ft = math.sqrt(x * x + y * y)
    return {
        "max_height_ft": h_max,
        "time_of_flight_s": t,
        "distance_ft": distance_ft,
        "vx": vx,
        "vy": vy,
        "vz": vz,
    }
