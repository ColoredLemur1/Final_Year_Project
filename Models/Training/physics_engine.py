"""
Physics engine: batted-ball trajectory from launch speed, angle, and spray direction.

Vacuum: closed form. With drag: numerical integration using air density (altitude),
velocity-dependent drag coefficient, and optional Magnus lift from spin (rpm).
"""

from __future__ import annotations

import math
from typing import Any

# English engineering units: ft, s, slug, lbf (F = m*a with m in slug, a in ft/s^2 -> lbf)
G = 32.2  # ft/s^2
MPH_TO_FPS = 1.467
DEG_TO_RAD = math.pi / 180.0
TWO_PI = 2.0 * math.pi

# Standard-day sea-level air density (slug/ft^3)
RHO_SLUG_FT3 = 0.0023769

# MLB ball ~ 5.125 oz mass; diameter ~ 2.9 in
BALL_DIAMETER_FT = (2.9 / 12.0)
BALL_RADIUS_FT = 0.5 * BALL_DIAMETER_FT
BALL_AREA_FT2 = math.pi * BALL_RADIUS_FT * BALL_RADIUS_FT
# mass in slug = lbm / g_c
BALL_MASS_SLUG = (5.125 / 16.0) / 32.174  # slug (use standard g for mass from oz)

# Approximate home-plate elevation (ft) for air density (Coors is the main outlier)
_STADIUM_ELEV_FT: dict[str, float] = {
    "COL": 5190.0,
    "ARI": 1117.0,
    "ATL": 1050.0,
    "TEX": 549.0,
    "MIL": 597.0,
    "MIN": 815.0,
    "STL": 465.0,
    "LAD": 515.0,
    "SDP": 23.0,
    "SFG": 8.0,
    "SEA": 17.0,
    "TOR": 270.0,
    "PHI": 40.0,
    "NYY": 55.0,
    "NYM": 55.0,
    "CHC": 600.0,
    "CWS": 600.0,
    "DET": 583.0,
    "CIN": 535.0,
    "PIT": 724.0,
    "DEN": 5190.0,
    "HOU": 50.0,
    "MIA": 10.0,
    "TB": 40.0,
    "BOS": 20.0,
    "WSN": 35.0,
    "BAL": 33.0,
    "OAK": 200.0,
    "KC": 886.0,
}


def stadium_elevation_ft(home_team: str | None) -> float:
    """Return nominal park elevation (ft) for a home_team abbreviation; default mid-pack."""
    if not home_team:
        return 500.0
    key = str(home_team).strip().upper()
    if not key or key == "__NA__":
        return 500.0
    return float(_STADIUM_ELEV_FT.get(key, 500.0))


def air_density_ratio(altitude_ft: float) -> float:
    """ISA troposphere: rho(h)/rho0 vs geopotential altitude (ft)."""
    h_m = max(0.0, float(altitude_ft)) * 0.3048
    L = 0.0065  # K/m
    T0 = 288.15  # K
    inner = 1.0 - (L * h_m) / T0
    if inner <= 0.25:
        inner = 0.25
    return float(inner ** 4.255)


def drag_coefficient_mph(v_mph: float) -> float:
    """Weak velocity dependence (drag crisis): higher Cd at lower speeds, bounded."""
    v = float(v_mph)
    # Piecewise linear between anchor points (Nathan-style ballpark)
    cd = 0.50 - 0.0012 * (v - 65.0)
    return float(max(0.28, min(0.52, cd)))


def lift_coefficient_spin(S: float) -> float:
    """Lift coefficient vs spin parameter S = R*|omega x v_hat|/|v| (order ~0.05–0.35)."""
    s = max(0.0, float(S))
    if s < 0.10:
        return 0.12 + 1.15 * s
    cl = 0.235 + 0.95 * s - 0.40 * s * s
    return float(max(0.0, min(0.78, cl)))


def _estimate_batted_spin_rpm(launch_speed_mph: float, launch_angle_deg: float) -> float:
    """Heuristic backspin (rpm) when pitch spin is unknown."""
    ev = float(launch_speed_mph)
    la = float(launch_angle_deg)
    rpm = 650.0 + 18.0 * ev + 14.0 * max(0.0, la)
    return float(max(600.0, min(3400.0, rpm)))


def _effective_spin_rpm(
    launch_speed_mph: float,
    launch_angle_deg: float,
    pitch_spin_rpm: float | None,
) -> float:
    """Map pitch rpm (if given) toward plausible batted-ball backspin; else EV/LA heuristic."""
    if pitch_spin_rpm is None or not math.isfinite(pitch_spin_rpm) or pitch_spin_rpm <= 0:
        return _estimate_batted_spin_rpm(launch_speed_mph, launch_angle_deg)
    ps = float(pitch_spin_rpm)
    base = _estimate_batted_spin_rpm(launch_speed_mph, launch_angle_deg)
    blended = 0.45 * ps + 0.55 * base + 8.0 * max(0.0, float(launch_angle_deg))
    return float(max(600.0, min(3600.0, blended)))


def _legacy_drag_step(
    vx_i: float,
    vy_i: float,
    vz_i: float,
    drag_coeff: float,
    dt: float,
) -> tuple[float, float, float, float, float, float]:
    """Previous k*|v| drag model (dimensionally tuned)."""
    speed = math.sqrt(vx_i * vx_i + vy_i * vy_i + vz_i * vz_i)
    if speed < 1e-6:
        return vx_i, vy_i, vz_i, 0.0, 0.0, -G
    k = drag_coeff * 0.001
    ax = -k * vx_i * speed
    ay = -k * vy_i * speed
    az = -G - k * vz_i * speed
    vx_i += ax * dt
    vy_i += ay * dt
    vz_i += az * dt
    return vx_i, vy_i, vz_i, ax, ay, az


def _magnus_acceleration_ft_s2(
    vx: float,
    vy: float,
    vz: float,
    spin_rpm: float,
    rho: float,
    use_magnus: bool,
) -> tuple[float, float, float]:
    if not use_magnus or spin_rpm <= 0.0:
        return 0.0, 0.0, 0.0
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    if speed < 1e-4:
        return 0.0, 0.0, 0.0
    vxy = math.hypot(vx, vy)
    if vxy < 1e-5:
        return 0.0, 0.0, 0.0
    # Backspin axis: horizontal, perpendicular to horizontal velocity (+x for spray toward +y)
    ox = vy / vxy
    oy = -vx / vxy
    oz = 0.0
    omega_rad_s = spin_rpm * TWO_PI / 60.0
    # omega vector along spin axis (ft/s as vector direction; magnitude in rad/s)
    wx, wy, wz = omega_rad_s * ox, omega_rad_s * oy, omega_rad_s * oz
    cx = wy * vz - wz * vy
    cy = wz * vx - wx * vz
    cz = wx * vy - wy * vx
    cross_mag = math.sqrt(cx * cx + cy * cy + cz * cz)
    if cross_mag < 1e-10:
        return 0.0, 0.0, 0.0
    ux, uy, uz = cx / cross_mag, cy / cross_mag, cz / cross_mag
    sin_t = cross_mag / (omega_rad_s * speed + 1e-12)
    sin_t = max(0.0, min(1.0, sin_t))
    S = BALL_RADIUS_FT * omega_rad_s * sin_t / speed
    Cl = lift_coefficient_spin(S)
    q = 0.5 * rho * BALL_AREA_FT2 * Cl * speed * speed
    am = q / BALL_MASS_SLUG
    return am * ux, am * uy, am * uz


def _drag_acceleration_ft_s2(vx: float, vy: float, vz: float, rho: float) -> tuple[float, float, float]:
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    if speed < 1e-6:
        return 0.0, 0.0, 0.0
    v_mph = speed / MPH_TO_FPS
    Cd = drag_coefficient_mph(v_mph)
    q = 0.5 * rho * BALL_AREA_FT2 * Cd * speed * speed
    ad = q / (BALL_MASS_SLUG * speed)
    return -ad * vx, -ad * vy, -ad * vz


def compute_trajectory(
    launch_speed_mph: float,
    launch_angle_deg: float,
    spray_angle_deg: float,
    use_drag: bool = False,
    drag_coeff: float = 0.35,
    dt: float = 0.005,
    *,
    spin_rate_rpm: float | None = None,
    pitch_spin_rpm: float | None = None,
    altitude_ft: float | None = None,
    home_team: str | None = None,
    use_magnus: bool = True,
    use_legacy_drag: bool = False,
) -> dict[str, Any]:
    """
    Batted-ball flight: max height, time of air, ground distance (ft from home along spray).

    use_drag=False: vacuum closed form (unchanged).
    use_drag=True: integrate with air rho(altitude), Cd(|v|), optional Magnus from spin_rpm.
    spin_rate_rpm: batted-ball backspin scale (rpm); if None, uses pitch_spin_rpm then heuristic.
    altitude_ft: if None, uses stadium_elevation_ft(home_team).
    use_legacy_drag=True: old constant drag_coeff k*|v| model (ignores Magnus and rho).
    """
    v0_fps = float(launch_speed_mph) * MPH_TO_FPS
    theta = float(launch_angle_deg) * DEG_TO_RAD
    phi = float(spray_angle_deg) * DEG_TO_RAD

    vx0 = v0_fps * math.cos(theta) * math.sin(phi)
    vy0 = v0_fps * math.cos(theta) * math.cos(phi)
    vz0 = v0_fps * math.sin(theta)

    if not use_drag:
        h_max = (vz0 * vz0) / (2.0 * G)
        t_flight = 2.0 * vz0 / G if vz0 > 0 else 0.0
        distance_ft = vy0 * t_flight
        return {
            "max_height_ft": h_max,
            "time_of_flight_s": t_flight,
            "distance_ft": distance_ft,
            "vx": vx0,
            "vy": vy0,
            "vz": vz0,
        }

    if use_legacy_drag:
        x, y, z = 0.0, 0.0, 0.0
        vx_i, vy_i, vz_i = vx0, vy0, vz0
        h_max = 0.0
        t = 0.0
        while z >= 0.0 or t == 0.0:
            vx_i, vy_i, vz_i, ax, ay, az = _legacy_drag_step(vx_i, vy_i, vz_i, drag_coeff, dt)
            x += vx_i * dt
            y += vy_i * dt
            z += vz_i * dt
            t += dt
            if z > h_max:
                h_max = z
            if t > 35.0:
                break
        distance_ft = math.hypot(x, y)
        return {
            "max_height_ft": h_max,
            "time_of_flight_s": t,
            "distance_ft": distance_ft,
            "vx": vx0,
            "vy": vy0,
            "vz": vz0,
        }

    alt = float(altitude_ft) if altitude_ft is not None else stadium_elevation_ft(home_team)
    rho = RHO_SLUG_FT3 * air_density_ratio(alt)
    spin = (
        float(spin_rate_rpm)
        if spin_rate_rpm is not None and math.isfinite(spin_rate_rpm) and spin_rate_rpm > 0
        else _effective_spin_rpm(launch_speed_mph, launch_angle_deg, pitch_spin_rpm)
    )

    x, y, z = 0.0, 0.0, 0.0
    xp, yp, zp = 0.0, 0.0, 0.0
    vx_i, vy_i, vz_i = vx0, vy0, vz0
    h_max = 0.0
    t = 0.0

    while (z >= 0.0 or t == 0.0) and t < 40.0:
        xp, yp, zp = x, y, z
        ddx, ddy, ddz = _drag_acceleration_ft_s2(vx_i, vy_i, vz_i, rho)
        mx, my, mz = _magnus_acceleration_ft_s2(vx_i, vy_i, vz_i, spin, rho, use_magnus)
        ax = ddx + mx
        ay = ddy + my
        az = ddz + mz - G
        vx_i += ax * dt
        vy_i += ay * dt
        vz_i += az * dt
        x += vx_i * dt
        y += vy_i * dt
        z += vz_i * dt
        t += dt
        if z > h_max:
            h_max = z
        if z < 0.0 and zp >= 0.0 and t > dt:
            frac = zp / (zp - z + 1e-12)
            frac = max(0.0, min(1.0, frac))
            x = xp + frac * (x - xp)
            y = yp + frac * (y - yp)
            z = 0.0
            t = t - dt + frac * dt
            break

    distance_ft = math.hypot(x, y)
    return {
        "max_height_ft": h_max,
        "time_of_flight_s": t,
        "distance_ft": distance_ft,
        "vx": vx0,
        "vy": vy0,
        "vz": vz0,
    }
