#!/usr/bin/env python3
"""Import the resolved CAD URDF into a reusable Isaac Sim USD asset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    root = project_root()
    asset = Path(config["robot"]["usd_asset"])
    config["robot"]["usd_asset"] = asset if asset.is_absolute() else root / asset
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root() / "isaac_sim/config/default.json")
    parser.add_argument("--headless", action="store_true", help="Run without the Isaac Sim UI.")
    args = parser.parse_args()
    config = load_config(args.config)
    resolved_urdf = project_root() / "isaac_sim/assets/linefollowingrobot_resolved.urdf"
    subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_urdf.py")), "--config", str(args.config), "--output", str(resolved_urdf)], check=True)

    # Isaac Sim modules must be imported only after SimulationApp starts.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})
    import omni.kit.commands
    from pxr import PhysxSchema, Usd, UsdPhysics

    destination = Path(config["robot"]["usd_asset"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Isaac Sim could not create the URDF import configuration.")
    import_config.merge_fixed_joints = False
    import_config.fix_base = False
    import_config.make_default_prim = True
    import_config.create_physics_scene = False
    if hasattr(import_config, "collision_from_visuals"):
        import_config.collision_from_visuals = True
    if hasattr(import_config, "convex_decomp"):
        import_config.convex_decomp = False

    imported, result_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(resolved_urdf),
        import_config=import_config,
        dest_path=str(destination),
    )
    if not imported or not destination.is_file():
        raise RuntimeError(f"URDF import failed: {result_path}")

    stage = Usd.Stage.Open(str(destination))
    links = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    if len(links) < 5 or len(joints) < 5:
        raise RuntimeError(f"Imported asset is incomplete: links={len(links)}, joints={len(joints)}")

    target_total_kg = config["robot"].get("total_mass_kg")
    if target_total_kg is not None:
        imported_masses: list[float] = []
        for link in links:
            mass = UsdPhysics.MassAPI(link).GetMassAttr().Get()
            if mass is None:
                raise RuntimeError(f"Imported rigid body {link.GetPath()} has no authored mass.")
            imported_masses.append(float(mass))
        imported_total_kg = sum(imported_masses)
        if abs(imported_total_kg - float(target_total_kg)) > 1.0e-6:
            raise RuntimeError(
                f"Imported mass {imported_total_kg:.9g} kg does not match "
                f"robot.total_mass_kg={float(target_total_kg):.9g} kg."
            )

    expected_wheels = set(config["robot"]["wheel_joints"])
    wheel_joints = [joint for joint in joints if joint.GetName() in expected_wheels]
    if {joint.GetName() for joint in wheel_joints} != expected_wheels:
        discovered = sorted(joint.GetName() for joint in joints)
        raise RuntimeError(f"Wheel joints are missing. Expected {sorted(expected_wheels)}, found {discovered}")
    drive_cfg = config["robot"]["wheel_drive"]
    motor_cfg = config["motor"]
    for joint in wheel_joints:
        # Nominal DC-motor line: damping is the constant stall/no-load slope, so
        # the drive reproduces `tau = Kt*(V - Ke*w)/R` and duty maps to the
        # no-load target speed.  run_line_following.py overrides these per
        # episode for domain randomization, so a re-import is only needed when
        # these *defaults* change.
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(float(drive_cfg["damping"]))
        drive.CreateMaxForceAttr().Set(float(drive_cfg["max_force_nm"]))
        drive.CreateTargetVelocityAttr().Set(0.0)
        joint_api = PhysxSchema.PhysxJointAPI.Apply(joint)
        joint_api.CreateMaxJointVelocityAttr().Set(float(drive_cfg["max_velocity_radps"]) * 180.0 / 3.141592653589793)
        # SolidWorks exports 0.1 N.m of joint friction, several times the whole
        # stall torque of a small gearmotor.  Keep the asset physical.
        joint_api.CreateJointFrictionAttr().Set(float(motor_cfg["coulomb_friction_nm"]))
    stage.GetRootLayer().Save()
    mass_suffix = "" if target_total_kg is None else f", total_mass_kg={float(target_total_kg):g}"
    print(
        f"Imported {destination} (links={len(links)}, joints={len(joints)}, "
        f"wheel_dofs={len(wheel_joints)}{mass_suffix})"
    )
    app.close()


if __name__ == "__main__":
    main()
