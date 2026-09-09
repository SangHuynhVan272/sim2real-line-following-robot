#!/usr/bin/env python3
"""Create an import-only URDF with absolute mesh URIs and calibrated mass.

The source CAD package stays unchanged. Isaac Sim's standalone URDF importer
does not resolve ROS ``package://`` mesh URIs without a ROS environment.

SolidWorks' material assignments are not the assembled robot's mass model.
When ``robot.total_mass_kg`` is configured, this import-only copy preserves the
CAD masses of every child link and assigns the remaining mass to ``base_link``.
Its inertia tensor is scaled by the same ratio.  The CAD source stays intact,
and a battery/payload change remains a one-value configuration update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    root = project_root()
    for key in ("source_urdf", "package_root", "usd_asset"):
        value = Path(config["robot"][key])
        config["robot"][key] = value if value.is_absolute() else root / value
    return config


def apply_total_mass(root_element: ET.Element, target_total_kg: float) -> tuple[float, float]:
    """Scale ``base_link`` so all URDF links sum to ``target_total_kg``.

    The exact assembled COM/inertia still requires a balance measurement.  In
    its absence, scaling the CAD base tensor is an explicit approximation and
    is preferable to silently simulating the much lighter CAD material mass.
    """
    if target_total_kg <= 0.0:
        raise ValueError("robot.total_mass_kg must be positive")
    links = root_element.findall("link")
    masses: dict[str, tuple[ET.Element, float]] = {}
    for link in links:
        mass = link.find("./inertial/mass")
        if mass is None or mass.get("value") is None:
            continue
        masses[str(link.get("name"))] = (mass, float(mass.get("value")))
    if "base_link" not in masses:
        raise ValueError("source URDF has no inertial mass for base_link")
    base_mass_element, original_base_kg = masses["base_link"]
    child_total_kg = sum(value for name, (_element, value) in masses.items() if name != "base_link")
    calibrated_base_kg = target_total_kg - child_total_kg
    if calibrated_base_kg <= 0.0:
        raise ValueError(
            f"robot.total_mass_kg={target_total_kg:g} is not above child-link mass "
            f"{child_total_kg:g} kg"
        )
    scale = calibrated_base_kg / original_base_kg
    base_mass_element.set("value", f"{calibrated_base_kg:.12g}")
    base_link = next(link for link in links if link.get("name") == "base_link")
    inertia = base_link.find("./inertial/inertia")
    if inertia is None:
        raise ValueError("source URDF has no inertia tensor for base_link")
    for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
        value = inertia.get(name)
        if value is None:
            raise ValueError(f"base_link inertia is missing {name}")
        inertia.set(name, f"{float(value) * scale:.12g}")
    return calibrated_base_kg, scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root() / "isaac_sim/config/default.json")
    parser.add_argument("--output", type=Path, default=project_root() / "isaac_sim/assets/linefollowingrobot_resolved.urdf")
    args = parser.parse_args()

    config = load_config(args.config)
    source = Path(config["robot"]["source_urdf"])
    package_root = Path(config["robot"]["package_root"])
    if not source.is_file() or not package_root.is_dir():
        raise SystemExit("Source URDF or ROS package directory is missing; check default.json.")

    uri_prefix = "package://linefollowingrobot/"
    resolved_prefix = package_root.resolve().as_uri() + "/"
    content = source.read_text(encoding="utf-8")
    if uri_prefix not in content:
        raise SystemExit(f"Expected mesh URI prefix {uri_prefix!r} was not found in {source}.")
    resolved = content.replace(uri_prefix, resolved_prefix)
    # SolidWorks exported an axis on the fixed camera joint.  A fixed joint
    # has no axis; removing it in this import-only copy avoids a PhysX URDF
    # warning while leaving the original CAD package untouched.
    root_element = ET.fromstring(resolved)
    for joint in root_element.findall("joint[@type='fixed']"):
        axis = joint.find("axis")
        if axis is not None:
            joint.remove(axis)
    target_total_kg = config["robot"].get("total_mass_kg")
    mass_message = ""
    if target_total_kg is not None:
        base_mass_kg, inertia_scale = apply_total_mass(root_element, float(target_total_kg))
        mass_message = (
            f"; total mass={float(target_total_kg):.6g} kg, "
            f"base_link={base_mass_kg:.6g} kg, inertia scale={inertia_scale:.6g}"
        )
    resolved = ET.tostring(root_element, encoding="unicode")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(resolved, encoding="utf-8")
    print(f"Wrote import-only URDF: {args.output}{mass_message}")


if __name__ == "__main__":
    main()
