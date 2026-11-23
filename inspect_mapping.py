"""Print the saved joint mapping and the source of each assignment."""
import json
from pathlib import Path

RESULT = Path(__file__).with_name("calibration_results") / "joint_map_final.json"


def main():
    record = json.loads(RESULT.read_text(encoding="utf-8"))
    anchors = record["known_anchors"]
    mapping = record["final_firmware_to_physical_map"]
    print(f'{record["timestamp"]} | {record["model"]} | retained comparison')
    for firmware, physical in sorted(mapping.items(), key=lambda pair: int(pair[0])):
        basis = "supplied anchor" if firmware in anchors else "model inference"
        print(f"Firmware J{firmware} -> physical J{physical} | {basis}")


if __name__ == "__main__":
    main()
