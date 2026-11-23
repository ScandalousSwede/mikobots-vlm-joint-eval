# MiKo joint evaluation

Which physical joint moved when the controller addressed firmware joint J2? In the retained March 2026 case, the image comparison selected physical J3. Three joints with known identities anchored the comparison. The model inferred the other three from simulator renders and camera strips of the arm.

## Inspect the saved result

From a clone or downloaded copy, run the Python 3 standard-library reader. It opens the [original JSON](calibration_results/joint_map_final.json) relative to the script. This path works offline with no model account, camera, robot or Studio installation.

```console
python3 inspect_mapping.py
```

| Firmware joint | Physical joint | Basis |
| --- | --- | --- |
| J1 | J2 | Supplied anchor |
| J2 | J3 | Model inference |
| J3 | J1 | Supplied anchor |
| J4 | J4 | Model inference |
| J5 | J5 | Model inference |
| J6 | J6 | Supplied anchor |

The March 16 record names GPT-4o and reports a match for each comparison. J1, J3 and J6 were supplied to the prompt from prior inspection. The response scores of 0.90–0.95 are model-reported confidence. Independent physical validation of the three inferred assignments would require separate observations.

## How the original comparison ran

Both historical scripts take their paths from constants in the source; they have no command-line path flags. The repository includes the result and scripts. A fresh comparison additionally requires the matching Studio asset tree, meshes, camera strips and a configured OpenAI account.

| Step | Source and inputs | Output |
| --- | --- | --- |
| Render | [sim_capture.py](sim_capture.py) uses NumPy, VTK and Pillow. It reads the MiKo Studio asset tree at `MiKoBots_Studio/assets/MiKoBots_data/MiKoBots`, including `Robot_library/MiKo/settings.json` and its meshes. | Per-joint images under `calibration_results/sim_strips/`. |
| Compare | [vlm_sim_compare.py](vlm_sim_compare.py) uses the OpenAI SDK and python-dotenv. For each firmware joint it reads `sim_strips/J{n}_highlight.png` and `J{n}_strip.jpg` under `calibration_results/`. | The script writes `calibration_results/joint_map_final.json`. Use a separate working copy to preserve the retained result. |

The public snapshot does not contain the Studio tree or the original camera strips. The offline reader is the reproducible entry point for this case.
