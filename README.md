# BOE band-edge and total-energy prediction

Train neural networks to predict the valence-band maximum (VBM), conduction-band
minimum (CBM), and total energy from BOE structural descriptors. Apply a trained
model to an XYZ trajectory to obtain band-edge and band-gap evolution.

The model uses two BOE modes. Each subnet outputs three values per BOE; summing
over BOEs and averaging the two subnet predictions gives structure-level VBM,
CBM, and energy. The band gap is computed as `CBM - VBM`.

## Choose a workflow

- **Use a trained model:** install dependencies, obtain a compatible model bundle,
  and follow [Trajectory prediction](#trajectory-prediction).
- **Train or reproduce the workflow:** start with [Prepare input data](#prepare-input-data),
  then generate descriptors and datasets and run the v1 trainer.

Training trajectories and validated pretrained weights are not included.
Source code for this related force-field/MD workflow is maintained separately in
[BOE-MLFF](https://github.com/yinywbecky/BOE-MLFF). That repository is currently
private, so the link may not be accessible. The band-edge project runs
independently and does not import code from that repository.

## Scope and installation

The supplied setup targets an 80-atom phosphorus reference structure, with
20 BOEs per mode and 42 input features by default. Force alignment and dataset
grouping assume these counts. General compositions, variable cells, and larger
systems are not supported by this workflow without further validation.

Use Python 3.11 or newer:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Choose a PyTorch build suitable for your CPU or CUDA environment. Dependencies
are listed but not version-locked. Bash/awk is required only for the supplied
force and total-energy extraction scripts.

## Main files

| Stage | Code / resources |
| --- | --- |
| Extract all force and energy frames | [ext_force.sh](ext_force.sh), [ext_energy.sh](ext_energy.sh) |
| Select frames | [plot_hist_v2.py](plot_hist_v2.py) |
| Generate BOE descriptors | [boe_cashe.py](boe_cashe.py), [gen_center_all_data.py](gen_center_all_data.py) |
| Align labels and split datasets | [gen_force_all_data.py](gen_force_all_data.py), [gen_force.py](gen_force.py) |
| Train the selected v1 model | [run_bandgap_v1.py](run_bandgap_v1.py), [model_badgap_v1.py](model_badgap_v1.py) |
| Dataset, normalization, plotting helpers | [run_badgap.py](run_badgap.py) |
| Predict electronic properties from XYZ | [predict_badgap_from_traj_v1.py](predict_badgap_from_traj_v1.py) |
| Reference and force-pair indexing | [POSCAR_REV.vasp](POSCAR_REV.vasp), [data1](data1), [data2](data2) |

`run_badgap.py` is a helper module, not an alternative training entrypoint in
this repository. Historical `badgap` filenames are retained for consistency.
Force columns are preserved in the input format, but the v1 loss trains only
VBM, CBM, and total energy; it does not train a force field.

## Prepare input data

Place the following files in each temperature directory (`100`, `150`, `200`,
and `300`) under the repository root:

```text
100/
    XDATCAR
    all_energy.txt
    forces_only.dat
    vbm.txt
    cbm.txt
```

All files must describe the same complete ordered sequence of ionic steps:

- `XDATCAR`: structures with the reference atom ordering.
- `all_energy.txt`: one total energy in eV per raw frame.
- `forces_only.dat`: 80 rows of XYZ forces in eV/A per raw frame.
- `vbm.txt` and `cbm.txt`: one band-edge energy in eV per raw frame.

From inside a temperature directory containing OUTCAR:

```bash
bash ../ext_force.sh
bash ../ext_energy.sh
```

These scripts extract all force blocks and `energy without entropy` values.
They do not extract band edges. Supply VBM/CBM labels using a method appropriate
to your VASP settings, and record the k-point/spin treatment, occupations, and
energy-reference/alignment convention. Do not treat arbitrary fixed band-row
numbers as a generally valid VBM/CBM extraction method.

Do not independently trim structure, force, energy, or band-edge files. Equal
frame counts alone do not prove correspondence; confirm ordering and restart
boundaries. This repository does not provide the original VASP calculation
settings or licensed pseudopotentials.

## Select frames and build datasets

Run from the repository root:

```powershell
python plot_hist_v2.py
python gen_center_all_data.py
python gen_force_all_data.py
```

Defaults preserve the selected v1 data setup:

| Parameter | Value |
| --- | --- |
| Temperatures | 100, 150, 200, 300 K |
| Sampling window | Raw frames `[6000, 10000)` |
| Samples | 1500, 800, 1500, 1500 respectively (5300 total) |
| Candidate stride / energy bins | 2 / 10 |
| Sampling / splitting seed | 42 |
| Split | 70% train, 15% validation, 15% test, by complete structure |
| Generated data directory | `low_temp_v3_1005` |

`frame_index` is zero-based and indexes the complete raw trajectory, not the
sampling window. Energy, VBM, CBM, structure, and force labels must refer to that
same index. The scripts reject energy/index mismatches, inconsistent series
lengths, and incomplete force blocks. Both modes use the same structure split.

A random trajectory split does not establish independence between nearby
frames. Independent trajectories are recommended for assessing generalization.
Regenerate datasets with these scripts rather than reusing legacy selections
whose index or split conventions differ.

## Train and evaluate

```powershell
python run_bandgap_v1.py --hidden-dim 64 --epochs 80 --device cpu
```

Use `--help` for dataset paths, output directory, loss weights, and other options.
Default loss weights are 1.0 for VBM, 1.0 for CBM, and 0.1 for total energy.
Only the validation split is used for checkpoint selection. The selection
criterion preserves the model's sum of VBM, CBM, and total-energy RMSE in eV.
The test split is evaluated after checkpoint selection.

Results go to `result_bandgap_v1`. The main weight files contain the
validation-composite-best model, matching the reported final evaluation.
Keep these files together as a portable inference bundle:

```text
result_bandgap_v1/
    model1_badgap_v1.pth
    model2_badgap_v1.pth
    feature_normalization.npz
    reference.vasp
    summary_badgap_v1.json
```

Normalization is fitted only on training features/targets. The bundle stores
both feature statistics and target mean offsets; inference does not require
access to the training CSVs. Retain the history, split metadata, and evaluation
outputs separately for reproducibility.

VBM/CBM errors are reported in **meV**, not meV/atom. Total-energy errors are
reported in **meV/atom** for the fixed 80-atom system. Raw metrics remain in eV
in the summary.

## Trajectory prediction

With a bundle from this trainer and an XYZ trajectory:

```powershell
python predict_badgap_from_traj_v1.py --model-dir result_bandgap_v1 --trajectory path\to\traj.xyz --output predictions --start 0 --stride 1 --device cpu
```

The trajectory must have the same atom count, species, and physical atom order
as the bundled reference. XYZ coordinates are interpreted in angstroms using
the **fixed lattice from the reference structure**. Variable-cell trajectories
and changes in atom ordering are not supported. For an MD trajectory, ensure the
force-field and electronic-property descriptor conventions are compatible.

The output CSV contains:

```text
step,vbm_eV,cbm_eV,bandgap_eV,total_energy_eV
```

`step` is the zero-based XYZ frame index, not necessarily the original MD
integration step or physical time. Total energy includes restoration of its
training mean offset. VBM/CBM share the energy reference used in their training
labels; the code does not establish an absolute vacuum reference.

The v1 three-output checkpoints are not interchangeable with older nine-output
force/eigenvalue networks. Use only trusted model files and their matching
normalization and reference resources.

## Model training and reproducibility

The model has been trained on research datasets. This repository provides the
training and trajectory-prediction code; previously trained checkpoints and
their benchmark results are not bundled with this code release.

This packaged version includes changes to raw-frame indexing, train/validation/
test splitting, and the training-to-inference interface. These changes mean that
earlier training results should be associated with their original code version,
data preparation, and checkpoints, rather than automatically attributed to this
version. A full research-data training and evaluation run of this packaged
version has not been performed as part of repository preparation.

Checks performed include 10 regression tests, a one-epoch synthetic-data model
bundle/prediction round trip, correct raw-frame indexing for 5300 real selected
samples, and a 12-frame real-data descriptor/label/split check for both modes.
All model files and generated datasets used in these checks are temporary.
These software checks are separate from the model's research training and do not
replace its scientific evaluation.

Record VASP settings, label extraction conventions, frame/split metadata,
dependency versions, training arguments, and the exact model bundle when
reproducing a result. No software license is selected automatically.
