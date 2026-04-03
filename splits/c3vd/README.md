# C3VD Split Files

Expected files:
- train_files.txt
- val_files.txt
- test_files.txt

Line format (non-triplet):

<folder> <frame_idx> l

Generate these files with:
python generate_c3vd_splits.py --data_path <C3VD_ROOT> --protocol folder_layout

`folder_layout` auto-detects either:
- `train/ val/ test`
- `training/ validation/ testing`

Note: If `splits/c3vd` is not writable on this machine, the generator falls back to
`splits/endovis/c3vd` automatically.
