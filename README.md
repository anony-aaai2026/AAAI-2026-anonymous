# AAAI-2026-anonymous

This repository contains code to reproduce the results in our AAAI 2026 submission.

## Requirements

- Python 3.11.7
- Required packages (install with pip):

``` bash
pip install -r requirements.txt
```
## Running the code
- Toy example:
```
python toy_example.py
```

- Real world data:
```
python real_world_data.py
```

This code uses the CASP protein structure dataset.

- **Filename expected:** `dataset/CASP.csv`
- **Source:** [CASP dataset from UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure)
- **License:** Open-access (UCI datasets are generally free for academic use)
- **Format:** CSV file with physicochemical features for protein tertiary structure prediction

To run the code successfully, please download the file manually from the UCI repository and place it in the following directory:
```
dataset/CASP.csv
```
You may need to create the `dataset/` folder first.



## Notes
- All identifying information has been removed for double-blind review.

- This code is intended for evaluation purposes during peer review.

