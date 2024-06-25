# Table of Contents
- [Prerequisites](#prerequisites)
  - [Dataset](#dataset)
  - [Environment](#environment)
- [Functionality](#functionality)
  - [main.py](#main.py)
  - [main.ipynb](#main.ipynb)
- [Important Note on Smoothing Algorithm](#important-note-on-smoothing-algorithm)
- [References](#references)

# Prerequisites

The below sections detail the necessary dataset and anaconda environment for running the code.

## Dataset

The dataset is available at https://vision.seas.harvard.edu/qsfs/Data.html (Xiong et al., 2015). Of these files, the images under `frog/Objects` and lighting directions in `frog/light_directions.txt` were used to produce the dissertation. The codebase is also readily compatible with the equivalent files for other models (e.g. `cat` or `scholar`), and other data provided in the same format should also work fine.

This subset is also included in `data.zip`.

### Environment

Python is needed to run the code. To create an anaconda environment with all required packages, the `environment.yml` file is provided. Running `conda env create -f environment.yml` will create an environment `ps-fyp` (this name can be changed within the file). `conda activate ps-fyp` then activates the environment and `main.py` or `main.ipynb` can be run.

# Functionality

All functionality is included in the `psfyp.py` file. This contains:

- Various helpful functions, such as angular error
- A `corrupt_images` function, to add noise and specular highlights to images
  - The function also takes a normal field and lighting directions as input, in order to model specular highlights using the Phong model (Bui-Tuong, 1975)
- A `PSSolver` class for reconstructing normals:
  - The class takes an observation matrix $O^{m\times n}$, for $n$ images containing $m$ pixels each, and a lighting matrix $L^{3\times n}$ of lighting directions
  - Two algorithms are available:
    - Basic Photometric Stereo (Woodham, 1980)
    - SBL Photometric Stereo (Ikehata et al., 2012)
      - The per-pixel solving algorithm is adapted from the MATLAB code of Ikehata et al. (2012), available at https://satoshi-ikehata.github.io/
  
- A `PSIntegrator` for integrating and smoothing normals:
  - Integration is implemented according to Frankot and Chellappa (1988)
  - Integrated surfaces can be exported to .obj files
  - Smoothing is implemented according to Jones, Durand and Zwicker (2004)

Detailed information about individual functions and classes is provided in the appropriate docstrings in `psfyp.py`.




There are two methods provided for running the code:

- The python source file `main.py`
- The Jupyter Notebook `main.ipynb`

### main.py

When run, `main.py` will:

1. Read in a collection of images and lighting directions
2. Run photometric stereo on these images
3. Output the reconstructed normals
   - If SBL PS was used, also output the error variances per image
4. Integrate the normals into a surface, and output this as a .obj file
5. Corrupt images with noise and specular highlights
6. Output the corrupted images
7. Repeat steps 2-4 for the corrupted images
8. Calculate the angular difference between baseline and corrupted normals, and output this
9. Smooth the corrupted normals
10. Integrate the smoothed normals, and output the surface as a .obj file
11. Calculate the angular difference between baseline and smoothed normals, and output this

#### Parameters

| Parameter                   | Description                                                  | Default                 |
| --------------------------- | ------------------------------------------------------------ | ----------------------- |
| `--images` [REQUIRED]       | Path to directory containing input images (e.g. `frog/Objects`). |                         |
| `--ldirs` [REQUIRED]        | Path to .txt file containing lighting directions (e.g. `frog/light_directions.txt`). |                         |
| `--results`                 | Path to directory to output results to. Will be created if it does not exist. | `results`               |
| `--algorithm`               | Photometric Stereo algorithm to use when reconstructing normals. One of "basic" or "sbl". | "sbl"                   |
| `--num-processes`           | Number of processes to use for SBL Photometric Stereo and normal smoothing. If -1, use half of the available processes (`multiprocessing.cpu_count() // 2`). | -1                      |
| `--region`                  | If using the "frog" dataset, crop images to a specific region of the model. Can be one of "whole", "head" or "tummy". | "whole"                 |
| `--specular`                | Specular weight for corrupting images. Should be in the range [0, 1]. | 0.3                     |
| `--noise`                   | Noise weight for corrupting images. Should be in the range [0, 1]. | 0.1                     |
| `--sbl-lambda`              | $\lambda$ parameter for SBL Photometric Stereo.              | $1.0\!\times\! 10^{-3}$ |
| `--sbl-sigma`               | $\sigma$ parameter for SBL Photometric Stereo.               | $1.0\!\times\! 10^{6}$  |
| `--sbl-max-iters`           | Maximum per-pixel iterations for SBL Photometric Stereo.     | 100                     |
| `--sbl-use-paper-algorithm` | If given, use the original update rules as provided in the paper by Ikehata et al. (2012). From testing, this doesn't work. |                         |
| `--no-smooth`               | If given, don't run the smoothing step after reconstructing corrupted normals (if smoothing is enabled, corrupted normals will still be output before smoothing). |                         |
| `--smoothing-f`             | $\sigma_f$ value for the normal smoothing algorithm. Controls the width of the window from which points are sampled. | $2.0$                   |
| `--smoothing-g`             | $\sigma_g$ value for the normal smoothing algorithm.         | $0.1$                   |
| `--window-size`             | Size of window from which points are sampled when smoothing the reconstructed normals. If -1, use twice the value of $\sigma_f$. |                         |

### main.ipynb

This notebook contains the same functionality as `main.py`, but split into cells that give more flexibility to play around with the outputs of each function. For example, it has cells to view the constituent parts of a corrupted image (such as the specular mask), which is not included in `main.py`

# References

- Bui-Tuong, P., 1975. Illumination for computer generated pictures. Cacm.

- Frankot, R. and Chellappa, R., 1988. A method for enforcing integrability in shape from
  shading algorithms. Ieee transactions on pattern analysis and machine intelligence, 10(4),
  pp.439–451.

- Ikehata, S., Wipf, D., Matsushita, Y. and Aizawa, K., 2012. Robust photometric stereo
  using sparse regression. 2012 ieee conference on computer vision and pattern recognition.
  pp.318–325.

- Jones, T., Durand, F. and Zwicker, M., 2004. Normal improvement for point rendering. Ieee
  computer graphics and applications, 24(4), pp.53–56.

- Woodham, R.J., 1980. Photometric method for determining surface orientation from multiple
  images. Optical engineering, 19(1), pp.139–144.

- Xiong, Y., Chakrabarti, A., Basri, R., Gortler, S.J., Jacobs, D.W. and Zickler, T., 2015. From
  shading to local shape. IEEE transactions on pattern analysis and machine intelligence,
  37(1), pp.67–79.
