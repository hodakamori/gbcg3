# gbcg3: Graph-Based Coarse Graining in Python 3

gbcg3 is a Python package that automates the creation of coarse-grained models for molecular simulations from all-atom models. This package reimplements [Graph-based coarse graining method](https://github.com/xmwebb/GBCG) in Python 3, providing an efficient tool for the modern computational chemistry community.

![](./profile_pic.png)

## Features

- **Automated Generation of Coarse-grained Models**: gbcg3 can generate coarse-grained models for molecular simulations from full atomistic models, streamlining the process and reducing the amount of manual work required.

- **Graph-based Coarse Graining**: This package utilizes the Graph-based coarse graining method, offering a powerful technique to simplify molecular systems while preserving their essential features.

- **Compatibility with LAMMPS**: You can use the LAMMPS trajectory and data files as inputs for this package. We plan to extend the range of compatible formats in future updates.

## Installation

To install gbcg3, use pip:

```bash
pip install gbcg3
```

## Usage

After you have installed the package, you can use it as follows:

```python
from gbcg3 import AA2CG

# Create a CoarseGrain object with your LAMMPS data file
aa2cg = AA2CG(
        traj=["atom.lammpstrj"],
        data="sys.data",
        niter=5,
        min_level=[2, 2, 2, 3, 4],
        max_level=[2, 3, 3, 3, 4],
        output_dir="output",
        log_filename=None,
        names="lmps2type.map",
    )

# Perform coarse graining
aa2cg.run()
```

This will create a coarse-grained model from your LAMMPS data file. Some examples are in `example`.

## Contributing

We welcome any contributions! If you have suggestions for additional features, find bugs, or want to improve the package in any other way, feel free to open an issue or a pull request.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
