import os
import tempfile
from .utils import compare_files
import pytest
from gbcg3 import AA2CG


@pytest.mark.parametrize(
    "testfile_dir, traj, data, mapfile, niter, weight_style, mode",
    [
        (
            "tests/spectral/toluene",
            "toluene.lammpstrj",
            "sys.data",
            "lmps2type.map",
            3,
            "mass",
            "spectral",
        ),
        (
            "tests/spectral/pentadecane",
            "pentadecane.lammpstrj",
            "sys.data",
            "lmps2type.map",
            3,
            "mass",
            "spectral",
        ),
        (
            "tests/spectral/rhodopsin",
            "rhodo.lammpstrj",
            "sys.data",
            "lmps2type.map",
            4,
            None,
            "spectral",
        ),
    ],
)
def test_spectral(testfile_dir, traj, data, mapfile, niter, weight_style, mode):
    with tempfile.TemporaryDirectory() as dname:
        aa2cg = AA2CG(
            traj=[os.path.join(testfile_dir, traj)],
            data=os.path.join(testfile_dir, data),
            niter=niter,
            name_mapfile=os.path.join(testfile_dir, mapfile),
            output_dir=dname,
            log_level="ERROR",
            weight_style=weight_style,
            mode=mode,
        )
        aa2cg.run()
        for f in os.listdir(dname):
            if os.path.isfile(os.path.join(dname, f)) and not (f.endswith(".data")):
                diff = compare_files(
                    os.path.join(testfile_dir, f),
                    os.path.join(dname, f),
                )
                assert diff == []

            elif os.path.isdir(os.path.join(dname, f)):
                for fname in os.listdir(os.path.join(dname, f)):
                    diff = compare_files(
                        os.path.join(testfile_dir, f, fname),
                        os.path.join(dname, f, fname),
                    )
                    assert diff == []
