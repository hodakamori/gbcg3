import difflib
import os
import tempfile
from typing import List

import pytest
from gbcg3 import AA2CG


def compare_files(test_file_path: str, expect_file_path: str) -> List[str]:
    with open(test_file_path) as f1, open(expect_file_path) as f2:
        text1_lines = f1.readlines()
        text2_lines = f2.readlines()
        d = difflib.Differ()

        diff = d.compare(text1_lines, text2_lines)
        differences = [l for l in diff if l.startswith("- ") or l.startswith("+ ")]

        return differences


@pytest.mark.parametrize(
    "testfile_dir, traj, data, mapfile, niter, min_level, max_level",
    [
        (
            "tests/progressive/dendrimer",
            "coords.lammpstrj",
            "sys.data",
            "lmps2type.map",
            5,
            [2, 2, 2, 2, 2],
            [6, 6, 6, 6, 6],
        ),
        (
            "tests/progressive/hypromellose",
            "atom.lammpstrj",
            "sys.data",
            "lmps2type.map",
            5,
            [2, 2, 2, 3, 4],
            [2, 3, 3, 3, 4],
        ),
    ],
)
def test_progressive(testfile_dir, traj, data, mapfile, niter, min_level, max_level):
    with tempfile.TemporaryDirectory() as dname:
        aa2cg = AA2CG(
            traj=[os.path.join(testfile_dir, traj)],
            data=os.path.join(testfile_dir, data),
            niter=niter,
            min_level=min_level,
            max_level=max_level,
            name_mapfile=os.path.join(testfile_dir, mapfile),
            output_dir=dname,
            log_level="ERROR",
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
