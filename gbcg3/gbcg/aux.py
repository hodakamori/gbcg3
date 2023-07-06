import os
from gbcg3.structure.lammps import LammpsStructure

def open_files(lmptrj:):
    # make the directories to contain coordinate files
    fxyz = []
    flmp = []
    fpdb = [[] for mol in self.lmptrj.mols]
    fmap = []
    fall = open(os.path.join(self.output_dir, "atoms.xyz"), "w")

    for i, moli in enumerate(self.lmptrj.mols):
        fname_xyz = os.path.join(self.output_dir, self.xyzdir, f"CG.mol_{i}.xyz")
        fname_lmp = os.path.join(
            self.output_dir, self.lmpdir, f"CG.mol_{i}.lampstrj"
        )
        fname_pdb = os.path.join(self.output_dir, self.pdbdir, f"CG.mol_{i}.0.pdb")
        fxyz.append(open(fname_xyz, "w"))
        flmp.append(open(fname_lmp, "w"))
        fpdb[i].append(open(fname_pdb, "w"))
        for iIter in range(self.niter):
            for lvl in range(
                self.min_level[iIter],
                self.max_level[iIter] + 1,
                1,
            ):
                fname_pdb = os.path.join(
                    self.output_dir, self.pdbdir, f"mol_{i}.{iIter+1}_{lvl}.pdb"
                )
                fpdb[i].append(open(fname_pdb, "w"))

    fname_map = os.path.join(self.output_dir, self.mapdir, "CG.map")
    fmap.append(open(fname_map, "w"))
    for iIter in range(self.niter):
        for lvl in range(
            self.min_level[iIter],
            self.max_level[iIter] + 1,
            1,
        ):
            fname_map = os.path.join(
                self.output_dir, self.mapdir, f"iter.{iIter+1}_{lvl}.map"
            )
            fmap.append(open(fname_map, "w"))
    self.fxyz, self.flmp, self.fpdb, self.fmap, self.fall = (
        fxyz,
        flmp,
        fpdb,
        fmap,
        fall,
    )
    return fxyz, flmp, fpdb, fmap, fall
