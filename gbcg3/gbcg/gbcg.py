import copy
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Literal

from gbcg3.gbcg.core import (
    assign_CG_types,
    get_CG_coords,
    reduction_mapping,
    temp_types,
)
from gbcg3.gbcg.helpers import unwrap_mols
from gbcg3.gbcg.io import (
    open_files,
    write_CG_lammpstrj,
    write_CG_map,
    write_CG_pdb,
    write_CG_xyz,
    write_data_file,
    write_groups,
    write_xyz,
)
from gbcg3.mapping import AtomMaps
from gbcg3.structure.lammps import LammpsStructure
from gbcg3.structure.lammps.trajectory import process_frame, skip_frame


@dataclass
class GraphBasedCoarseGraining:
    structure: LammpsStructure = None
    cgmap: AtomMaps = None
    mode: Literal["progressive", "spectral"] = "progressive"
    niter: int = 5

    pdbdir: str = "pdb_files/"
    mapdir: str = "map_files/"
    xyzdir: str = "xyz_files/"
    lmpdir: str = "lammpstrj_files/"

    min_level: Optional[List[int]] = field(default_factory=[2, 2, 2, 2, 2])
    max_level: Optional[List[int]] = field(default_factory=[6, 6, 6, 6, 6])
    output_dir: str = None

    max_samp: Optional[int] = 1
    sfreq: Optional[float] = 1.0
    max_size: Optional[float] = float("inf")
    sim_ratio: Optional[float] = 1
    typing: Optional[str] = "all"

    weight_style: Optional[Literal["mass", "diff"]] = None

    def _open_files(self) -> None:
        self.fxyz, self.flmp, self.fpdb, self.fmap, self.fall = open_files(
            self.structure,
            self.output_dir,
            self.xyzdir,
            self.lmpdir,
            self.pdbdir,
            self.mapdir,
            self.niter,
            self.min_level,
            self.max_level,
            self.mode,
        )

    def __post_init__(self):
        self.logger = logging.getLogger("gbcg3")

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        for outdir in [self.pdbdir, self.mapdir, self.xyzdir, self.lmpdir]:
            if not os.path.isdir(os.path.join(self.output_dir, outdir)):
                os.makedirs(os.path.join(self.output_dir, outdir))

        self._open_files()
        self.frame: int = 0
        self.nsamp: int = 0
        self.ntraj: List[str] = len(
            self.structure.traj_list
        )  # number of trajectory files
        self.ftraj: List[int] = [0] * self.ntraj  # list with trajectory file ids
        self.still_data = [
            0
        ] * self.ntraj  # list to indicate if data remains to be read
        self.tan: float = 0.0
        self.cgmol: List = []
        self.cghist: List = []

    def map_molecules(self):
        for i, moli in enumerate(self.structure.mols):
            self.logger.info("Reduction Summary for molecule {}\n\n".format(i))
            CGmoli, histi = reduction_mapping(
                self.output_dir,
                self.logger,
                self.niter,
                self.min_level,
                self.max_level,
                self.max_size,
                moli,
                self.structure.atoms,
                copy.deepcopy(self.structure.bonds),
                self.mode,
                weight_style=self.weight_style,
            )
            write_groups(
                self.output_dir, i, CGmoli, self.structure.atoms, self.cgmap.names_map
            )
            self.cgmol.append(CGmoli)
            self.cghist.append(histi)

    def assign_cg_type(self):
        self.logger.info("Assigning preliminary CG site types")
        self.ftyp, self.cgmap.cgtypes, self.nCgType = assign_CG_types(
            self.output_dir,
            self.typing,
            self.sim_ratio,
            self.structure.atoms,
            self.cgmol,
        )

    def get_types_at_each_history_level(self):
        # GET TYPES AT EACH HISTORY LEVEL
        nhist = len(self.cghist[0])
        nmol = len(self.cghist)
        tmpCGmol = [[] for i in range(nhist)]
        for i in range(nhist):
            for j in range(nmol):
                tmpCGmol[i].extend([self.cghist[j][i]])
        for i in range(nhist):
            temp_types(self.typing, self.sim_ratio, self.structure.atoms, tmpCGmol[i])

        # BEGIN PROCESSING, FRAME BY FRAME
        self.logger.info("Mapping supplied trajectories")
        for i, f in enumerate(self.structure.traj_list):
            self.ftraj[i] = open(f, "r")
            self.still_data[i] = self.ftraj[i].readline()

    def write(self):
        # BEGIN PROCESSING, FRAME BY FRAME
        self.logger.info("Mapping supplied trajectories")
        for i, f in enumerate(self.structure.traj_list):
            self.ftraj[i] = open(f, "r")
            self.still_data[i] = self.ftraj[i].readline()

        # READ DATA WHILE ALL FILES STILL OPEN
        while all(self.still_data) and self.nsamp < self.max_samp:
            # IF NEED TO SAMPLE
            if self.frame % self.sfreq == 0:
                ti = time.time()
                # PROCESS TRAJECTORY FRAME
                (self.structure.atoms, L, halfL, box) = process_frame(
                    self.ftraj, "all", self.structure.atoms
                )

                # UNWRAP MOLECULES TO COMPUTE ANY SELF PROPERTIES
                self.structure.atoms = unwrap_mols(
                    self.structure.atoms, self.structure.bonds, L, halfL
                )
                write_xyz(self.structure.atoms, self.cgmap.names_map, self.fall)

                # WRITE OUT COORDINATES FOR THE GROUPS
                molcpy = []
                for i, CGmoli in enumerate(self.cgmol):
                    for j, histj in enumerate(self.cghist[i]):
                        if self.nsamp == 0:
                            tmp = get_CG_coords(
                                copy.deepcopy(histj), self.structure.atoms
                            )
                            write_CG_pdb(
                                tmp,
                                self.structure.atoms,
                                self.cgmap.names_map,
                                self.fpdb[i][j],
                            )
                            write_CG_map(
                                tmp,
                                self.structure.atoms,
                                self.cgmap.names_map,
                                self.fmap[j],
                            )
                    tmp = get_CG_coords(copy.deepcopy(CGmoli), self.structure.atoms)
                    molcpy.append(tmp)
                    write_CG_lammpstrj(tmp, self.flmp[i], self.nsamp, box)
                    write_CG_xyz(
                        tmp, self.structure.atoms, self.cgmap.names_map, self.fxyz[i]
                    )
                # WRITE OUT DATA FILE
                if self.nsamp == 0:
                    write_data_file(
                        self.ftyp,
                        self.output_dir,
                        self.structure.atoms,
                        molcpy,
                        box,
                        self.nCgType,
                        self.cgmap.cgtypes,
                        self.mode,
                    )

                # FINISH BOOK KEEPING
                self.tan += time.time() - ti
                self.nsamp += 1

            # OTHERWISE... SKIP FRAME
            else:
                for fid in self.ftraj:
                    skip_frame(fid)
            self.frame += 1
            if self.frame % 1000 == 0:
                self.logger.info(
                    f"# {self.nsamp} samples taken after {self.frame} frames..."
                )
            # GET CONTINUE CONDITION
            for i, fid in enumerate(self.ftraj):
                self.still_data[i] = fid.readline()
