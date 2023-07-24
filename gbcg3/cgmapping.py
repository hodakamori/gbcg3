import logging
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from gbcg3.gbcg.gbcg import GraphBasedCoarseGraining
from gbcg3.mapping import AtomMaps
from gbcg3.structure.lammps import LammpsStructure


@dataclass
class AA2CG:
    traj: List[str] = None
    data: str = None
    mode: Literal["progressive", "spectral"] = "progressive"
    niter: int = None
    min_level: Optional[List[int]] = field(default_factory=lambda: [2, 2, 2, 2, 2])
    max_level: Optional[List[int]] = field(default_factory=lambda: [6, 6, 6, 6, 6])
    output_dir: Optional[str] = os.getcwd()
    name_mapfile: Optional[str] = None
    priority_mapfile: Optional[str] = None
    sfreq: Optional[int] = 1
    max_samp: Optional[int] = 1
    sim_ratio: Optional[float] = 1
    typing: Optional[str] = "all"
    max_size: Optional[float] = float("inf")
    log_level: Optional[
        Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    ] = logging.DEBUG
    log_filename: Optional[str] = None
    weight_style: Optional[Literal["mass", "diff"]] = None

    def __post_init__(self) -> None:
        self.logger = logging.getLogger("gbcg3")
        self.logger.setLevel(logging.getLevelName(self.log_level))

        if self.log_filename:
            handler = logging.FileHandler(self.log_filename)
            self.logger.addHandler(handler)

        self.logger.info("Screening files")
        self.structure = LammpsStructure(
            traj_list=self.traj, inc_list="all", data=self.data
        )
        self.cgmap = AtomMaps(
            mass_map=self.structure.mass_map,
            name_mapfile=self.name_mapfile,
            priority_mapfile=self.priority_mapfile,
        )
        self.structure.assign_cgmap(self.cgmap)
        self.structure.assign_mols()

    def run(self) -> None:
        gbcg = GraphBasedCoarseGraining(
            structure=self.structure,
            cgmap=self.cgmap,
            max_samp=self.max_samp,
            min_level=self.min_level,
            max_level=self.max_level,
            sfreq=self.sfreq,
            niter=self.niter,
            max_size=self.max_size,
            sim_ratio=self.sim_ratio,
            typing=self.typing,
            output_dir=self.output_dir,
            weight_style=self.weight_style,
            mode=self.mode,
        )
        gbcg.map_molecules()
        gbcg.assign_cg_type()
        gbcg.get_types_at_each_history_level()
        gbcg.write()
        self.logger.info("Wrapping up...")
        gbcg.tan /= gbcg.nsamp
