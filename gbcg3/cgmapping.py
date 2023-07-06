from dataclasses import dataclass, field
from gbcg3.structure.lammps import LammpsStructure
from gbcg3.cgmap.cgmap import CGMap
from gbcg3.gbcg.gbcg import GraphBasedCoarseGraining
from typing import List, Optional, IO, Literal
import logging
import os


@dataclass
class AA2CG:
    traj: List[str] = None
    data: str = None
    niter: int = None
    min_level: List[int] = field(default_factory=[2, 2, 2, 2, 2])
    max_level: List[int] = field(default_factory=[6, 6, 6, 6, 6])
    output_dir: Optional[str] = os.getcwd()
    names: Optional[str] = None
    pmap: Optional[str] = None
    sfreq: Optional[int] = 1
    max_samp: Optional[int] = 1
    sim_ratio: Optional[float] = 1
    typing: Optional[str] = "all"
    max_size: Optional[float] = float("inf")
    log_level: Optional[
        Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    ] = logging.DEBUG
    log_filename: Optional[str] = None

    def __post_init__(self):
        self.logger = logging.getLogger("gbcg3")
        self.logger.setLevel(logging.getLevelName(self.log_level))

        if self.log_filename:
            handler = logging.FileHandler(self.log_filename)
            self.logger.addHandler(handler)

        self.logger.info("Screening files")
        self.structure = LammpsStructure(
            traj_list=self.traj, inc_list="all", data=self.data
        )
        self.cgmap = CGMap(
            mass_map=self.structure.mass_map,
            names=self.names,
            pmap=self.pmap,
        )
        self.structure.assign_cgmap(self.cgmap)
        self.structure.assign_mols()

    def run(self):
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
        )
        gbcg.map_molecules()
        gbcg.assign_cg_type()
        gbcg.get_types_at_each_history_level()
        gbcg.write()
        self.logger.info("Wrapping up...")
        gbcg.tan /= gbcg.nsamp
