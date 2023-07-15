import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, TypeVar, Union

from gbcg3.structure.aux import Atoms, get_adj_list, get_mass_map, load_atoms

T = TypeVar("T", bound="LammpsStructure")


@dataclass
class LammpsStructure:
    traj_list: List[Path] = field(default_factory=list)
    inc_list: Union[str, List[Path]] = field(default_factory=list)
    data: Path = None
    traj: List[Atoms] = field(default_factory=list)

    def assign_mols(self: T):
        untested = sorted(self.atoms["id"].keys())  # add everyone to being untested
        tested = []  # initialize list for tracking who has been tested
        queue = []  # initialize queue list
        mols = []
        self.logger.info(f"# Total number of atoms to be assigned: {len(untested)}")

        while untested:
            wait = []  # initialize wait list
            if not queue:  # add to queue list if necessary
                queue.append(untested[0])
                mols.append([])
            for i in queue:  # go through current queue list
                neighbors = self.bonds[i]  # find neighbor atoms
                mols[-1].append(i)  # add to current molecule
                neighbors = [
                    ni for ni in neighbors if ni not in tested and ni not in queue
                ]  # only explore if untested/not in queue
                idi = self.atoms["id"][i]
                for j in neighbors:  # for each neighbor
                    idj = self.atoms["id"][j]
                tested.append(i)  # add i to tested listed
                untested.pop(untested.index(i))  # remove i from untested list
                wait.extend(neighbors)  # add neighbors to wait list
            queue = list(set(wait[:]))

        self.logger.info(f"# Total number of molecules: {len(mols)}")
        self.atoms["nmol"] = len(mols)
        self.atoms["molid"] = [-1] * len(self.atoms["type"])
        the_mols = [0] * self.atoms["nmol"]
        for i, mol in enumerate(mols):
            self.logger.info(f"#---Number of atoms in mol {i}: {len(mol)}")
            the_mols[i] = sorted(mol)
            for j in mol:
                self.atoms["molid"][self.atoms["id"][j]] = i

        self.mols = the_mols
        return self.atoms, the_mols

    def assign_cgmap(self: T, cgmap) -> T:
        self.atoms["mass"] = [cgmap.mass_map[typ] for typ in self.atoms["type"]]
        self.atoms["priority"] = [cgmap.priority_map[typ] for typ in self.atoms["type"]]
        return self

    def __post_init__(self: T):
        self.logger = logging.getLogger("gbcg3")
        self.atoms = load_atoms(self.traj_list, self.inc_list)
        self.mass_map = get_mass_map(self.data, self.logger)
        self.bonds = get_adj_list(self.data, self.atoms, self.logger)
