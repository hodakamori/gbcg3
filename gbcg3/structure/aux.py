from logging import Logger
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict, Union

import numpy as np
from gbcg3.structure.utils import append_to_dict


class Atoms(TypedDict):
    id: Dict[int, int]
    type: List[int]
    charge: List[float]
    coords: List[List[float]]
    force: List[List[float]]
    mass: List[float]
    priority: List[float]


def load_atoms(traj_list: List[Path], inc_list: Union[str, List[Path]]) -> Atoms:
    # OPEN FILES
    fid_list = [0] * len(traj_list)
    for i, f in enumerate(traj_list):
        fid_list[i] = open(f, "r")
        fid_list[i].readline()

    # EXTRACT HEADER INFORMATION
    natm = []
    box = np.zeros([3, 2])
    for fid in fid_list:
        for i in range(2):
            fid.readline()
        line = fid.readline().strip().split()
        natm += [int(line[0])]
        fid.readline()

        # GET BOX INFORMATION
        box[0][:] = [v for v in fid.readline().strip().split()]
        box[1][:] = [v for v in fid.readline().strip().split()]
        box[2][:] = [v for v in fid.readline().strip().split()]
        line = fid.readline().strip().split()
        line = line[2:]
        ind_id = line.index("id")
        ind_typ = line.index("type")

    # PARTIALLY INITIALIZE 'atoms' STRUCTURE
    atoms: Atoms = {}
    atoms["id"] = {}
    atoms["type"] = []
    atoms["charge"] = []

    # GET ATOM INFORMATION
    L = box[:, 1] - box[:, 0]
    count = 0
    for i, fid in enumerate(fid_list):
        for j in range(natm[i]):
            line = fid.readline().strip().split()
            ind_j = int(line[ind_id])
            type_j = int(line[ind_typ])
            if inc_list == "all" or type_j in inc_list:
                atoms["id"][ind_j] = count
                atoms["type"] += [type_j]
                atoms["charge"] += [0.0]
                count += 1

    # FINISH INITIALIZATION
    atoms["coords"] = np.zeros([count, 3])
    atoms["forces"] = np.zeros([count, 3])
    atoms["count"] = count

    # CLOSE FILES
    for i, f in enumerate(traj_list):
        fid_list[i].close()

    return atoms


def get_mass_map(data: Path, logger: Logger) -> Dict[int, float]:
    if data != "none":
        logger.info(f"# Extracting masses from {data} ...")
        fid = open(data)
        line = fid.readline().strip().split()
        while True:
            if len(line) == 3 and line[1] == "atom" and line[2] == "types":
                ntype = int(line[0])
                logger.info(f"# A total of {ntype} atom types reported!!!")
            if len(line) == 1 and line[0] == "Masses":
                fid.readline()
                mass_map = {}
                for i in range(ntype):
                    line = fid.readline().strip().split()
                    mass_map[int(line[0])] = float(line[1])
                logger.info("# Masses field found and recorded! Breaking from file...")
                fid.close()
                break
            line = fid.readline().strip().split()
    return mass_map


def get_adj_list(data: Path, atoms: Atoms, logger: Logger) -> Dict[int, List[int]]:
    # EXAMINE TOPOLOGY FROM DATA FILE
    adjlist = {}
    if data != "none":
        logger.info(f"# Extracting topology from {data} ...")
        fid = open(data)
        line = fid.readline().strip().split()
        while True:
            if len(line) == 2 and line[1] == "bonds":
                nbond = int(line[0])
                logger.info(f"# A total of {nbond} bonds reported!!!")
            if len(line) == 1 and line[0] == "Bonds":
                fid.readline()
                for j in range(nbond):
                    line = fid.readline().strip().split()
                    bond = [int(el) for el in line]
                    if bond[2] in atoms["id"].keys():
                        append_to_dict(adjlist, bond[2], bond[3])
                        append_to_dict(adjlist, bond[3], bond[2])
                logger.info("# Bonds field found and recorded! Breaking from file...")
                fid.close()
                break
            line = fid.readline().strip().split()

    return adjlist
