from logging import Logger
from pathlib import Path
from typing import Dict, List
from gbcg3.structure.lammps.helpers import append_to_dict
from gbcg3.structure.lammps.types import Atoms


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


def get_charge_map(files, atoms):
    if files["data"] != "none":
        print("# Extracting charges from ", files["data"], " ...")
        fid = open(files["data"])
        line = fid.readline().strip().split()
        qtot = 0.0
        while True:
            if len(line) == 2 and line[1] == "atoms":
                natm = int(line[0])
                print("# A total of ", natm, " atoms reported!!!")
            if len(line) == 3 and line[1] == "atom" and line[2] == "types":
                ntype = int(line[0])
                q4type = [0.0] * ntype
                n4type = [0.0] * ntype
            if len(line) >= 1 and line[0] == "Atoms":
                fid.readline()
                for j in range(natm):
                    line = fid.readline().strip().split()
                    ind = int(line[0])
                    typ = int(line[2])
                    q = float(line[3])
                    if ind in atoms["id"]:
                        ptr = atoms["id"][ind]
                        atoms["charge"][ptr] = q
                        qtot += q
                    q4type[typ - 1] += q
                    n4type[typ - 1] += 1.0
                fid.close()
                break
            line = fid.readline().strip().split()
    qavg = [qi / ni if ni > 0 else 0 for qi, ni in zip(q4type, n4type)]

    # create a type dictionary
    qmap = {}
    for i in range(ntype):
        qmap[i + 1] = qavg[i]
    return qmap
