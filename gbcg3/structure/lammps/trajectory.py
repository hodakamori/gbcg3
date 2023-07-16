from logging import Logger
from pathlib import Path
from typing import List, Union

import numpy as np
from gbcg3.structure.lammps.types import Atoms


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


# ==================================================================
#  AUX: process_frame(fid)
# ==================================================================
def process_frame(fid_list, inc_list, atoms):
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
        ind_x = line.index("x")
        ind_y = line.index("y")
        ind_z = line.index("z")
        forces_present = False
        if "fx" in line:
            forces_present = True
            ind_fx = line.index("fx")
            ind_fy = line.index("fy")
            ind_fz = line.index("fz")

    # GET ATOM INFORMATION
    L = box[:, 1] - box[:, 0]
    for i, fid in enumerate(fid_list):
        for j in range(natm[i]):
            line = fid.readline().strip().split()
            ind_j = int(line[ind_id])
            type_j = int(line[ind_typ])
            if inc_list == "all" or type_j in inc_list:
                id_j = atoms["id"][ind_j]
                atoms["coords"][id_j] = np.array(
                    [float(i) for i in [line[ind_x], line[ind_y], line[ind_z]]]
                )
                if forces_present:
                    atoms["forces"][id_j] = np.array(
                        [float(i) for i in [line[ind_fx], line[ind_fy], line[ind_fz]]]
                    )
                else:
                    atoms["forces"][id_j] = np.zeros([3])

    return (atoms, L, 0.5 * L, box)


# ==================================================================
#  AUX: skip_frame(ftraj)
# ==================================================================
def skip_frame(ftraj):
    # SKIP HEADER INFO
    for i in range(2):
        ftraj.readline()
    line = ftraj.readline().strip().split()
    natm = int(line[0])
    for i in range(5 + natm):
        ftraj.readline()


# ==================================================================
#  AUX: screen_frame(fid)
# ==================================================================
def screen_frame(traj_list, inc_list):
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
    atoms = {}
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


# ==================================================================
#  AUX: skip_frame(ftraj)
# ==================================================================
def skip_frame(ftraj):
    # SKIP HEADER INFO
    for i in range(2):
        ftraj.readline()
    line = ftraj.readline().strip().split()
    natm = int(line[0])
    for i in range(5 + natm):
        ftraj.readline()
