#!/home/mwebb/anaconda/bin/python
# This is a module that contains functions for reading information from files commonly used in LAMMPS
# import imp
from numpy import *
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Union, Optional
import logging


@dataclass
class Atoms:
    id: Dict[int, int] = field(default_factory=dict)
    type: List[int] = field(default_factory=list)
    charge: List[float] = field(default_factory=list)
    coords: List[List[float]] = field(default_factory=list)
    force: List[List[float]] = field(default_factory=list)
    mass: List[float] = field(default_factory=list)
    priority: List[float] = field(default_factory=list)


@dataclass
class LammpsStructure:
    traj_list: List[str] = field(default_factory=list)
    inc_list: Union[str, List[str]] = field(default_factory=list)
    data: str = None
    traj: List[Atoms] = field(default_factory=list)

    def _load_atoms(self):
        # OPEN FILES
        fid_list = [0] * len(self.traj_list)
        for i, f in enumerate(self.traj_list):
            fid_list[i] = open(f, "r")
            fid_list[i].readline()

        # EXTRACT HEADER INFORMATION
        natm = []
        box = zeros([3, 2])
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
                if self.inc_list == "all" or type_j in self.inc_list:
                    atoms["id"][ind_j] = count
                    atoms["type"] += [type_j]
                    atoms["charge"] += [0.0]
                    count += 1

        # FINISH INITIALIZATION
        atoms["coords"] = zeros([count, 3])
        atoms["forces"] = zeros([count, 3])
        atoms["count"] = count

        # CLOSE FILES
        for i, f in enumerate(self.traj_list):
            fid_list[i].close()

        return atoms

    def _get_mass_map(self):
        if self.data != "none":
            self.logger.info(f"# Extracting masses from {self.data} ...")
            fid = open(self.data)
            line = fid.readline().strip().split()
            while True:
                if len(line) == 3 and line[1] == "atom" and line[2] == "types":
                    ntype = int(line[0])
                    self.logger.info(f"# A total of {ntype} atom types reported!!!")
                if len(line) == 1 and line[0] == "Masses":
                    fid.readline()
                    mass_map = {}
                    for i in range(ntype):
                        line = fid.readline().strip().split()
                        mass_map[int(line[0])] = float(line[1])
                    self.logger.info(
                        "# Masses field found and recorded! Breaking from file..."
                    )
                    fid.close()
                    break
                line = fid.readline().strip().split()

        return mass_map

    def _get_adj_list(self):
        # EXAMINE TOPOLOGY FROM DATA FILE
        adjlist = {}
        if self.data != "none":
            self.logger.info(f"# Extracting topology from {self.data} ...")
            fid = open(self.data)
            line = fid.readline().strip().split()
            while True:
                if len(line) == 2 and line[1] == "bonds":
                    nbond = int(line[0])
                    self.logger.info(f"# A total of {nbond} bonds reported!!!")
                if len(line) == 1 and line[0] == "Bonds":
                    fid.readline()
                    for j in range(nbond):
                        line = fid.readline().strip().split()
                        bond = [int(el) for el in line]
                        if bond[2] in self.atoms["id"].keys():
                            append_to_dict(adjlist, bond[2], bond[3])
                            append_to_dict(adjlist, bond[3], bond[2])
                    self.logger.info(
                        "# Bonds field found and recorded! Breaking from file..."
                    )
                    fid.close()
                    break
                line = fid.readline().strip().split()
        return adjlist

    def assign_mols(self):
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

        # self.logger.info("********************************************************\n\n")
        self.mols = the_mols
        return self.atoms, the_mols

    def assign_cgmap(self, cgmap):
        self.atoms["mass"] = [cgmap.mass_map[typ] for typ in self.atoms["type"]]
        self.atoms["priority"] = [cgmap.priority[typ] for typ in self.atoms["type"]]
        return self

    def __post_init__(self):
        self.logger = logging.getLogger("gbcg3")
        self.atoms = self._load_atoms()
        self.mass_map = self._get_mass_map()
        self.bonds = self._get_adj_list()


# ==================================================================
#  AUX: append_to_dict
# ==================================================================
def append_to_dict(dic, key, val):
    if key in dic:
        dic[key].append(val)
    else:
        dic[key] = [val]


# ==================================================================
#  AUX: init_atoms
# ==================================================================
def init_atoms(N):
    atoms = {}
    atoms["coords"] = zeros([N, 3])
    atoms["forces"] = zeros([N, 3])
    atoms["type"] = [0] * N
    atoms["id"] = {}
    atoms["pos"] = zeros([N, 3])
    atoms["charge"] = [0.0] * N
    return atoms


# ==================================================================
#  AUX: get_charge_map
# ==================================================================
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


# ==================================================================
#  AUX: process_frame(fid)
# ==================================================================
def process_frame(fid_list, inc_list, atoms):
    # EXTRACT HEADER INFORMATION
    natm = []
    box = zeros([3, 2])
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
                atoms["coords"][id_j] = array(
                    [float(i) for i in [line[ind_x], line[ind_y], line[ind_z]]]
                )
                if forces_present:
                    atoms["forces"][id_j] = array(
                        [float(i) for i in [line[ind_fx], line[ind_fy], line[ind_fz]]]
                    )
                else:
                    atoms["forces"][id_j] = zeros([3])

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
    box = zeros([3, 2])
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
    atoms["coords"] = zeros([count, 3])
    atoms["forces"] = zeros([count, 3])
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
