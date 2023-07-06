import copy
import os
from math import *

import numpy as np
from gbcg3.utils.logger import print_status

# ==================================================================
# CONSTANTS AND DIRECTORIES
# ==================================================================
mlight = 3.500


# ==================================================================
#  AUX: is_number
# ==================================================================
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ==================================================================
#  AUX: approximate_sigma
# ==================================================================
def approximate_sigma(node, supp, atoms):
    nodeId = atoms["id"][node]
    rmax = atoms["coords"][nodeId].copy()
    rmin = atoms["coords"][nodeId].copy()
    for i in supp:
        iId = atoms["id"][i]
        ri = atoms["coords"][iId].copy()
        rmax[ri > rmax] = ri[ri > rmax]
        rmin[ri < rmin] = ri[ri < rmin]
    rmax += 2.5
    rmin -= 2.5
    dr = rmax - rmin
    ravg = np.mean(dr)
    return ravg * 0.5


# ==================================================================
#  AUX: compute_angle
# ==================================================================
def compute_angle(ri, rj, rk):
    rji = ri - rj
    rjk = rk - rj
    cost = np.dot(rji, rjk) / np.linalg.norm(rji) / np.linalg.norm(rjk)
    theta = np.arccos(cost) * 180.0 / np.pi
    return theta


# ==================================================================
#  AUX: wrap_into_box
# ==================================================================
def wrap_into_box(r, box):
    # shift origin to 0 0 0 to do wrapping
    L = np.array([x[1] - x[0] for x in box])
    r -= box[:, 0]
    shift = np.array([floor(ri / Li) for ri, Li in zip(r, L)])
    r -= shift * L
    # shift back
    r += box[:, 0]
    return r


# ==================================================================
#  AUX: process_mapfile
# ==================================================================
# def process_mapfile(mapfile, massMap, mapType):
#     the_map = {}
#     if mapfile is not None:
#         fid = open(mapfile, "r")
#         lines = [line.strip().split() for line in fid]
#         for line in lines:
#             if is_number(line[1]):
#                 the_map[int(line[0])] = float(line[1])
#             else:
#                 the_map[int(line[0])] = line[1]
#     else:
#         if mapType == "priority":
#             # create priority dictionary based on mass
#             for i, m in massMap.items():
#                 if round(m) <= 3.1:
#                     the_map[i] = -1
#                 else:
#                     the_map[i] = 1.0 / round(m)
#                     # the_map[i] = 1
#         elif mapType == "name":
#             for i, m in massMap.items():
#                 the_map[i] = (
#                     mass2el[m]
#                     if m in mass2el
#                     else mass2el[min(mass2el.keys(), key=lambda k: abs(k - m))]
#                 )
#     return the_map


# ==================================================================
#  AUX: unwrap_mols(fid)
# ==================================================================
def unwrap_mols(atoms, bonds, L, halfL):
    # UNWRAP MOLECULES ACCORDING TO PATH WALKS USING BFS
    untested = sorted(atoms["id"].keys())  # add everyone to being untested
    tested = []  # initialize list for tracking who has been tested
    queue = []  # initialize queue list
    while untested:  # while there are untested atp,s
        wait = []  # initialize wait list
        if not queue:  # add to queue list if necessary
            queue.append(untested[0])
        for i in queue:  # go through current queue list
            neighbors = bonds[i]  # find neighbor atoms
            neighbors = [
                ni for ni in neighbors if ni not in tested and ni not in queue
            ]  # only explore if untested
            idi = atoms["id"][i]
            ri = atoms["coords"][idi]
            for j in neighbors:  # for each neighbor
                idj = atoms["id"][j]
                rj = atoms["coords"][idj]
                dr = rj[:] - ri[:]  # compute distance
                shift = np.array([round(val) for val in dr / L])
                atoms["coords"][idj] -= (
                    shift * L
                )  # get nearest image and  adjust coordinates
            tested.append(i)  # add i to tested listed
            untested.pop(untested.index(i))  # remove i from untested list
            wait.extend(neighbors)  # add neighbors to wait list
        queue = list(set(wait[:]))  # make queue the wait list
    return atoms


# ==================================================================
#  AUX: get_bead_coords
# ==================================================================
def get_bead_coords(chain, atoms):
    nbeads = len(chain["groups"])
    coords = np.zeros([nbeads, 3])
    for i, g in enumerate(chain["groups"]):  # for each group
        for j in g:  # for each atom in group
            idj = atoms["id"][j]
            coords[i, :] += (
                atoms["coords"][idj] * atoms["mass"][idj]
            )  # add mass-weighted coordinate
        coords[i, :] /= chain["mass"][i]
    return coords


# ==================================================================
#  AUX: prioritize
# ==================================================================
def prioritize(thisId, avail, atoms, adjlist):
    # first screen by connectivity
    rank = [len(adjlist[i]) for i in avail]
    pval = [atoms["priority"][atoms["id"][i]] for i in avail]
    maxC = max(rank)
    # if there are equal connectivity proceed to priority numbers
    if rank.count(maxC) > 1:
        rank = [pi if ri == maxC else 0 for ri, pi in zip(rank, pval)]
        maxC = max(rank)
        if rank.count(maxC) > 1:
            printf(
                "# Priority is ambiguous for assignment of atom {} to the following: ".format(
                    thisId
                )
            )
            for theId, theRank in zip(avail, rank):
                if theRank == maxC:
                    printf("{} ".format(theId))
                printf(
                    "\n# Consider a different set of priority values. Just rolling with the first one for now...\n"
                )
        return avail[rank.index(maxC)]
    else:
        return avail[rank.index(maxC)]


# ==================================================================
#  AUX: get_CG_coords
# ==================================================================
def get_CG_coords(mol, atoms):
    for node, group in mol.items():
        nodeId = atoms["id"][node]
        mi = atoms["mass"][nodeId]
        crds = mi * atoms["coords"][nodeId][:]
        M = mi
        for i in group["in"]:
            iId = atoms["id"][i]
            mi = atoms["mass"][iId]
            crds += mi * atoms["coords"][iId][:]
            M += mi
        crds /= M
        group["coords"] = crds[:]

    return mol


# ==================================================================
#  AUX: add_if_heavy
# ==================================================================
def add_if_heavy(node, neighbor, beads, atoms):
    nid = atoms["id"][neighbor]
    mi = atoms["mass"][nid]
    if mi > mlight:
        beads[node]["nheavy"] += 1
    if neighbor in beads:
        for j in beads[neighbor]["in"]:
            if j not in beads[node]["in"]:
                mj = atoms["mass"][atoms["id"][j]]
                if mj > mlight:
                    beads[node]["nheavy"] += 1
    return


# ==================================================================
#  AUX: how_many_heavy
# ==================================================================
def how_many_heavy(node, nlist, beads, atoms, options):
    nheavy = 0
    for neighbor in nlist:
        nid = atoms["id"][neighbor]
        mi = atoms["mass"][nid]
        if mi > mlight:
            nheavy += 1
        if neighbor in beads:
            for j in beads[neighbor]["in"]:
                if j not in beads[node]["in"]:
                    mj = atoms["mass"][atoms["id"][j]]
                    if mj > mlight:
                        nheavy += 1
    return nheavy


# ==================================================================
#  AUX: get_mass
# ==================================================================
def get_mass(atoms, node, group):
    m = atoms["mass"][atoms["id"][node]]
    for i in group:
        m += atoms["mass"][atoms["id"][i]]
    return m


# ==================================================================
#  AUX: contract
# ==================================================================
def contract(curr, touched, queue, node, neighbors, neighCN, minCN, atoms, max_size):
    # def contract(curr, touched, queue, node, neighbors, neighCN, minCN, atoms, options):
    contractList = [n for n, c in zip(neighbors, neighCN) if c <= minCN]  # testing
    mList = [get_mass(atoms, n, curr[n]["in"]) for n in contractList]

    # order contract list based on smallest to largest mass
    contractList = [
        q for (p, q) in sorted(zip(mList, contractList), key=lambda pair: pair[0])
    ]
    mList = [get_mass(atoms, n, curr[n]["in"]) for n in contractList]

    # Check how many heavy atoms would be added
    mtot = get_mass(atoms, node, curr[node]["in"])
    addList = []
    for i in contractList:
        madd = get_mass(atoms, i, curr[i]["in"])
        mpot = madd + mtot
        if mpot <= max_size:
            addList.append(i)
            mtot = mpot
    contractList = addList

    # if nheavy_add + curr[node]['nheavy'] > max_size:
    if mtot > max_size:
        touched.add(node)
    else:
        for cnxn in contractList:  # for each
            touched.add(cnxn)  # mark as touched
            curr[node]["in"].add(cnxn)  # augment this as contracted to this node
            add_if_heavy(node, cnxn, curr, atoms)
            curr[cnxn]["adj"].remove(
                node
            )  # remove contractor from adjacency list of contracted
            curr[node]["adj"].remove(
                cnxn
            )  # remove contracted from adjacency list of contractor
            curr[node]["adj"].update(
                curr[cnxn]["adj"]
            )  # resolve the adjacency list of contracted
            curr[node]["in"].update(
                curr[cnxn]["in"]
            )  # resolve the container list of contracted

            # modify adjacency lists of neighbors to point to new node
            for i in curr[cnxn]["adj"]:
                curr[i]["adj"].remove(cnxn)
                curr[i]["adj"].add(node)
            del curr[cnxn]  # remove contracted from current list
            while cnxn in queue:
                queue.pop(queue.index(cnxn))
        touched.add(node)

    return curr, touched, queue


# ==================================================================
#  AUX: make_level_queue
# ==================================================================
def make_level_queue(beads, lvl, atoms, touched):
    queue = []
    queue = [i for i, beadi in beads.items() if len(beadi["adj"]) == lvl]
    istouched = [1 if i in touched else 0 for i in queue]
    queue = [i for i in queue if i not in touched]
    nheavy = [beads[node]["nheavy"] for node in queue]
    nn = [len(beads[node]["adj"] - touched) for node in queue]
    ndg = [
        sum(
            [len(beads[cnct]["adj"] - touched) for cnct in beads[node]["adj"] - touched]
        )
        for node in queue
    ]
    val = [1000 * i + 100 * j + 10 * k for i, j, k in zip(nheavy, nn, ndg)]
    queue = [
        q for (p, q) in sorted(zip(val, queue), key=lambda pair: pair[0], reverse=True)
    ]
    return queue


# ==================================================================
#  AUX: reorder_queue
# ==================================================================
def reorder_queue(queue, touched, beads, tried):
    # ordering 1
    req = [i for i in queue]

    nheavy = [beads[node]["nheavy"] for node in req]
    nn = [len(beads[node]["adj"] - touched) for node in req]
    ndg = [
        sum([len(beads[cnct]["adj"] - touched) for cnct in beads[i]["adj"] - touched])
        for i in req
    ]
    val = [1000 * i + 100 * j + 10 * k for i, j, k in zip(nheavy, nn, ndg)]
    req = [q for (p, q) in sorted(zip(nn, req), key=lambda pair: pair[0], reverse=True)]
    queue = req
    return queue


# ==================================================================
#  AUX: get_overlap
# ==================================================================
def get_overlap(listi, listj):
    n = 0
    ni = len(listi)
    nj = len(listj)
    for el in listi:
        if el in listj:
            listj.pop(listj.index(el))
            n += 1
    return float(n) / float(max(ni, nj))


# ==================================================================
#  AUX: add_type
# ==================================================================
def add_type(typing, atoms, Id, group):
    typ = atoms["type"][Id]
    if typing == "heavy":
        if atoms["mass"][Id] > mlight:
            group["type"].append(typ)
    else:
        group["type"].append(typ)
    return


# ==================================================================
#  AUX: update_charge
# ==================================================================
def update_charge(beadList, atoms):
    for node, bead in beadList.items():
        iId = atoms["id"][node]
        q = atoms["charge"][iId]
        for ib in bead["in"]:
            jId = atoms["id"][ib]
            q += atoms["charge"][jId]
        bead["charge"] = q
    return


# ==================================================================
#  AUX: update_masses
# ==================================================================
def update_masses(beadList, atoms):
    for node, bead in beadList.items():
        iId = atoms["id"][node]
        m = atoms["mass"][iId]
        for ib in bead["in"]:
            jId = atoms["id"][ib]
            m += atoms["mass"][jId]
        bead["mass"] = m
    return


# ==================================================================
#  AUX: temp_types
# ==================================================================
def temp_types(typing, sim_ratio, atoms, beadsList):
    # aggregate and count constituent atom types
    typeList = []

    # iterate over beads and also assign indices
    beadId = 0
    for beads in beadsList:
        for node in sorted(beads.keys()):
            group = beads[node]
            beadId += 1
            group["id"] = beadId
            nodeId = atoms["id"][node]
            add_type(typing, atoms, nodeId, group)
            for j in group["in"]:
                jId = atoms["id"][j]
                add_type(typing, atoms, jId, group)
            theType = [
                [x, group["type"].count(x)] for x in set(group["type"])
            ]  # count constituent atom types
            theType = [
                el for el in sorted(theType, key=lambda pair: pair[0])
            ]  # organize in ascending order
            if theType not in typeList:
                typeList.append(theType)
            group["type"] = theType

    # sort the list of possible types for reduction
    nInType = [len(x) for x in typeList]
    typeList = [
        t for (t, n) in sorted(zip(typeList, nInType), key=lambda pair: pair[1])
    ]

    # Assign all the atom types
    nOfType = [0 for t in typeList]
    for beads in beadsList:
        for node, group in beads.items():
            iType = typeList.index(group["type"])
            group["type"] = iType
            nOfType[iType] += 1

    # Check for similarity in constitution
    uniqueTypes = []
    uniqueExpnd = []
    nunique = 0
    queue = typeList[:]
    typeMap = {}
    while queue:
        ti = queue.pop()  # take something out of queue
        listi = []
        # generate the expanded list
        for el in ti:
            listi.extend([el[0]] * el[1])
        # check for similarity to existing unique types
        simScore = [0] * nunique
        maxSimScore = -1.0
        for j in range(nunique):
            listj = uniqueExpnd[j][:]
            simScore[j] = get_overlap(listi, listj)
        if nunique > 0:
            maxSimScore = max(simScore)
            imax = simScore.index(maxSimScore)
        if maxSimScore >= sim_ratio:
            typeMap[typeList.index(ti)] = simScore.index(maxSimScore)
        else:
            uniqueTypes.append(ti[:])
            uniqueExpnd.append(listi[:])
            typeMap[typeList.index(ti)] = nunique
            nunique += 1

    # re-assign all the atom types
    nOfType = [0 for i in range(nunique)]
    for beads in beadsList:
        for node, group in beads.items():
            group["type"] = typeMap[group["type"]]
            nOfType[group["type"]] += 1

    # get average properties for the CG types
    CGmap = {}
    for i in range(nunique):
        CGmap[i] = {"mass": 0.0, "charge": 0.0}
    for beads in beadsList:
        for i, group in beads.items():
            iCGtype = group["type"]
            iId = atoms["id"][i]
            iType = atoms["type"][iId]
            mi = atoms["mass"][iId]
            qi = atoms["charge"][iId]
            CGmap[iCGtype]["mass"] += mi
            CGmap[iCGtype]["charge"] += qi
            for j in group["in"]:
                jId = atoms["id"][j]
                jType = atoms["type"][jId]
                mj = atoms["mass"][jId]
                qj = atoms["charge"][jId]
                CGmap[iCGtype]["mass"] += mj
                CGmap[iCGtype]["charge"] += qj

    for i, CGtype in CGmap.items():
        CGtype["mass"] /= nOfType[i]
        CGtype["charge"] /= nOfType[i]

    # write out summary
    # print("# Total number of CG beads: {}".format(len(beads)))
    # print("# Total number of CG types: {}".format(nunique))
    # print("               {:^5s} {:^10s} {:^10s}".format("Ncg","<mass>","<charge>"))
    # for i in range(nunique):
    #    print("-CG Type {:>3d}: {:^5d} {:>10.3f} {:>10.3f}".format(i+1,nOfType[i],CGmap[i]['mass'],CGmap[i]['charge']))

    return atoms, beadsList


# ==================================================================
#  AUX: assign_CG_types
# ==================================================================
def assign_CG_types(output_dir, typing, sim_ratio, atoms, beadsList):
    # aggregate and count constituent atom types
    typeList = []

    # iterate over beads and also assign indices
    beadId = 0
    for beads in beadsList:
        for node in sorted(beads.keys()):
            group = beads[node]
            beadId += 1
            group["id"] = beadId
            nodeId = atoms["id"][node]
            add_type(typing, atoms, nodeId, group)
            for j in group["in"]:
                jId = atoms["id"][j]
                add_type(typing, atoms, jId, group)
            theType = [
                [x, group["type"].count(x)] for x in set(group["type"])
            ]  # count constituent atom types
            theType = [
                el for el in sorted(theType, key=lambda pair: pair[0])
            ]  # organize in ascending order
            if theType not in typeList:
                typeList.append(theType)
            group["type"] = theType

    # sort the list of possible types for reduction
    nInType = [len(x) for x in typeList]
    typeList = [
        t for (t, n) in sorted(zip(typeList, nInType), key=lambda pair: pair[1])
    ]

    # Assign all the atom types
    nOfType = [0 for t in typeList]
    for beads in beadsList:
        for node, group in beads.items():
            iType = typeList.index(group["type"])
            group["type"] = iType
            nOfType[iType] += 1

    # Check for similarity in constitution
    uniqueTypes = []
    uniqueExpnd = []
    nunique = 0
    queue = typeList[:]
    typeMap = {}
    while queue:
        ti = queue.pop()  # take something out of queue
        listi = []
        # generate the expanded list
        for el in ti:
            listi.extend([el[0]] * el[1])
        # check for similarity to existing unique types
        simScore = [0] * nunique
        maxSimScore = -1.0
        for j in range(nunique):
            listj = uniqueExpnd[j][:]
            simScore[j] = get_overlap(listi, listj)
        if nunique > 0:
            maxSimScore = max(simScore)
            imax = simScore.index(maxSimScore)
        if maxSimScore >= sim_ratio:
            typeMap[typeList.index(ti)] = simScore.index(maxSimScore)
        else:
            uniqueTypes.append(ti[:])
            uniqueExpnd.append(listi[:])
            typeMap[typeList.index(ti)] = nunique
            nunique += 1

    # re-assign all the atom types
    nOfType = [0 for i in range(nunique)]
    for beads in beadsList:
        for node, group in beads.items():
            group["type"] = typeMap[group["type"]]
            nOfType[group["type"]] += 1

    # get average properties for the CG types
    CGmap = {}
    for i in range(nunique):
        CGmap[i] = {"mass": 0.0, "charge": 0.0}
    for beads in beadsList:
        for i, group in beads.items():
            iCGtype = group["type"]
            iId = atoms["id"][i]
            iType = atoms["type"][iId]
            mi = atoms["mass"][iId]
            qi = atoms["charge"][iId]
            CGmap[iCGtype]["mass"] += mi
            CGmap[iCGtype]["charge"] += qi
            for j in group["in"]:
                jId = atoms["id"][j]
                jType = atoms["type"][jId]
                mj = atoms["mass"][jId]
                qj = atoms["charge"][jId]
                CGmap[iCGtype]["mass"] += mj
                CGmap[iCGtype]["charge"] += qj

    for i, CGtype in CGmap.items():
        CGtype["mass"] /= nOfType[i]
        CGtype["charge"] /= nOfType[i]

    # write out summary
    fid = open(os.path.join(output_dir, "typing.summary.txt"), "w")
    fid.write("#===========================\n")
    fid.write("# Typing Summary\n")
    fid.write("#===========================\n")
    fid.write("# Total number of CG beads: {}\n".format(len(beads)))
    fid.write("# Total number of CG types: {}\n".format(nunique))
    fid.write(
        "             {:^5s} {:^10s} {:^10s}\n".format("Ncg", "<mass>", "<charge>")
    )
    for i in range(nunique):
        fid.write(
            "-CG Type {:>3d}: {:^5d} {:>10.3f} {:>10.3f}\n".format(
                i + 1, nOfType[i], CGmap[i]["mass"], CGmap[i]["charge"]
            )
        )
    fid.write("\n~~~~~CG TYPES~~~~~\n")

    for i in range(nunique):
        fid.write("Type {}: ".format(i + 1))
        for j in uniqueExpnd[i]:
            fid.write("{} ".format(j))
        fid.write("\n")

    return fid, CGmap, nOfType


# ==================================================================
#  AUX: redution_mapping
# ==================================================================
# determines the grouping of atoms that form CG beads. Information is in
# a dictionary with the following fields
# 'adj' - the adjacency list (set of atoms for bonding)
# 'in'  - constituent atoms
# 'nheavy' - number of heavy atoms in the CG atom
# 'type' - type of the CG bead
# 'coords' - coordinates of the bead
# 'id' - assigned ID
def reduction_mapping(
    logger, niter, min_level, max_level, max_size, moli, atoms, adjlist
):
    history = []
    curr = {}
    queue = []

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up initial list and mark atoms to be reduced
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logger.info(f"Initial Number of atoms: {len(moli)}")
    logger.info(f"# Beginning coarse-graining of molecule with {len(moli)} atoms")
    for i in moli:
        idi = atoms["id"][i]  # key to index in rest of atoms structure
        if atoms["priority"][idi] == -1:
            queue.append(i)
        else:
            mi = atoms["mass"][idi]
            qi = atoms["charge"][idi]
            if mi > mlight:
                curr[i] = {
                    "adj": set(adjlist[i]),
                    "in": set(),
                    "nheavy": 1,
                    "type": [],
                    "mass": mi,
                    "charge": qi,
                }
            else:
                curr[i] = {
                    "adj": set(adjlist[i]),
                    "in": set(),
                    "nheavy": 0,
                    "type": [],
                    "mass": mi,
                    "charge": qi,
                }
    logger.info(
        f"# Initial contraction of {len(queue)} atoms into {len(curr)} remaining groups"
    )

    # Perform initial contraction for negative priority atoms
    logger.info(
        f"Initial contraction consists of {len(queue)} into {len(curr)} groups\n"
    )
    for i in queue:
        neighbors = adjlist[i][:]  # determine the set of available neighbors
        mergeId = prioritize(i, neighbors, atoms, adjlist)  # find who to contract into
        neighbors.pop(
            neighbors.index(mergeId)
        )  # remove contractor from the available neighbors of contracted
        curr[mergeId]["in"].add(i)  # augment list to reflect the contraction
        add_if_heavy(mergeId, i, curr, atoms)
        curr[mergeId]["adj"].remove(
            i
        )  # remove the contracted from adjacency list of contractor
        curr[mergeId]["adj"].update(
            neighbors
        )  # resolve the adjacency list of contracted
    update_masses(curr, atoms)
    update_charge(curr, atoms)
    history.append(copy.deepcopy(curr))

    # Start coordination level reductions, contracting from degree 2 and up
    for iIter in range(niter):
        logger.info(f"Iteration {iIter + 1}")
        logger.info(f"Reduction Round {iIter + 1}.")
        logger.info(f"Initial number of groups: {len(curr)}")
        touched = set()
        for lvl in range(min_level[iIter], max_level[iIter] + 1):
            logger.info(f"# Examining nodes with degree equal to {lvl}...")
            queue = make_level_queue(curr, lvl, atoms, touched)
            tried = set()
            logger.info(f"# There are {len(queue)} nodes in the queue...")
            while queue:
                node = queue.pop(0)  # obtain index for first in queue
                if get_mass(atoms, node, curr[node]["in"]) >= max_size:
                    touched.add(node)
                else:
                    neighbors = (
                        curr[node]["adj"] - touched
                    )  # set of neighbors who are not in the touch list
                    if neighbors:
                        neighCN = [
                            len(curr[n]["adj"]) for n in neighbors
                        ]  # number of connections for neighbor (including with 'node')
                        minCN = min(neighCN)
                        if minCN == lvl:  # if no one with lower connectivity,
                            if node in tried:
                                curr, touched, queue = contract(
                                    curr,
                                    touched,
                                    queue,
                                    node,
                                    neighbors,
                                    neighCN,
                                    minCN,
                                    atoms,
                                    max_size,
                                )
                            else:
                                tried.add(node)
                                queue.append(node)  # then send to end of queue for now
                        elif minCN > lvl:  # if only higher connectivity
                            touched.add(node)
                        else:  # otherwise find all lowest
                            minCN = lvl - 1  # testing
                            curr, touched, queue = contract(
                                curr,
                                touched,
                                queue,
                                node,
                                neighbors,
                                neighCN,
                                minCN,
                                atoms,
                                max_size,
                            )

                    else:
                        touched.add(node)
                    queue = reorder_queue(queue, touched, curr, tried)
            update_masses(curr, atoms)
            update_charge(curr, atoms)
            logger.info("# Queue is exhausted and vertex groups formed...")
            logger.info(f"# There are currently {len(curr)} vertex groups...")
            logger.info(f"Reduction at level {lvl} --> {len(curr)} groups")
            history.append(copy.deepcopy(curr))

    # summary.write("\n\n")
    return curr, history
