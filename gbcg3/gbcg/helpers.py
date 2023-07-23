from math import floor

import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


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


def compute_angle(ri, rj, rk):
    rji = ri - rj
    rjk = rk - rj
    cost = np.dot(rji, rjk) / np.linalg.norm(rji) / np.linalg.norm(rjk)
    theta = np.arccos(cost) * 180.0 / np.pi
    return theta


def wrap_into_box(r, box):
    # shift origin to 0 0 0 to do wrapping
    L = np.array([x[1] - x[0] for x in box])
    r -= box[:, 0]
    shift = np.array([floor(ri / Li) for ri, Li in zip(r, L)])
    r -= shift * L
    # shift back
    r += box[:, 0]
    return r


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


def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)


def power_iteration(A):
    n, d = A.shape
    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)
    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < 1e-15:
            break

        v = v_new
        ev = ev_new

    return ev_new, v_new
