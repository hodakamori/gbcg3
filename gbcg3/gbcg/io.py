import datetime
import os
import numpy as np
from gbcg3.gbcg.helpers import wrap_into_box, approximate_sigma, compute_angle

version = 1


def write_data_file(ftyp, output_dir, atoms, CGmols, box, nOfType, CGmap) -> None:
    # acquire system information
    nCgType = len(nOfType)
    nBonType = 0
    nAngType = 0
    nDihType = 0

    # total number of atoms
    natm = sum([len(i) for i in CGmols])

    # form adjacency matrix and bonds list and get size info
    adjmat = np.zeros([natm + 1, natm + 1])
    bonds = []
    bond2typ = []
    btypes = []
    bavg = []  # stores average bond length and number of occurences per bond type
    id2typ = {}
    coords = {}
    savg = [[0.0, 0.0] for i in range(nCgType)]
    for mol in CGmols:
        for i in sorted(mol.keys()):
            bead = mol[i]
            radius = approximate_sigma(i, bead["in"], atoms)
            idi = bead["id"]
            ityp = bead["type"]
            savg[ityp][0] += radius
            savg[ityp][1] += 1.0
            coords[idi] = bead["coords"][:]
            ri = bead["coords"][:]
            id2typ[idi] = ityp
            for j in bead["adj"]:
                idj = mol[j]["id"]
                jtyp = mol[j]["type"]
                rj = mol[j]["coords"][:]
                adjmat[idi, idj] = 1
                adjmat[idj, idi] = 1
                if ityp < jtyp:
                    btype = (ityp, jtyp)
                else:
                    btype = (jtyp, ityp)
                if btype not in btypes:
                    btypes.append(btype)
                    bavg.append([0.0, 0.0])
                if idi < idj:
                    bonds.append((idi, idj))
                    dr = ri - rj
                    bond2typ.append(btypes.index(btype))
                    bavg[btypes.index(btype)][0] += np.linalg.norm(dr)
                    bavg[btypes.index(btype)][1] += 1.0
    nbonds = len(bonds)

    # angles
    angles = []
    ang2typ = []
    atypes = []
    aavg = []
    for i, j in bonds:
        (potkj,) = np.nonzero(adjmat[j, :])
        (potki,) = np.nonzero(adjmat[i, :])
        kjs = [k for k in potkj if k != i]
        kis = [k for k in potki if k != j]
        ityp = id2typ[i]
        jtyp = id2typ[j]
        ri = coords[i][:]
        rj = coords[j][:]

        # check connections from j --> k (i,j,k) or (k,j,i)
        for k in kjs:
            rk = coords[k][:]
            ktyp = id2typ[k]
            if i < k:
                ang = (i, j, k)
            else:
                ang = (k, j, i)
            if ang not in angles:
                if ityp < ktyp:
                    atype = (ityp, jtyp, ktyp)
                else:
                    atype = (ktyp, jtyp, ityp)
                if atype not in atypes:
                    atypes.append(atype)
                    aavg.append([0.0, 0.0])
                theta = compute_angle(ri, rj, rk)
                angles.append(ang)
                ang2typ.append(atypes.index(atype))
                aavg[atypes.index(atype)][0] += theta
                aavg[atypes.index(atype)][1] += 1.0

        # check connections from i --> k (j,i,k) or (k,i,j)
        for k in kis:
            rk = coords[k][:]
            ktyp = id2typ[k]
            if j < k:
                ang = (j, i, k)
            else:
                ang = (k, i, j)
            if ang not in angles:
                if jtyp < ktyp:
                    atype = (jtyp, ityp, ktyp)
                else:
                    atype = (ktyp, ityp, jtyp)
                if atype not in atypes:
                    atypes.append(atype)
                    aavg.append([0.0, 0.0])
                theta = compute_angle(ri, rj, rk)
                angles.append(ang)
                ang2typ.append(atypes.index(atype))
                aavg[atypes.index(atype)][0] += theta
                aavg[atypes.index(atype)][1] += 1.0

    nangles = len(angles)
    deg2 = np.dot(adjmat[:, :], adjmat[:, :])

    # dihedrals
    # XXX right now excludes connections that go to any of the constituent atom angles
    # may need to add impropers
    dihedrals = []
    dih2typ = []
    dtypes = []
    for i, j, k in angles:
        (potlk,) = np.nonzero(adjmat[k, :])
        (potli,) = np.nonzero(adjmat[i, :])
        lks = [l for l in potlk if l not in [i, j]]
        lis = [l for l in potli if l not in [j, k]]
        ityp = id2typ[i]
        jtyp = id2typ[j]
        ktyp = id2typ[k]

        # check (i,j,k,l) or (l,k,j,i)
        for l in lks:
            ltyp = id2typ[l]
            if i < l:
                dih = (i, j, k, l)
            else:
                dih = (l, k, j, i)
            if dih not in dihedrals:
                if ityp < ltyp:
                    dtype = (ityp, jtyp, ktyp, ltyp)
                elif ltyp < ityp:
                    dtype = (ltyp, ktyp, jtyp, ityp)
                else:
                    if jtyp < ktyp:
                        dtype = (ityp, jtyp, ktyp, ltyp)
                    else:
                        dtype = (ltyp, ktyp, jtyp, ityp)
                if dtype not in dtypes:
                    dtypes.append(dtype)
                dihedrals.append(dih)
                dih2typ.append(dtypes.index(dtype))

        # check (l,i,j,k) or (k,j,i,l)
        for l in lis:
            ltyp = id2typ[l]
            if l < k:
                dih = (l, i, j, k)
            else:
                dih = (k, j, i, l)
            if dih not in dihedrals:
                if ltyp < ktyp:
                    dtype = (ltyp, ityp, jtyp, ktyp)
                elif ktyp < ltyp:
                    dtype = (ktyp, jtyp, ityp, ltyp)
                else:
                    if ityp < jtyp:
                        dtype = (ltyp, ityp, jtyp, ktyp)
                    else:
                        dtype = (ktyp, jtyp, ityp, ltyp)
                if dtype not in dtypes:
                    dtypes.append(dtype)
                dihedrals.append(dih)
                dih2typ.append(dtypes.index(dtype))

    deg3 = np.dot(deg2, adjmat)
    ndihedrals = len(dihedrals)

    # write out the data file
    nBonType = len(btypes)
    nAngType = len(atypes)
    nDihType = len(dtypes)
    fid = open(os.path.join(output_dir, "sys.cg.data"), "w")
    fid.write(
        "LAMMPS data file via GBCG Mapping, version {} {}\n\n".format(
            version, str(datetime.date.today())
        )
    )
    fid.write("{} atoms\n".format(natm))
    fid.write("{} atom types\n".format(nCgType))
    fid.write("{} bonds\n".format(nbonds))
    fid.write("{} bond types\n".format(nBonType))
    fid.write("{} angles\n".format(nangles))
    fid.write("{} angle types\n".format(nAngType))
    fid.write("{} dihedrals\n".format(ndihedrals))
    fid.write("{} dihedral types\n\n".format(nDihType))

    # box
    fid.write("{} {} xlo xhi\n".format(box[0, 0], box[0, 1]))
    fid.write("{} {} ylo yhi\n".format(box[1, 0], box[1, 1]))
    fid.write("{} {} zlo zhi\n\n".format(box[2, 0], box[2, 1]))

    # Masses
    fid.write("Masses\n\n")
    for i, CGtype in CGmap.items():
        fid.write(
            "{} {:>10.4f}\n".format(i + 1, CGtype["mass"])
        )  # add one from internal typing

    # Pair Coeffs
    fid.write("\nPair Coeffs\n\n")
    for i, CGtype in CGmap.items():
        savg[i][0] /= savg[i][1]
        fid.write("{:<5d} {:>10.4f} {:>10.4f}\n".format(i + 1, 0.1, savg[i][0]))

    # Bond Coeffs
    fid.write("\nBond Coeffs\n\n")
    ftyp.write("\n~~~~~BOND TYPES~~~~~\n")
    for i, btype in enumerate(btypes):
        bavg[i][0] /= bavg[i][1]
        fid.write("{} {:>8.3f} {:>8.3f}\n".format(i + 1, 100.0, bavg[i][0]))
        ftyp.write("Type {}: ({})--({})\n".format(i + 1, btype[0] + 1, btype[1] + 1))

    # Angle Coeffs
    ftyp.write("\n~~~~~ANGLE TYPES~~~~~\n")
    fid.write("\nAngle Coeffs\n\n")
    for i, atype in enumerate(atypes):
        aavg[i][0] /= aavg[i][1]
        fid.write("{} {} {:>10.4f}\n".format(i + 1, 25.0, aavg[i][0]))
        ftyp.write(
            "Type {}: ({})--({})--({})\n".format(
                i + 1, atype[0] + 1, atype[1] + 1, atype[2] + 1
            )
        )

    # Dihedral Coeffs
    fid.write("\nDihedral Coeffs\n\n")
    ftyp.write("\n~~~~~DIHEDRAL TYPES~~~~~\n")
    for i, dtype in enumerate(dtypes):
        fid.write(
            "{} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}\n".format(
                i + 1, 0.0, 0.0, 0.0, 0.0
            )
        )
        ftyp.write(
            "Type {}: ({})--({})--({})--({})\n".format(
                i + 1, dtype[0] + 1, dtype[1] + 1, dtype[2] + 1, dtype[3] + 1
            )
        )

    # CHECK CHARGES
    qtot = 0.0
    ntot = 0.0
    for j, mol in enumerate(CGmols):
        qmol = 0.0
        for i in sorted(mol.keys()):
            bead = mol[i]
            ityp = bead["type"]
            qmol += CGmap[ityp]["charge"]
            qtot += CGmap[ityp]["charge"]
            ntot += 1.0
        print("# Charge for molecule {}: {}".format(j, qmol))
    qavg = qtot / ntot

    # Atoms
    fid.write("\nAtoms\n\n")
    for j, mol in enumerate(CGmols):
        for i in sorted(mol.keys()):
            bead = mol[i]
            idi = bead["id"]
            ityp = bead["type"]
            ri = bead["coords"].copy()
            # ri  = wrap_into_box(ri,box)
            qi = CGmap[ityp]["charge"] - qavg
            fid.write(
                "{} {} {} {:>8.3f} {:>15.5f} {:>15.5f} {:>15.5f} 0 0 0\n".format(
                    idi,
                    j + 1,
                    ityp + 1,
                    qi,
                    bead["coords"][0],
                    bead["coords"][1],
                    bead["coords"][2],
                )
            )

    # Bonds
    if len(bonds) > 0:
        fid.write("\nBonds\n\n")
        for i, (bond, btype) in enumerate(zip(bonds, bond2typ)):
            fid.write("{} {} {} {}\n".format(i + 1, btype + 1, bond[0], bond[1]))

    # Angles
    if len(angles) > 0:
        fid.write("\nAngles\n\n")
        for i, (angle, atype) in enumerate(zip(angles, ang2typ)):
            fid.write(
                "{} {} {} {} {}\n".format(
                    i + 1, atype + 1, angle[0], angle[1], angle[2]
                )
            )

    # Dihedrals
    if len(dihedrals) > 0:
        fid.write("\nDihedrals\n\n")
        for i, (dihedral, dtype) in enumerate(zip(dihedrals, dih2typ)):
            fid.write(
                "{} {} {} {} {} {}\n".format(
                    i + 1, dtype + 1, dihedral[0], dihedral[1], dihedral[2], dihedral[3]
                )
            )

    fid.close()
    ftyp.close()


def write_CG_lammpstrj(CGatoms, fid, timestep, box) -> None:
    N = len(CGatoms)
    # write the header
    fid.write("ITEM: TIMESTEP\n\t{}\n".format(timestep))
    fid.write("ITEM: NUMBER OF ATOMS\n\t{}\n".format(N))
    fid.write("ITEM: BOX BOUNDS pp pp pp\n")
    fid.write("{:>10.4f}{:>10.4f}\n".format(box[0, 0], box[0, 1]))
    fid.write("{:>10.4f}{:>10.4f}\n".format(box[1, 0], box[1, 1]))
    fid.write("{:>10.4f}{:>10.4f}\n".format(box[2, 0], box[2, 1]))
    fid.write("ITEM: ATOMS id type x y z\n")

    # write the coordinates
    for i in sorted(CGatoms.keys()):
        CGatom = CGatoms[i]
        crds = CGatom["coords"][:].copy()
        crds = wrap_into_box(crds, box)
        typ = CGatom["type"]
        Id = CGatom["id"]
        fid.write(
            "{:<10d}{:<4d}{:>12.4f}{:>12.4f}{:>12.4f}\n".format(
                Id, typ + 1, crds[0], crds[1], crds[2]
            )
        )


def write_CG_map(CGatoms, atoms, map, fid) -> None:
    N = len(CGatoms)
    mass = 0.0
    for i in sorted(CGatoms.keys()):
        fid.write(
            "{:>6d} {:>5d} {:>10.5f} {:>8.3f} {:>6d}".format(
                CGatoms[i]["id"],
                CGatoms[i]["type"] + 1,
                CGatoms[i]["mass"],
                CGatoms[i]["charge"],
                i,
            )
        )
        for j in sorted(CGatoms[i]["in"]):
            fid.write(" {:>6d}".format(j))
        fid.write("\n")


def write_CG_pdb(CGatoms, atoms, map, fid) -> None:
    N = len(CGatoms)
    fid.write("COMPND    UNNAMED\n")
    fid.write("AUTHOR    GENEATED BY CG_mapping.py\n")
    imap = {}
    cnt = 0
    for i in sorted(CGatoms.keys()):
        cnt += 1
        imap[i] = cnt
        CGatom = CGatoms[i]
        ptr = atoms["id"][i]
        crds = CGatom["coords"][:]
        typ = atoms["type"][ptr]
        lbl = map[typ]
        fid.write(
            "{:<6s}{:>5d} {:<4s}{:1s}{:>3s}  {:>4d}{:<1s}   {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}      {:>4s}{:<s}\n".format(
                "HETATM",
                imap[i],
                lbl[:3],
                " ",
                "UNL",
                1,
                " ",
                crds[0],
                crds[1],
                crds[2],
                1.00,
                0.00,
                "",
                lbl[0],
            )
        )
    for i in sorted(CGatoms.keys()):
        CGatom = CGatoms[i]
        ptr = atoms["id"][i]
        crds = CGatom["coords"][:]
        typ = atoms["type"][ptr]
        lbl = map[typ]
        fid.write("{:<6s}{:>5d}".format("CONECT", imap[i]))
        for cnct in CGatom["adj"]:
            fid.write("{:>5d}".format(imap[cnct]))
        fid.write("\n")


def write_CG_xyz(CGatoms, atoms, map, fid) -> None:
    N = len(CGatoms)
    fid.write("{:<} \n\n".format(N))
    for i, CGatom in CGatoms.items():
        ptr = atoms["id"][i]
        crds = CGatom["coords"][:]
        typ = atoms["type"][ptr]
        lbl = map[typ]
        fid.write(
            "{:<10s} {:>15.3f} {:>15.3f} {:>15.3f}\n".format(
                lbl, crds[0], crds[1], crds[2]
            )
        )


def write_xyz(atoms, map, fid) -> None:
    N = len(atoms["type"])
    fid.write("{:<} \n\n".format(N))
    for atm, ptr in atoms["id"].items():
        crds = atoms["coords"][ptr][:]
        typ = atoms["type"][ptr]
        lbl = map[typ]
        fid.write(
            "{:<10s} {:>15.3f} {:>15.3f} {:>15.3f}\n".format(
                lbl, crds[0], crds[1], crds[2]
            )
        )


def write_groups(output_dir, i, beads, atoms, map) -> None:
    fid = open(os.path.join(output_dir, f"mol_{i}.groups.dat"), "w")
    for node, group in beads.items():
        nhvy = group["nheavy"]
        nall = len(group["in"]) + 1
        ptr = atoms["id"][node]
        typ = atoms["type"][ptr]
        lbl = map[typ]
        fid.write("{}({}) {} {}-- ".format(node, lbl, nall, nhvy))
        for neighbor in group["adj"]:
            ptr = atoms["id"][neighbor]
            typ = atoms["type"][ptr]
            lbl = map[typ]
            fid.write("{}({}) ".format(neighbor, lbl))
        fid.write("\n\t >> {} : ".format(len(group["in"])))
        for atom in group["in"]:
            ptr = atoms["id"][atom]
            typ = atoms["type"][ptr]
            lbl = map[typ]
            fid.write("{}({}) ".format(atom, lbl))
        fid.write("\n")
    fid.close()
