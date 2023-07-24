import argparse
import os

from gbcg3.utils.logger import print_status


# ==================================================================
#  AUX: create_parser
# ==================================================================
def create_parser():
    parser = argparse.ArgumentParser(
        description='Generates a graph-based coarse-grained (GBCG) particle mapping based on a specified degree of connectivity, i.e., the progressive grouping scheme.\
  Details of the formalism can be found in the publication: Michael A. Webb and Juan J. de Pablo, "A systematic graph-based approach to molecular coarse-graining," J. Comp. Theo. Chem., XX, 2018. See below for various available options.'
    )

    # ARGUMENTS
    parser.add_argument(
        "-data",
        dest="datafile",
        default=None,
        help="Name for the lammps data file. This is required for obtaining connectivity information. (default: none)",
    )

    parser.add_argument(
        "-traj",
        default=None,
        help='Trajectory file(s) with atom coordinates in LAMMPS format. Multiple files can be included in a quoted list with space separation between the file names. (example: "poly.lammpstrj" or "traj.1 traj.2") Note: trajectory files are not actually used in the mapping scheme. Multiple files will be read simultaneously and are intended to for different groups of atoms rather than a sequence for the same group of atoms',
    )

    parser.add_argument(
        "-min_level",
        dest="min_level",
        default=2,
        help='Minimum degree level for graph coarsening.\
          After initial reduction, we start at this level and proceed up to the maximum degree level\
          to subsume neighboring nodes. Example: -min_level 2 will start with nodes of connectivity 2, then 3, then ... (default: 2) It is also possible to specify this as a quoted list with as many elements as there are rounds in the coarsening to alter the minimum degree from round to round. Example: -min_level "2 3 2" will set the minimum degree to 2, then 3, then 2 for 3 rounds of coarsening',
    )

    parser.add_argument(
        "-max_level",
        dest="max_level",
        default=3,
        help="Maximum degree level for graph coarsening, as explained in the options for -min_level. \
          (default: 3) It is also possible to specify this as a quoated list with as many elements as there are rounds in the coarsening.",
    )

    parser.add_argument(
        "-niter",
        dest="niter",
        default=1,
        help="Number of coarsening iterations. (default: 1) ",
    )

    parser.add_argument(
        "-samp_freq",
        dest="samp_freq",
        default=1,
        help="Frequency (in terms of frames) to map configurations.\
                              (default = 1)",
    )

    parser.add_argument(
        "-max_samp",
        dest="max_samp",
        default=1,
        help="Maximum number of configurations to map.\
                              (default = 1)",
    )

    parser.add_argument(
        "-max_size",
        dest="max_size",
        default=None,
        help="Maximum mass of a CG bead. (default = None)",
    )

    parser.add_argument(
        "-similar",
        dest="simrat",
        default=1.0,
        help="Threshold ratio for evaluating similarity among CG atom types. If the overlap in constituents for two CG atoms types exceeds this number, they will be treated as a single type. (default = 1.0)",
    )

    parser.add_argument(
        "-typing",
        dest="typing",
        default="all",
        help='Typing method for CG atom types. Option "all" will use all of the contituent atom types. Option "heavy" will only use the heavy atoms for the CG typing. (default = "all")',
    )

    parser.add_argument(
        "-tmap",
        dest="tmap",
        default=None,
        help="File name for listing names for lammps for lammps atom types. If none supplied, then the mass from the data file is used to determine atom names. (default: none)",
    )

    parser.add_argument(
        "-pmap",
        dest="pmap",
        default=None,
        help="File name to list initial priority values for lammps atom types. Atom types assigned a value of -1 will be contracted before any coarse-graining iterations. If no file is supplied, then the mass from the data file is used for this purpose, and atoms with mass < -3.5 are given a -1 value. (default: none)",
    )

    parser.add_argument(
        "-weights",
        dest="weights",
        default="none",
        help='Style of vertex weights to be used in adjacency matrix. Available options include "mass" and "diff"  (default = "none")',
    )

    return parser


# ==================================================================
#  AUX: convert_args
# ==================================================================
def convert_args(args):
    # SET FILES
    files = {}
    files["traj"] = [i for i in args.traj.split()]
    files["data"] = args.datafile
    files["names"] = args.tmap
    files["pmap"] = args.pmap
    files["summary"] = open("summary.txt", "w")

    # SET ARGUMENTS TO OPTIONS STRUCTURE
    options = {}
    options["min_level"] = [int(lvl) for lvl in args.min_level.split()]
    options["max_level"] = [int(lvl) for lvl in args.max_level.split()]
    options["niter"] = int(args.niter)
    options["weightStyle"] = args.weights

    # EXECUTE CONSISTENCY CHECKS
    if options["niter"] != len(options["min_level"]):
        options["min_level"] = [options["min_level"][0]] * options["niter"]
    if options["niter"] != len(options["max_level"]):
        options["max_level"] = [options["max_level"][0]] * options["niter"]
    options["sfreq"] = int(args.samp_freq)
    options["max_samp"] = int(args.max_samp)
    options["sim_ratio"] = float(args.simrat)
    options["typing"] = args.typing
    if args.max_size is not None:
        options["max_size"] = float(args.max_size)
    else:
        options["max_size"] = float("inf")

    # PRINT OUT CONFIRMATION OF PARAMETERS
    print_status("Processing input")
    print("# Trajectory file(s): ", files["traj"])
    print("# Mapping frequency set to ", repr(options["sfreq"]), " frames...")
    print(
        "# Mapping to include up to ", repr(options["max_samp"]), " configurations..."
    )
    print("# Minimum degree for coarsening: ", repr(options["min_level"]), " ...")
    print("# Maximum degree for coarsening: ", repr(options["max_level"]), " ...")
    print("# Number of coarsening rounds set to ", repr(options["niter"]), " ...")

    return (files, options)
