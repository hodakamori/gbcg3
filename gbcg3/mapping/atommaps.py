from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from gbcg3.gbcg.helpers import is_number
from gbcg3.utils.element_db import mass2el
from typing_extensions import Literal


@dataclass
class AtomMaps:
    mass_map: Dict[int, float] = field(default_factory=dict)
    name_mapfile: Path = None
    priority_mapfile: Path = None
    cgtypes: Dict[int, str] = field(default_factory=dict)

    def _process_mapfile(
        self, mapfile: Path, map_type: Literal["name", "priority"]
    ) -> Dict[int, float]:
        the_map = {}
        if mapfile is not None:
            fid = open(mapfile, "r")
            lines = [line.strip().split() for line in fid]
            for line in lines:
                if is_number(line[1]):
                    the_map[int(line[0])] = float(line[1])
                else:
                    the_map[int(line[0])] = line[1]
        else:
            if map_type == "priority":
                # create priority dictionary based on mass
                for i, m in self.mass_map.items():
                    if round(m) <= 3.1:
                        the_map[i] = -1
                    else:
                        the_map[i] = 1.0 / round(m)
                        # the_map[i] = 1
            elif map_type == "name":
                for i, m in self.mass_map.items():
                    the_map[i] = (
                        mass2el[m]
                        if m in mass2el
                        else mass2el[min(mass2el.keys(), key=lambda k: abs(k - m))]
                    )
        return the_map

    def __post_init__(self) -> None:
        self.names_map = self._process_mapfile(self.name_mapfile, "name")
        self.priority_map = self._process_mapfile(self.priority_mapfile, "priority")
