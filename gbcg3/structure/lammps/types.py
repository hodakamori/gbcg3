from typing import Dict, List, TypedDict


class Atoms(TypedDict):
    id: Dict[int, int]
    type: List[int]
    charge: List[float]
    coords: List[List[float]]
    force: List[List[float]]
    mass: List[float]
    priority: List[float]
