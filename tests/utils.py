import difflib
from typing import List


def compare_files(test_file_path: str, expect_file_path: str) -> List[str]:
    with open(test_file_path) as f1, open(expect_file_path) as f2:
        text1_lines = f1.readlines()
        text2_lines = f2.readlines()
        d = difflib.Differ()

        diff = d.compare(text1_lines, text2_lines)
        differences = [l for l in diff if l.startswith("- ") or l.startswith("+ ")]

        return differences
