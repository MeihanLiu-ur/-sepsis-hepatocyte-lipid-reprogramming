# -*- coding: utf-8 -*-
"""Export the 12 a-priori hepatocyte lipid programs to machine-readable files.

The definitions live in `scripts/lib_sepsis.py` (dict ``LIPID_PROGRAMS``) and are the
single source of truth used for every AUCell score reported in the manuscript.

This script parses that file with ``ast`` (no third-party imports, no heavy
dependencies) and writes:

    genesets/programs_long.csv   program, gene        (one row per gene; 12 programs)
    genesets/programs_wide.csv   program, n_genes, genes (semicolon-joined)
    genesets/programs.gmt        standard GMT, 1 line per program, description = n_genes

Usage:
    python3 export_genesets.py
"""
from __future__ import annotations

import ast
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "scripts", "lib_sepsis.py")
OUT = os.path.join(HERE, "genesets")


def load_programs(path: str) -> dict[str, list[str]]:
    """Extract LIPID_PROGRAMS from lib_sepsis.py without importing it."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "LIPID_PROGRAMS":
                    return ast.literal_eval(node.value)
    raise SystemExit(f"LIPID_PROGRAMS not found in {path}")


def main() -> None:
    programs = load_programs(LIB)
    os.makedirs(OUT, exist_ok=True)

    with open(os.path.join(OUT, "programs_long.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["program", "gene"])
        for p, genes in programs.items():
            for g in genes:
                w.writerow([p, g])

    with open(os.path.join(OUT, "programs_wide.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["program", "n_genes", "genes"])
        for p, genes in programs.items():
            w.writerow([p, len(genes), ";".join(genes)])

    with open(os.path.join(OUT, "programs.gmt"), "w", encoding="utf-8") as fh:
        for p, genes in programs.items():
            fh.write("\t".join([p, str(len(genes))] + genes) + "\n")

    total = sum(len(v) for v in programs.values())
    print(f"{len(programs)} programs, {total} gene entries written to {OUT}")
    for p, genes in programs.items():
        print(f"  {p:<22} {len(genes):>3} genes")


if __name__ == "__main__":
    main()
