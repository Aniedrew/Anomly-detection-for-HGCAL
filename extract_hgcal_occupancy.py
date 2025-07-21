import ROOT
from collections import defaultdict
import sys
import os
import csv

# Read input/output file paths from command-line arguments
if len(sys.argv) != 3:
    print("Usage: python extract_hgcal_occupancy.py <input_root_file> <output_csv_file>")
    sys.exit(1)

INPUT_FILE = sys.argv[1]
OUTPUT_CSV = sys.argv[2]

# Open the input ROOT file and get the TTree
input_file = ROOT.TFile.Open(INPUT_FILE)
tree = input_file.Get("detidGeometryHGCAL/HGCALDigiTree")

# Map to store occupancy counts and wafer type
# structure: occupancy_map[(lumi, layer, u, v)] = [count, type]
occupancy_map = {}

# Loop over all entries in the tree
for i, entry in enumerate(tree):
    key = (entry.lumi, entry.layer, entry.waferU, entry.waferV)
    if i % 1000000 == 0:
        print(f"Entry key: lumi={entry.lumi}, layer={entry.layer}, waferU={entry.waferU}, waferV={entry.waferV}, type={entry.type}")
    if key in occupancy_map:
        occupancy_map[key][0] += 1
    else:
        occupancy_map[key] = [1, entry.type]  # [count, type]

# Write to CSV
with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["lumi", "layer", "waferU", "waferV", "count", "type"])
    for (lumi, layer, u, v), (count, wafer_type) in occupancy_map.items():
        writer.writerow([lumi, layer, u, v, count, wafer_type])

print(f"Wrote occupancy data to {OUTPUT_CSV}")

