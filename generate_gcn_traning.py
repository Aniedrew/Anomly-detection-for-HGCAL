import csv
import numpy as np
from collections import defaultdict
from tqdm import tqdm

INPUT_CSV = "/eos/user/q/qingxian/condor_output/hgcal_digi_occupancy_all.csv"
OUTPUT_CSV = "/eos/user/q/qingxian/SWAN_projects/AE_RDF/hgcal_gcn_training_layer1_combined.csv"

def normalize_occupancy(wafer_counts, norm_type):
    values = np.array(list(wafer_counts.values()), dtype=np.float32)
    if len(values) == 0:
        return {}
    if norm_type == "minmax":
        min_v, max_v = values.min(), values.max()
        if max_v == min_v:
            # Avoid division by zero if all values are the same
            return {k: 0.0 for k in wafer_counts}
        return {k: (v - min_v) / (max_v - min_v) for k, v in wafer_counts.items()}
    elif norm_type == "max":
        max_v = values.max()
        if max_v == 0:
            # Avoid division by zero if max is zero
            return {k: 0.0 for k in wafer_counts}
        return {k: v / max_v for k, v in wafer_counts.items()}
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

def collect_all_wafer_keys():
    """
    Collect all unique (u, v) wafer coordinates for layer==1 across the whole dataset.
    Returns:
        wafer_list (list of tuple): Sorted list of all wafer coordinates (u, v).
    """
    wafer_set = set()
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['layer']) == 1:
                u = int(row['waferU'])
                v = int(row['waferV'])
                wafer_set.add((u, v))
    wafer_list = sorted(wafer_set)
    return wafer_list

def process_lumi_block(records):
    """
    Process one lumi block's wafer occupancy records.
    Normalize occupancy counts by type, then combine the normalized counts by summation.
    
    Args:
        records (list of dict): Rows of wafer data for a single lumi.
    Returns:
        combined (dict): Combined normalized occupancy {(u,v): float}
    """
    # Separate wafer counts by type: 0, 1, 2
    type_dict = {0: {}, 1: {}, 2: {}}
    for row in records:
        u = int(row['waferU'])
        v = int(row['waferV'])
        count = float(row['count'])
        t = int(row['type'])
        if t in type_dict:
            type_dict[t][(u, v)] = count

    # Define normalization type for each wafer type
    norm_type_map = {0: "minmax", 1: "minmax", 2: "max"}
    norm_dicts = {}
    for t in [0, 1, 2]:
        if len(type_dict[t]) > 0:
            norm_dicts[t] = normalize_occupancy(type_dict[t], norm_type_map[t])
        else:
            norm_dicts[t] = {}

    # Combine normalized counts by summing over all types
    combined = defaultdict(float)
    for t in [0, 1, 2]:
        for pos, val in norm_dicts[t].items():
            combined[pos] += val

    return combined

def main():
    # Collect all wafer (u,v) coordinates present in layer 1
    wafer_list = collect_all_wafer_keys()
    print(f"Total unique wafers (u,v): {len(wafer_list)}")

    # Group records by lumi for layer==1
    lumi_blocks = defaultdict(list)
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['layer']) == 1:
                lumi_blocks[int(row['lumi'])].append(row)

    # Write output CSV:
    # First column: lumi id
    # Following columns: combined normalized occupancy for each wafer (u,v) in fixed order
    with open(OUTPUT_CSV, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        header = ['lumi'] + [f'u{u}_v{v}' for (u, v) in wafer_list]
        writer.writerow(header)

        for lumi in tqdm(sorted(lumi_blocks)):
            combined = process_lumi_block(lumi_blocks[lumi])
            row = [lumi]
            for pos in wafer_list:
                row.append(combined.get(pos, 0.0))
            writer.writerow(row)

    print(f"Saved combined GCN training CSV to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

