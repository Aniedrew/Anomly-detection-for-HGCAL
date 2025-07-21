import csv
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ====== Configurable paths ======
INPUT_CSV = "/eos/user/q/qingxian/condor_output/hgcal_digi_occupancy_all.csv"  # Path to the input occupancy file
RD2F_TEMPLATE_FILE = "/eos/user/q/qingxian/condor_output/rd2f_template_d2_values.txt"  # File containing all valid d^2 values
OUTPUT_CSV = "/eos/user/q/qingxian/SWAN_projects/AE_RDF/hgcal_rdf_training_layer1.csv"

# ====== Load allowed d² values ======
def load_allowed_d2(file_path):
    with open(file_path, 'r') as f:
        allowed_d2 = set(int(line.strip()) for line in f if line.strip().isdigit())
    return allowed_d2

# ====== Normalize wafer occupancy counts ======
def normalize_occupancy(wafer_counts, norm_type):
    values = np.array(list(wafer_counts.values()), dtype=np.float32)
    if norm_type == "minmax":
        min_v, max_v = values.min(), values.max()
        if max_v == min_v:
            return {k: 0.0 for k in wafer_counts}
        return {k: (v - min_v) / (max_v - min_v) for k, v in wafer_counts.items()}
    elif norm_type == "max":
        max_v = values.max()
        if max_v == 0:
            return {k: 0.0 for k in wafer_counts}
        return {k: v / max_v for k, v in wafer_counts.items()}
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

# ====== Compute RDF vector ======
def compute_rdf_vector(norm_counts, allowed_d2):
    rdf_vector = defaultdict(float)
    keys = list(norm_counts.keys())
    for i in range(len(keys)):
        (u1, v1) = keys[i]
        w1 = norm_counts[keys[i]]
        for j in range(i, len(keys)):  # allow j == i here
            (u2, v2) = keys[j]
            w2 = norm_counts[keys[j]]
            du = u1 - u2
            dv = v1 - v2
            d2 = du**2 + dv**2 - du * dv
            if d2 in allowed_d2:
                if i == j:
                    rdf_vector[d2] += w1 * w2  # self term
                else:
                    rdf_vector[d2] += 2 * w1 * w2  # cross terms counted twice symmetrically
    vec = [rdf_vector.get(d2, 0.0) for d2 in sorted(allowed_d2)]
    return vec

# ====== Process one lumi block for layer 1 ======
def process_lumi_block(records, allowed_d2):
    # Separate wafer by type
    type_dict = {0: {}, 1: {}, 2: {}}
    for row in records:
        u = int(row['waferU'])
        v = int(row['waferV'])
        count = float(row['count'])
        t = int(row['type'])
        if t in type_dict:
            type_dict[t][(u, v)] = count

    rdf_vector_total = np.zeros(len(allowed_d2), dtype=np.float32)
    for t in [0, 1, 2]:
        norm_type = "minmax" if t in [0, 1] else "max"
        if len(type_dict[t]) == 0:
            continue
        norm_counts = normalize_occupancy(type_dict[t], norm_type)
        rdf_vector_total += np.array(compute_rdf_vector(norm_counts, allowed_d2), dtype=np.float32)

    return rdf_vector_total

# ====== Main pipeline ======
def main():
    allowed_d2 = load_allowed_d2(RD2F_TEMPLATE_FILE)

    # Organize data by lumi
    lumi_blocks = defaultdict(list)
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['layer']) == 1:
                lumi_blocks[int(row['lumi'])].append(row)

    d2_list_sorted = sorted(allowed_d2)

    with open(OUTPUT_CSV, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['lumi'] + [f'd2_{d2}' for d2 in d2_list_sorted])
        for lumi in tqdm(sorted(lumi_blocks)):
            vec = process_lumi_block(lumi_blocks[lumi], allowed_d2)
            writer.writerow([lumi] + vec.tolist())

    print(f"Output written to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

