import re
import numpy as np

def parse_trajectory_file(filename,natoms):
    trajectories = []
    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Detect a new trajectory block
        if "TRAJECTORY NUMBER" in line:
            traj = {}

            # ---- 1. trajectory number ----
            m = re.search(r"TRAJECTORY NUMBER\s+(\d+)", line)
            traj_number = int(m.group(1))
            traj["trajectory_number"] = traj_number

            i += 1

            # ---- 2. cycle count & time ----
            m = re.search(r"CYCLE COUNT IS:\s+(\d+).+TIME:\s+([0-9.E+-]+)", lines[i])
            traj["cycle_count"] = int(m.group(1))
            traj["time"] = float(m.group(2))

            i += 1

            # ---- 3. random number line ----
            traj["random_numbers"] = lines[i].strip()
            i += 1

            # ---- 4. energies ----
            kinetic = float(lines[i].split()[2])
            potential = float(lines[i].split()[5])
            traj["kinetic_energy"] = kinetic
            traj["potential_energy"] = potential
            i += 1

            total = float(lines[i].split()[2])
            traj["total_energy"] = total
            i += 1

          # ---- 5. Q/P table ----
            Q = []
            P = []
            i += 1  # skip header line 'Q   P'

            # read until 'ATOMS' appears
            for _ in range(natoms):
                  if i >= n:
                        break   # safety check if file ends unexpectedly

                  parts = lines[i].split()

                  if len(parts) != 6:
                     raise ValueError(f"Expected 6 numbers for Q,P but got {len(parts)} on line {i}: {parts}")

                # Extract Q and P for this atom
                  q1, q2, q3, p1, p2, p3 = map(float, parts)

                  Q.append([q1, q2, q3])
                  P.append([p1, p2, p3])

                  i += 1

            traj["Q"] = Q
            traj["P"] = P
        
            trajectories.append(traj)

        else:
            i+=1
       
    return trajectories

def dfs(start, adjacency_matrix, visited, component):
    """
    Perform DFS on an adjacency matrix starting from node 'start'.
    """
    visited[start] = True
    component.append(start)

    n = len(adjacency_matrix)

    for neighbor in range(n):
        if adjacency_matrix[start][neighbor] == 1 and not visited[neighbor]:
            dfs(neighbor, adjacency_matrix, visited, component)



def find_connected_components(adjacency_matrix):
    """
    Finds connected components (molecular fragments) in an adjacency matrix.
    """
    n = len(adjacency_matrix)
    visited = [False] * n
    all_components = []

    for atom in range(n):
        if not visited[atom]:
            component = []
            dfs(atom, adjacency_matrix, visited, component)
            all_components.append(component)

    return all_components


# Covalent radii for elements 1–118 (Å)
covalent_radii = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47,
    "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99,
    "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92,
    "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87,
    "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41,
    "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
    "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50,
    "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06, "Pa": 2.00, "U": 1.96, "Np": 1.90,
    "Pu": 1.87, "Am": 1.80, "Cm": 1.69, "Bk": 1.60, "Cf": 1.60, "Es": 1.60,
    "Fm": 1.60, "Md": 1.60, "No": 1.60, "Lr": 1.60,
    "Rf": 1.60, "Db": 1.60, "Sg": 1.60, "Bh": 1.60, "Hs": 1.60, "Mt": 1.60,
    "Ds": 1.60, "Rg": 1.60, "Cn": 1.60, "Nh": 1.60, "Fl": 1.60, "Mc": 1.60,
    "Lv": 1.60, "Ts": 1.60, "Og": 1.60
}




def adjacency_from_xyz_with_radii(coords, elements, C=1.3):
    """
    coords   : (N,3) list or numpy array of xyz coordinates
    elements : list of atomic symbols ["C", "N", "H", ...]
    C        : scaling factor (default 1.3)

    Returns: adjacency matrix (N x N)
    """
    coords = np.array(coords)
    N = len(coords)
    A = np.zeros((N, N), dtype=int)

    for i in range(N):
        for j in range(i+1, N):
            ri = covalent_radii[elements[i]]
            rj = covalent_radii[elements[j]]

            cutoff = C * (ri + rj)
            dist = np.linalg.norm(coords[i] - coords[j])

            if dist <= cutoff:
                A[i][j] = 1
                A[j][i] = 1

    return A

def normalize_components(components):
    """Normalize connectivity components to hashable, canonical form."""
    return tuple(tuple(sorted(comp)) for comp in sorted(components))

def write_xyz_block(f, coords, elements, header):
    """Write a single XYZ block to open file f (no extra blank line)."""
    N = len(elements)
    f.write(f"{N}\n")
    f.write(header + "\n")
    for elem, (x, y, z) in zip(elements, coords):
        f.write(f"{elem}  {x:.6f}  {y:.6f}  {z:.6f}\n")

# ----------------- Build product index and write per-trajectory files -----------------

def build_product_index_and_write_files(trajectory_file, elements,
                                        print_interval=5, C=1.3):
    """
    Parse trajectory file, build product_index and write per-trajectory product xyz files.
    Returns product_index dict.
    """
    traject = parse_trajectory_file(trajectory_file)
    traj_numbers = sorted({e["trajectory_number"] for e in traject})
    product_index = {trj: [] for trj in traj_numbers}

    for trj_no in traj_numbers:

        traj_entries = [e for e in traject if e["trajectory_number"] == trj_no]
        cycle_counts = sorted(e["cycle_count"] for e in traj_entries)

        outname = f"trajectory_{trj_no}_products.xyz"
        with open(outname, "w") as outfile:

            current_product = None
            start_cycle = None
            start_time = None
            Q_last = None

            for cycle in cycle_counts:

                if cycle % print_interval != 0:
                    continue

                entry = next(e for e in traj_entries if e["cycle_count"] == cycle)
                Q = np.array(entry["Q"])
                time = entry["time"]*10 # convert time in correct fs timescale. fs_time = venus_time*10 

                adj = adjacency_from_xyz_with_radii(coords=Q, elements=elements, C=C)
                components = find_connected_components(adj)
                comp_norm = normalize_components(components)

                # Product change detected
                if comp_norm != current_product:

                    # Close previous product (it ended *at this* cycle/time)
                    if current_product is not None:
                        end_cycle = cycle
                        end_time = time
                        duration = end_time - start_time

                        # append to index
                        product_index[trj_no].append({
                            "components": current_product,
                            "start_cycle": start_cycle,
                            "end_cycle": end_cycle,
                            "start_time": start_time,
                            "end_time": end_time
                        })

                        # write last geometry of previous product as xyz block
                        header = (f"components {current_product} | "
                                  f"cycle {start_cycle} to {end_cycle} | time = {duration:.6f}")
                        write_xyz_block(outfile, Q_last, elements, header)

                    # Start new product
                    current_product = comp_norm
                    start_cycle = cycle
                    start_time = time

                # update last geometry irrespective of change
                Q_last = Q.copy()

            # After looping cycles, close the final product (ends at last sampled cycle)
            if current_product is not None:
                end_cycle = cycle_counts[-1]
                end_time = next(e["time"] for e in traj_entries if e["cycle_count"] == end_cycle)*10
                duration = end_time - start_time

                product_index[trj_no].append({
                    "components": current_product,
                    "start_cycle": start_cycle,
                    "end_cycle": end_cycle,
                    "start_time": start_time,
                    "end_time": end_time
                })

                header = (f"Final product: components {current_product} | "
                          f"cycle {start_cycle} to {end_cycle} | time = {duration:.6f}")
                write_xyz_block(outfile, Q_last, elements, header)

#        print(f"Wrote product file: {outname}  (found {len(product_index[trj_no])} products)")

    return product_index

# ----------------- Read multi-frame XYZ (product file) -----------------

def read_multi_xyz(filename):
    """
    Read a multi-frame XYZ file which contains multiple successive XYZ blocks.
    Returns a list of tuples: [(elements_list, coords_array), ...]
    """
    frames = []
    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    nlines = len(lines)
    while i < nlines:
        # skip blank lines
        while i < nlines and lines[i].strip() == "":
            i += 1
        if i >= nlines:
            break

        # Expect natoms
        natom_line = lines[i].strip()
        try:
            natoms = int(natom_line)
        except ValueError:
            raise ValueError(f"Expected integer atom count at line {i+1}, got: {natom_line}")

        # Comment and atom lines
        if i + 1 >= nlines:
            raise ValueError("XYZ file truncated (missing comment line).")
        comment = lines[i + 1].rstrip("\n")
        start = i + 2
        end = start + natoms
        if end > nlines:
            raise ValueError("XYZ file truncated (missing atom lines).")
        atom_lines = lines[start:end]

        elems = []
        coords = []
        for a in atom_lines:
            parts = a.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed atom line in XYZ: '{a.strip()}'")
            elems.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        frames.append((elems, np.array(coords)))
        i = end

    return frames

# ----------------- Search across product_index -----------------

def search_products_file_across_trajectories(products_xyz_file, product_index, elements, C=1.3):
    frames = read_multi_xyz(products_xyz_file)
    results = {}

    for idx, (elems_frame, coords) in enumerate(frames, start=1):
        if elems_frame != elements:
            print(f"[WARNING] frame {idx}: element order differs from expected elements list.")

        adj = adjacency_from_xyz_with_radii(coords=coords, elements=elements, C=C)
        components = find_connected_components(adj)
        comp_norm = normalize_components(components)

        matches = []
        for trj_no, products in product_index.items():
            for prod in products:
                if prod["components"] == comp_norm:
                    duration = prod["end_time"] - prod["start_time"]
                    matches.append({
                        "traj_no": trj_no,
                        "start_cycle": prod["start_cycle"],
                        "end_cycle": prod["end_cycle"],
                        "start_time": prod["start_time"],
                        "end_time": prod["end_time"],
                        "duration": duration
                    })

        results[idx] = {"components": comp_norm, "matches": matches, "no_matches": len(matches) == 0}

        # print summary
        if matches:
            for m in matches:
                print(f"#{idx} 1 {m['duration']:.2f}")
                print("components:", comp_norm)
                print(f"no of instances: {len(matches)}")
                print(f"cycles {m['start_cycle']} -> {m['end_cycle']}\n"
                      f"time {m['start_time']:.6f} -> {m['end_time']:.6f}\nduration = {m['duration']:.2f}")
                print("\n")
        else:
            print(f"#{idx} 0")
            print("Not found.")
            print("\n")
    return results



