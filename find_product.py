import numpy as np
from FIND_PRODUCT.functions import *

# ----------------- User settings -----------------
elements = ["C", "N", "C", "H", "H", "H", "I"]   # must match atom order
print_interval = 5
TRAJ_FILE = "T10.out"
PRODUCTS_XYZ = "products_to_search.xyz"
CUTOFF_SCALE = 1.3
# ----------------- Main flow -----------------

if __name__ == "__main__":
    product_index = build_product_index_and_write_files(TRAJ_FILE, elements,
                                                       print_interval=print_interval,
                                                       C=CUTOFF_SCALE)
    search_results = search_products_file_across_trajectories(PRODUCTS_XYZ, product_index, elements, C=CUTOFF_SCALE)

