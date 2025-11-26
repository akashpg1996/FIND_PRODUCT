import argparse
import numpy as np
from FIND_PRODUCT.functions import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", required=True, help="Trajectory file path")
    args = parser.parse_args()

    TRAJ_FILE = args.traj 

    elements = ["C", "N", "C", "H", "H", "H", "I"]
    print_interval = 5
    PRODUCTS_XYZ = "/scratch/ssd_4TB/akash/production/MODE_SPEC/extract/products.xyz"
    CUTOFF_SCALE = 1.3

    product_index = build_product_index_and_write_files(
        TRAJ_FILE, elements, print_interval=print_interval, C=CUTOFF_SCALE
    )

    search_results = search_products_file_across_trajectories(
        PRODUCTS_XYZ, product_index, elements, C=CUTOFF_SCALE
    )


