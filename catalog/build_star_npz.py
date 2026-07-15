import numpy as np
import pandas as pd
from pathlib import Path


# --------------------------------------------------------
# Convert RA and DEC (degrees) to unit vectors
# --------------------------------------------------------
def radec_to_unit_vector(ra_deg, dec_deg):

    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)

    return np.column_stack((x, y, z))


# --------------------------------------------------------
# Folder containing this script
# --------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

csv_path = BASE_DIR / "stars_thin.csv"
output_path = BASE_DIR / "stars.npz"


# --------------------------------------------------------
# Load catalog
# --------------------------------------------------------


df = pd.read_csv(csv_path)



# --------------------------------------------------------
# Extract required fields
# --------------------------------------------------------
designation = df["designation"].values

ra = df["ra"].values.astype(np.float64)

dec = df["dec"].values.astype(np.float64)

j_mag = df["j_m"].values.astype(np.float32)


# --------------------------------------------------------
# Compute unit vectors
# --------------------------------------------------------
print("Computing unit vectors")

unit_vectors = radec_to_unit_vector(ra, dec)


# --------------------------------------------------------
# Save database
# --------------------------------------------------------


np.savez_compressed(
    output_path,
    designation=designation,
    ra=ra,
    dec=dec,
    j_mag=j_mag,
    unit_vectors=unit_vectors
)


print("Total stars :", len(ra))
print("Saved to :", output_path)