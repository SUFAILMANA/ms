import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# --- Constants ---
EXPECTED_BANDS = {
    "Control": 782,
    "SCA Wild": 570,
    "IVS Mutant": 485,
    "IVS Wild": 344,
    "SCA Mutant": 266,
}
MIN_BAND = 266
MAX_BAND = 782
INTENSITY_THRESHOLD = 100  # Adjust if needed

# --- Helper functions ---
def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return np.array(image)

def detect_lanes(image, num_lanes=8):
    height, width = image.shape[:2]
    lane_width = width // num_lanes
    lanes = [(i * lane_width, (i + 1) * lane_width) for i in range(num_lanes)]
    return lanes

def detect_bands(gray_img, x_start, x_end):
    roi = gray_img[:, x_start:x_end]
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bands = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 5 and w > 10:
            center_y = y + h // 2
            bands.append(center_y)
    return sorted(bands)

def estimate_band_sizes(band_positions, ladder_positions):
    if len(ladder_positions) < 2:
        return []

    band_sizes = []
    ladder_bp = [500, 400, 300, 200]  # use thick 500bp band as anchor
    ladder_px = sorted(ladder_positions)

    for band_y in band_positions:
        distances = [abs(band_y - lp) for lp in ladder_px]
        closest = np.argmin(distances)
        if closest < len(ladder_bp):
            band_sizes.append(ladder_bp[closest])
        else:
            band_sizes.append(None)
    return band_sizes

def classify_genotype(sizes):
    sca = "SCA "
    ivs = "IVS "

    # SCA Logic
    if 570 in sizes and 266 in sizes:
        sca += "Heterozygous"
    elif 570 in sizes:
        sca += "Wild"
    elif 266 in sizes:
        sca += "Homo Mutant"
    else:
        sca += "Undetermined"

    # IVS Logic
    if 344 in sizes and 485 in sizes:
        ivs += "Heterozygous"
    elif 344 in sizes:
        ivs += "Wild"
    elif 485 in sizes:
        ivs += "Homo Mutant"
    else:
        ivs += "Undetermined"

    return sca + " / " + ivs

# --- Streamlit UI ---
st.title("🧬 ARMS PCR Gel Genotyper")
st.write("Upload a gel image to automatically detect lanes, bands, and determine genotypes for SCA and IVS mutations.")

uploaded_file = st.file_uploader("📁 Upload Gel Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image_np = load_image(uploaded_file)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    lanes = detect_lanes(image_np, num_lanes=8)

    # Detect ladder from center lane (assumed to be lane 4)
    ladder_band_positions = detect_bands(gray, *lanes[4])
    ladder_band_positions = [y for y in ladder_band_positions if 200 <= y <= 800]

    results = []
    st.image(image_np, caption="Uploaded Gel Image", use_column_width=True)

    for idx, (x1, x2) in enumerate(lanes):
        band_positions = detect_bands(gray, x1, x2)
        band_sizes = estimate_band_sizes(band_positions, ladder_band_positions)
        band_sizes = [s for s in band_sizes if s and MIN_BAND <= s <= MAX_BAND]
        genotype = classify_genotype(band_sizes)
        results.append({
            "Sample": f"Sample {idx + 1}",
            "Detected Bands": band_sizes,
            "Genotype": genotype,
        })

    df = pd.DataFrame(results)
    st.dataframe(df)

    st.download_button("⬇️ Download Results as CSV", df.to_csv(index=False), "gel_genotypes.csv", "text/csv")
