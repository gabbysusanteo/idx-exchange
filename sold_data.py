
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Week 1 — Monthly Dataset Aggregation

# ----------------
# SOLD: concatenate
# ----------------
# Find all monthly sold files (CRMLSSoldYYYYMM.csv)
sold_files = sorted(glob.glob("CRMLSSold*.csv"))

# Load each file and store row counts before concatenation
sold_frames = []
print("Sold files and row counts:")
for f in sold_files:
    df = pd.read_csv(f, low_memory=False)
    print(f"  {f}: {len(df):,} rows")                 # row count before concat (per file)
    sold_frames.append(df)

# Concatenate into one dataframe
if sold_frames:
    sold = pd.concat(sold_frames, ignore_index=True)
else:
    sold = pd.DataFrame()

print(f"\nSold rows AFTER concatenation: {len(sold):,}")   # row count after concat

# PropertyType breakdown BEFORE Residential filter (if present)
if "PropertyType" in sold.columns:
    print("\nSold: PropertyType value counts BEFORE filter:")
    print(sold["PropertyType"].value_counts(dropna=False).to_string())

# Filter to Residential only
if "PropertyType" in sold.columns:
    sold_residential = sold[sold["PropertyType"] == "Residential"].copy()
else:
    sold_residential = sold.copy()

print(f"\nSold rows AFTER Residential filter: {len(sold_residential):,}")  # row count after filter

# Save sold outputs
sold.to_csv("CRMLSSold_combined.csv", index=False)
sold_residential.to_csv("CRMLSSold_residential.csv", index=False)


# -------------------
# LISTINGS: concatenate
# -------------------
# Find all monthly listing files (CRMLSListingYYYYMM.csv)
listing_files = sorted(glob.glob("CRMLSListing*.csv"))

# Load each file and store row counts before concatenation
listing_frames = []
print("\nListing files and row counts:")
for f in listing_files:
    df = pd.read_csv(f, low_memory=False)
    print(f"  {f}: {len(df):,} rows")                 # row count before concat (per file)
    listing_frames.append(df)

# Concatenate into one dataframe
if listing_frames:
    listings = pd.concat(listing_frames, ignore_index=True)
else:
    listings = pd.DataFrame()

print(f"\nListing rows AFTER concatenation: {len(listings):,}")   # row count after concat

# PropertyType breakdown BEFORE Residential filter (if present)
if "PropertyType" in listings.columns:
    print("\nListings: PropertyType value counts BEFORE filter:")
    print(listings["PropertyType"].value_counts(dropna=False).to_string())

# Filter to Residential only
if "PropertyType" in listings.columns:
    listings_residential = listings[listings["PropertyType"] == "Residential"].copy()
else:
    listings_residential = listings.copy()

print(f"\nListing rows AFTER Residential filter: {len(listings_residential):,}")  # row count after filter

# Save listing outputs
listings.to_csv("CRMLSListing_combined.csv", index=False)
listings_residential.to_csv("CRMLSListing_residential.csv", index=False)

# Weeks 2–3 — Dataset Structuring, Validation, EDA, and FRED mortgage enrichment
# --------------------
# 1) Load combined datasets (from Week 1)
# --------------------
print("Loading combined sold/listing CSVs...")
sold = pd.read_csv("CRMLSSold_combined.csv", low_memory=False)
listings = pd.read_csv("CRMLSListing_combined.csv", low_memory=False)
print(f"Sold rows loaded: {len(sold):,}")
print(f"Listings rows loaded: {len(listings):,}")
print()

# --------------------
# 2) Dataset understanding (rows, cols, dtypes)
# --------------------
print("SOLD: shape (rows,cols):", sold.shape)
print("LISTINGS: shape (rows,cols):", listings.shape)
print()

print("SOLD: first 10 columns:")
print(list(sold.columns[:10]))
print()

print("SOLD: dtypes (first 20):")
print(sold.dtypes.head(20).to_string())
print()

# --------------------
# 3) Unique property types & filter to Residential
# --------------------
print("Unique PropertyType values in SOLD (sample):")
if "PropertyType" in sold.columns:
    print(sold["PropertyType"].unique())
else:
    print("PropertyType not found in sold dataset.")

print("\nUnique PropertyType values in LISTINGS (sample):")
if "PropertyType" in listings.columns:
    print(listings["PropertyType"].unique())
else:
    print("PropertyType not found in listings dataset.")

# Apply Residential filter (handbook requirement)
if "PropertyType" in sold.columns:
    sold_res = sold[sold["PropertyType"] == "Residential"].copy()
else:
    sold_res = sold.copy()

if "PropertyType" in listings.columns:
    listings_res = listings[listings["PropertyType"] == "Residential"].copy()
else:
    listings_res = listings.copy()

print(f"\nSold rows BEFORE filter: {len(sold):,}  AFTER Residential filter: {len(sold_res):,}")
print(f"Listings rows BEFORE filter: {len(listings):,}  AFTER Residential filter: {len(listings_res):,}")
print()

# -------------------
# 4) Missing value analysis & flag >90% missing
# -------------------


def missing_summary(df, name):
    print(f"--- Missing value summary: {name} ---")
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    mv = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct}).sort_values("missing_pct", ascending=False)
    print(mv.head(40).to_string())
    high_missing = mv[mv["missing_pct"] > 90]
    print(f"\nColumns with >90% missing in {name}: {len(high_missing)}")
    if len(high_missing):
        print(high_missing.to_string())
    print()

missing_summary(sold_res, "SOLD (residential)")
missing_summary(listings_res, "LISTINGS (residential)")

# --------------------
# 5) Numeric distribution summary (ClosePrice, LivingArea, DaysOnMarket)
# --------------------
print("--- Numeric distribution summary (SOLD) ---")
cols = []
for c in ["ClosePrice","LivingArea","DaysOnMarket"]:
    if c in sold_res.columns:
        cols.append(c)

if cols:
    print(sold_res[cols].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]).round(2).to_string())
else:
    print("One or more of ClosePrice, LivingArea, DaysOnMarket not present in sold dataset.")
print()

# --------------------
# 6) Quick EDA answers required
# --------------------
print("--- Quick EDA answers (SOLD) ---")
# Residential vs other share (if PropertyType present in original sold)
if "PropertyType" in sold.columns:
    prop_share = sold["PropertyType"].value_counts(normalize=True).mul(100).round(2)
    print("PropertyType share (%):")
    print(prop_share.to_string())
else:
    print("PropertyType not available to compute share.")

# Median & average close price
if "ClosePrice" in sold_res.columns:
    print(f"Median ClosePrice: ${sold_res['ClosePrice'].median():,.0f}")
    print(f"Average ClosePrice: ${sold_res['ClosePrice'].mean():,.0f}")
else:
    print("ClosePrice not found in sold_res.")

# Days on market distribution
if "DaysOnMarket" in sold_res.columns:
    dom = sold_res["DaysOnMarket"].dropna()
    if len(dom):
        print(f"DaysOnMarket — median: {dom.median():.0f}, mean: {dom.mean():.1f}, max: {dom.max():.0f}")
    else:
        print("No DaysOnMarket data available.")
else:
    print("DaysOnMarket not found in sold_res.")

# Percent above vs below list price
if set(["ClosePrice","ListPrice"]).issubset(sold_res.columns):
    tmp = sold_res[["ClosePrice","ListPrice"]].dropna()
    if len(tmp):
        above = (tmp["ClosePrice"] >= tmp["ListPrice"]).sum()
        total = len(tmp)
        print(f"Sold at/above list price: {above:,} / {total:,} ({above/total*100:.1f}%)")
    else:
        print("ClosePrice/ListPrice pairs missing.")
else:
    print("ClosePrice or ListPrice not available to compute sold-above-list metric.")

# Date consistency example: CloseDate before ListingContractDate
if set(["CloseDate","ListingContractDate"]).issubset(sold_res.columns):
    cd = pd.to_datetime(sold_res["CloseDate"], errors="coerce")
    ld = pd.to_datetime(sold_res["ListingContractDate"], errors="coerce")
    bad = (cd < ld).sum()
    print(f"Rows with CloseDate before ListingContractDate: {bad:,}")
else:
    print("CloseDate or ListingContractDate not present for date consistency check.")

# Top counties by median price
if set(["CountyOrParish","ClosePrice"]).issubset(sold_res.columns):
    top_counties = sold_res.groupby("CountyOrParish")["ClosePrice"].median().sort_values(ascending=False).head(10)
    print("\nTop 10 counties by median ClosePrice:")
    print(top_counties.apply(lambda x: f"${x:,.0f}").to_string())
else:
    print("CountyOrParish or ClosePrice missing; cannot compute county medians.")
print()

# --------------------
# 7) Save filtered datasets (residential)
# --------------------
sold_res.to_csv("CRMLSSold_residential_filtered.csv", index=False)
listings_res.to_csv("CRMLSListing_residential_filtered.csv", index=False)
print("Saved: CRMLSSold_residential_filtered.csv")
print("Saved: CRMLSListing_residential_filtered.csv")
print()

# --------------------
# 8) Mortgage rate enrichment (FRED) — resample weekly -> monthly and merge
# --------------------
print("Fetching FRED MORTGAGE30US series and resampling to monthly averages...")
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=["observation_date"])
mortgage.columns = ["date", "rate_30yr_fixed"]
mortgage["year_month"] = mortgage["date"].dt.to_period("M")
mortgage_monthly = mortgage.groupby("year_month", as_index=False)["rate_30yr_fixed"].mean()
print(f"Mortgage monthly rows: {len(mortgage_monthly):,}")
print()

# Create year_month keys and merge for sold (key off CloseDate)
if "CloseDate" in sold_res.columns:
    sold_res["year_month"] = pd.to_datetime(sold_res["CloseDate"], errors="coerce").dt.to_period("M")
    sold_with_rates = sold_res.merge(mortgage_monthly, on="year_month", how="left")
    null_rates_sold = sold_with_rates["rate_30yr_fixed"].isnull().sum()
    print(f"Sold rows missing mortgage rate after merge: {null_rates_sold:,}")
    sold_with_rates.to_csv("CRMLSSold_with_rates.csv", index=False)
    print("Saved: CRMLSSold_with_rates.csv")
else:
    print("CloseDate not in sold_res: skipping mortgage merge for sold.")
    sold_with_rates = sold_res.copy()

# Create year_month keys and merge for listings (key off ListingContractDate)
if "ListingContractDate" in listings_res.columns:
    listings_res["year_month"] = pd.to_datetime(listings_res["ListingContractDate"], errors="coerce").dt.to_period("M")
    listings_with_rates = listings_res.merge(mortgage_monthly, on="year_month", how="left")
    null_rates_listings = listings_with_rates["rate_30yr_fixed"].isnull().sum()
    print(f"Listings rows missing mortgage rate after merge: {null_rates_listings:,}")
    listings_with_rates.to_csv("CRMLSListing_with_rates.csv", index=False)
    print("Saved: CRMLSListing_with_rates.csv")
else:
    print("ListingContractDate not in listings_res: skipping mortgage merge for listings.")
    listings_with_rates = listings_res.copy()

print("\nWeeks 2–3 script complete.")

# Weeks 4–5 — Data Cleaning & Preparation
INPUT = "CRMLSSold_residential_filtered.csv"   # from Week 2–3
OUTPUT = "CRMLSSold_cleaned_week5.csv"

print("Loading:", INPUT)
sold = pd.read_csv(INPUT, low_memory=False)
rows_before = len(sold)
cols_before = len(sold.columns)
print(f"Rows before cleaning: {rows_before:,} | Columns: {cols_before}")
print()

# -------------------------
# 1) Convert date fields to datetime
# -------------------------
date_cols = ["CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate"]
print("Converting date fields to datetime (coerce invalid -> NaT):")
for col in date_cols:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors="coerce")
        print(f"  Converted {col}")
    else:
        print(f"  {col} not present")
print()

# -------------------------
# 2) Drop unnecessary / redundant columns (only if present)
# (example list; adjust if needed)
# -------------------------
drop_cols = [
    "ListingKey", "ListingKeyNumeric", "ListingId",
    "OriginatingSystemName", "OriginatingSystemSubName",
    "ListAgentEmail", "BuyerAgentMlsId", "BuyerOfficeAOR"
]
to_drop = [c for c in drop_cols if c in sold.columns]
if to_drop:
    sold.drop(columns=to_drop, inplace=True)
    print(f"Dropped {len(to_drop)} columns: {to_drop}")
else:
    print("No listed redundant/PII columns found to drop.")
print()

# -------------------------
# 3) Ensure numeric fields are numeric
# -------------------------
numeric_cols = ["ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea",
                "LotSizeAcres", "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt"]
print("Coercing numeric fields to numeric types (invalid -> NaN):")
for col in numeric_cols:
    if col in sold.columns:
        sold[col] = pd.to_numeric(sold[col], errors="coerce")
        print(f"  {col}: converted")
    else:
        print(f"  {col}: not present")
print()

# -------------------------
# 4) Flag invalid numeric values (do not delete)
# - ClosePrice <= 0
# - LivingArea <= 0
# - DaysOnMarket < 0
# - BedroomsTotal < 0
# - BathroomsTotalInteger < 0
# -------------------------
print("Adding numeric validation flags:")
sold["flag_invalid_price"] = sold.get("ClosePrice", np.nan) <= 0
sold["flag_invalid_area"] = sold.get("LivingArea", np.nan) <= 0
sold["flag_invalid_dom"] = sold.get("DaysOnMarket", np.nan) < 0
sold["flag_invalid_beds"] = sold.get("BedroomsTotal", np.nan) < 0
sold["flag_invalid_baths"] = sold.get("BathroomsTotalInteger", np.nan) < 0

for f in ["flag_invalid_price","flag_invalid_area","flag_invalid_dom","flag_invalid_beds","flag_invalid_baths"]:
    print(f"  {f}: {sold[f].sum():,}")
print()

# -------------------------
# 5) Date consistency checks
# - listing_after_close_flag: ListingContractDate > CloseDate
# - purchase_after_close_flag: PurchaseContractDate > CloseDate
# - negative_timeline_flag: PurchaseContractDate < ListingContractDate
# -------------------------
print("Date consistency flags (Listing -> Purchase -> Close):")
# ensure date columns exist
for c in ["ListingContractDate","PurchaseContractDate","CloseDate"]:
    if c not in sold.columns:
        sold[c] = pd.NaT

sold["listing_after_close_flag"] = sold["ListingContractDate"] > sold["CloseDate"]
sold["purchase_after_close_flag"] = sold["PurchaseContractDate"] > sold["CloseDate"]
sold["negative_timeline_flag"] = sold["PurchaseContractDate"] < sold["ListingContractDate"]

print(f"  listing_after_close_flag: {sold['listing_after_close_flag'].sum():,}")
print(f"  purchase_after_close_flag: {sold['purchase_after_close_flag'].sum():,}")
print(f"  negative_timeline_flag: {sold['negative_timeline_flag'].sum():,}")
print()

# -------------------------
# 6) Geographic data checks / flags
# - flag_missing_coords: Latitude or Longitude is null
# - flag_zero_coords: Latitude == 0 or Longitude == 0 (sentinel)
# - flag_positive_longitude: Longitude > 0 (California should be negative)
# - flag_coords_out_of_range: lat/lon outside valid ranges
# -------------------------
print("Geographic data quality flags:")
# create columns if they don't exist to avoid KeyError
if "Latitude" not in sold.columns:
    sold["Latitude"] = np.nan
if "Longitude" not in sold.columns:
    sold["Longitude"] = np.nan

sold["flag_missing_coords"] = sold["Latitude"].isnull() | sold["Longitude"].isnull()
sold["flag_zero_coords"] = (sold["Latitude"] == 0) | (sold["Longitude"] == 0)
sold["flag_positive_longitude"] = sold["Longitude"] > 0
sold["flag_coords_out_of_range"] = (
    (sold["Latitude"] > 90) | (sold["Latitude"] < -90) |
    (sold["Longitude"] > 180) | (sold["Longitude"] < -180)
)

print(f"  flag_missing_coords: {sold['flag_missing_coords'].sum():,}")
print(f"  flag_zero_coords: {sold['flag_zero_coords'].sum():,}")
print(f"  flag_positive_longitude: {sold['flag_positive_longitude'].sum():,}")
print(f"  flag_coords_out_of_range: {sold['flag_coords_out_of_range'].sum():,}")
print()

# -------------------------
# 7) Final summary & save
# -------------------------
rows_after = len(sold)
cols_after = len(sold.columns)
print("Cleaning summary:")
print(f"  Rows before: {rows_before:,}")
print(f"  Rows after : {rows_after:,}  (no rows deleted; flags only)")
print(f"  Cols before: {cols_before}")
print(f"  Cols after : {cols_after}")
print()

print("Data types after cleaning (first 20):")
print(sold.dtypes.head(20).to_string())
print()

# Save cleaned dataset
sold.to_csv(OUTPUT, index=False)
print("Saved cleaned dataset:", OUTPUT)

# Week 6 — Feature Engineering & Segment Summaries

print("\n=== WEEK 6: FEATURE ENGINEERING & SEGMENT SUMMARIES ===")

# Re-parse date columns safely (they may already be datetime)
for c in ["CloseDate", "PurchaseContractDate", "ListingContractDate"]:
    if c in sold.columns:
        sold[c] = pd.to_datetime(sold[c], errors="coerce")

# Engineered metrics
# Price ratio: ClosePrice / ListPrice
if set(["ClosePrice", "ListPrice"]).issubset(sold.columns):
    sold["price_ratio_list"] = sold["ClosePrice"] / sold["ListPrice"]
else:
    sold["price_ratio_list"] = np.nan

# Close to original list ratio: ClosePrice / OriginalListPrice
if set(["ClosePrice", "OriginalListPrice"]).issubset(sold.columns):
    sold["price_ratio_original_list"] = sold["ClosePrice"] / sold["OriginalListPrice"]
else:
    sold["price_ratio_original_list"] = np.nan

# Price per square foot
if set(["ClosePrice", "LivingArea"]).issubset(sold.columns):
    sold["price_per_sqft"] = sold["ClosePrice"] / sold["LivingArea"]
else:
    sold["price_per_sqft"] = np.nan

# Sold at or above list flag
sold["sold_above_list"] = sold.get("price_ratio_list") >= 1.0

# Time-series keys
if "CloseDate" in sold.columns:
    sold["close_year"] = sold["CloseDate"].dt.year
    sold["close_month"] = sold["CloseDate"].dt.month
    sold["close_yr_mo"] = sold["CloseDate"].dt.to_period("M").astype(str)
    sold["close_yr_mo_int"] = (sold["close_year"].fillna(0).astype(int) * 100 + sold["close_month"].fillna(0).astype(int)).astype("Int64")
else:
    sold["close_year"] = pd.NA
    sold["close_month"] = pd.NA
    sold["close_yr_mo"] = pd.NA
    sold["close_yr_mo_int"] = pd.NA

# Timeline / velocity metrics (days)
if set(["PurchaseContractDate","ListingContractDate"]).issubset(sold.columns):
    sold["days_listing_to_contract"] = (sold["PurchaseContractDate"] - sold["ListingContractDate"]).dt.days
else:
    sold["days_listing_to_contract"] = np.nan

if set(["CloseDate","PurchaseContractDate"]).issubset(sold.columns):
    sold["days_contract_to_close"] = (sold["CloseDate"] - sold["PurchaseContractDate"]).dt.days
else:
    sold["days_contract_to_close"] = np.nan

# Sample output showing new columns
sample_cols = [
    "CloseDate", "ClosePrice", "ListPrice", "OriginalListPrice", "price_ratio_list",
    "price_ratio_original_list", "LivingArea", "price_per_sqft", "sold_above_list",
    "days_listing_to_contract", "days_contract_to_close", "close_yr_mo", "close_yr_mo_int"
]
print("\nSample engineered columns (first 10 rows):")
existing_sample_cols = [c for c in sample_cols if c in sold.columns]
print(sold[existing_sample_cols].head(10).to_string(index=False))

# Save engineered dataset
sold.to_csv("engineered_sold_week6.csv", index=False)
print("\nSaved: engineered_sold_week6.csv")

# Segment summaries: by PropertyType and by CountyOrParish
print("\n--- Segmented summary: by PropertyType ---")
if "PropertyType" in sold.columns and "ClosePrice" in sold.columns:
    seg_pt = sold.groupby("PropertyType").agg(
        transactions = ("ClosePrice","count"),
        median_price = ("ClosePrice","median"),
        avg_price    = ("ClosePrice","mean"),
        median_ppsf  = ("price_per_sqft","median"),
        median_dom   = ("DaysOnMarket","median")
    ).sort_values("transactions", ascending=False)
    # print readable
    print(seg_pt.round(2).to_string())
else:
    print("PropertyType or ClosePrice missing — cannot produce PropertyType segmentation.")

print("\n--- Segmented summary: Top 15 CountyOrParish by transactions ---")
if "CountyOrParish" in sold.columns and "ClosePrice" in sold.columns:
    seg_county = sold.groupby("CountyOrParish").agg(
        transactions = ("ClosePrice","count"),
        median_price = ("ClosePrice","median"),
        median_ppsf  = ("price_per_sqft","median"),
        median_dom   = ("DaysOnMarket","median")
    ).sort_values("transactions", ascending=False).head(15)
    print(seg_county.round(2).to_string())
else:
    print("CountyOrParish or ClosePrice missing — cannot produce county segmentation.")

# Week 7 — Outlier Detection (IQR) & Data Quality

print("\n=== WEEK 7: IQR OUTLIER DETECTION & DATA QUALITY ===")

# Which columns to check
iqr_cols = [c for c in ["ClosePrice", "LivingArea", "DaysOnMarket"] if c in sold.columns]
print("IQR columns:", iqr_cols)

# Store before counts and medians
before_count = len(sold)
print(f"Rows before filtering: {before_count:,}")

before_medians = sold[iqr_cols].median(numeric_only=True) if iqr_cols else None
if before_medians is not None:
    print("\nMedian values BEFORE filtering:")
    print(before_medians.to_string())
else:
    print("No numeric columns available for median comparison.")

# Percentile check (useful guidance)
print("\nPercentiles (1%,5%,25%,50%,75%,95%,99%) for IQR columns:")
for col in iqr_cols:
    print(f"\n{col} percentiles:")
    print(sold[col].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]).round(2).to_string())

# Create IQR flags per column (flag True = outlier)
flag_cols = []
for col in iqr_cols:
    q1 = sold[col].quantile(0.25)
    q3 = sold[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    flag_name = f"flag_iqr_{col}"
    sold[flag_name] = ~sold[col].between(lower, upper, inclusive="both")
    flag_cols.append(flag_name)
    print(f"\n{col}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, lower={lower:.2f}, upper={upper:.2f}")
    print(f"  Outliers flagged in {flag_name}: {sold[flag_name].sum():,}")

# Combined any-iqr-outlier flag
if flag_cols:
    sold["flag_any_iqr_outlier"] = sold[flag_cols].any(axis=1)
    print(f"\nTotal rows flagged by any IQR outlier: {sold['flag_any_iqr_outlier'].sum():,}")
else:
    sold["flag_any_iqr_outlier"] = False
    print("No IQR flag columns created (missing numeric fields).")

# Always-invalid numeric rules (business rules) — flag (do not delete)
# ClosePrice <= 0, LivingArea <= 0, DaysOnMarket < 0
sold["flag_invalid_price"] = sold.get("ClosePrice", np.nan) <= 0
sold["flag_invalid_area"] = sold.get("LivingArea", np.nan) <= 0
sold["flag_invalid_dom"] = sold.get("DaysOnMarket", np.nan) < 0

print("\nAlways-invalid numeric flag totals:")
print(f"  flag_invalid_price: {sold['flag_invalid_price'].sum():,}")
print(f"  flag_invalid_area : {sold['flag_invalid_area'].sum():,}")
print(f"  flag_invalid_dom  : {sold['flag_invalid_dom'].sum():,}")

# Build filtered dataset: exclude always-invalid numeric rows and IQR outliers
base_invalid = sold["flag_invalid_price"] | sold["flag_invalid_area"] | sold["flag_invalid_dom"]
filtered = sold[~base_invalid & ~sold["flag_any_iqr_outlier"]].copy()

after_count = len(filtered)
print(f"\nRows after filtering: {after_count:,}  (removed {before_count - after_count:,} rows)")

# Medians after filtering
if iqr_cols:
    after_medians = filtered[iqr_cols].median(numeric_only=True)
    print("\nMedian values AFTER filtering:")
    print(after_medians.to_string())
else:
    print("No numeric columns to show medians after filtering.")

# Save outputs
sold.to_csv("sold_with_outlier_flags_week7.csv", index=False)
filtered.to_csv("sold_filtered_week7.csv", index=False)
print("\nSaved: sold_with_outlier_flags_week7.csv (full dataset with flags)")
print("Saved: sold_filtered_week7.csv (filtered analysis-ready dataset)")

# Short written comparison (printed)
print("\n--- Written comparison summary ---")
print(f"Rows before filtering : {before_count:,}")
print(f"Rows after filtering  : {after_count:,}")
if before_medians is not None and after_medians is not None:
    print("\nMedian comparison (before -> after):")
    for col in iqr_cols:
        b = before_medians.get(col, float("nan"))
        a = after_medians.get(col, float("nan"))
        print(f"  {col}: {b:.2f} -> {a:.2f}  (change: {a - b:.2f})")
else:
    print("Median comparison not available (missing numeric columns).")

print("\nWeek 7 complete.")
