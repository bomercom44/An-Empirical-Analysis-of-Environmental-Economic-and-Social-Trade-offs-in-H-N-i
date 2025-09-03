# An-Empirical-Analysis-of-Environmental-Economic-and-Social-Trade-offs-in-H-N-i
Reproducibility Pack — Hà Nội Urban Freight (E–Ec–S & DOBI)

This repository contains all code, configuration, and processed data needed to recreate and validate the dissertation results on sustainable last-mile logistics in Hà Nội. The workflow builds KPIs, fits the core models (H1–H4), runs robustness checks (service-time and low-speed thresholds), and reconstructs Pareto frontiers used in the paper.

No live Google APIs are called here; the project uses exported travel times and geocoding results.
All runs are deterministic given the inputs and config.

requirements:
pandas>=2.3
numpy>=2.3
PyYAML>=6.0
statsmodels>=0.14
scikit-learn>=1.7
matplotlib>=3.10
tqdm>=4.66


1) What you will reproduce

KPIs (driver–day): emissions/cost intensities, worked hours, DOBI and components.

Core regression models (H1–H4) at three congestion thresholds (12/15/18 km·h⁻¹) and multiple service-time assumptions (8/10/12/15 min).

Robustness analyses:

Service-time robustness (rank stability, mean shifts).

Threshold sensitivity for the low-speed share (only S_DOBI and congestion component change, as expected).

Empirical Pareto frontiers and scenario deltas.

Validation tables that the dissertation cites.

2) Repo layout
.
├─ 01_build_kpis.py
├─ 02_fit_models.py                 # baseline model runner
├─ 02_fit_models_safe_verbose.py    # safer/verbose variant (handles collinearity)
├─ 03_frontier_empirical.py
├─ 05_validation_service_time_robustness.py
├─ 06_threshold_sensitivity.py
├─ config.yml
├─ schedule_with_arrival_times.csv  # cleaned raw legs (this repo's copy)
├─ outputs/                         # all generated artifacts land here
│   ├─ kpis_driverday_12_drive.csv
│   ├─ kpis_driverday_15_drive.csv
│   ├─ kpis_driverday_18_drive.csv
│   ├─ legs_enriched_12_drive.csv
│   ├─ legs_enriched_15_drive.csv
│   ├─ legs_enriched_18_drive.csv
│   ├─ model_*                      # regression summaries & tidy tables
│   ├─ frontier_*                   # CSVs and PNGs
│   └─ validation_*                 # robustness outputs
└─ README.md

3) System requirements

Python 3.12

Packages: pandas 2.3+, numpy 2.3+, PyYAML 6+, statsmodels 0.14+, scikit-learn 1.7+, matplotlib 3.10+, tqdm

4) Quick start
Windows (PowerShell)
# 1) Go to project folder
cd "C:\Users\<YOU>\OneDrive\Desktop\Disseration\part4"

# 2) Create & activate a virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 3) Install dependencies
pip install pandas numpy pyyaml statsmodels scikit-learn matplotlib tqdm

macOS / Linux
cd /path/to/project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy pyyaml statsmodels scikit-learn matplotlib tqdm

5) Configuration (what the code assumes)

Open config.yml (already filled with final values). Key points:

Time & SOP (company policy)

depot_load_minutes: 30

service_minutes_list: [8, 10, 12, 15] (BAU = 10; robustness at 8/12/15)

paid_break_minutes: 60

late_cutoff_hhmm: "18:30"

DOBI

low_speed_thresholds: [12, 15, 18]

dobi_weights: [0.4, 0.3, 0.3] (congestion, stop density, shift variability)

Costing (monthly pools) for 4 vehicles

monthly_totals_vnd.fuel_total: 21000000

monthly_totals_vnd.maintenance_total: 3600000

monthly_costs (driver/assistant/other), allocated per working day

Fuel & EF

co2_per_litre.diesel: 2.68, petrol: 2.31

l_per_100km mapping by vehicle model

Geocoding confidence

geocode_quality_field: geometry_type (values: ROOFTOP, RANGE_INTERPOLATED, GEOMETRIC_CENTER, APPROXIMATE)

Dates

date_format: "%d/%m/%Y", dayfirst: true, timezone Asia/Ho_Chi_Minh

6) Step-by-step reproduction
A. Build KPIs & legs (all service-time scenarios in one go)
# Windows, in the project root
.\.venv\Scripts\Activate.ps1
python 01_build_kpis.py --config .\config.yml


Outputs (in outputs/):

kpis_driverday_12_drive.csv, kpis_driverday_15_drive.csv, kpis_driverday_18_drive.csv

legs_enriched_12_drive.csv, legs_enriched_15_drive.csv, legs_enriched_18_drive.csv

Console will print a reconciliation for BAU (service 10):

Fuel cost sum ≈ 21,000,000 VND

Maintenance cost sum ≈ 3,600,000 VND

Also prints per-day labour+other allocation and BAU driver-day count

Validation check 1: The two totals should match within rounding (a few VND okay).
Validation check 2: Driver-day count equals the number of observed driver–days in the month (e.g., 96).

B. Fit models (H1–H4) for each service-time file

Use the safer, verbose script (handles multicollinearity, centers/standardizes, writes tidy tables).

# From project root
python .\02_fit_models_safe_verbose.py `
  --kpis_glob ".\outputs\kpis_driverday_*_drive.csv" `
  --outdir ".\outputs" `
  --verbose


Outputs written into outputs/:

model_summaries_by_service.csv (one tidy table per hypothesis × service time)

model_quickview_by_service.csv (signs, CIs, adj.R², N)

Text summaries: model_*_summary.txt

Validation check 3 (signs):

H1 (Emissions–speed): coefficient on low_speed_share_time_t15 positive for CO₂_g/km (more low-speed → higher g/km).

H2 (Workload–stops): coefficient on stops positive for WorkedHours.

H4 (Congestion burden): coefficient on low_speed_share_time_t15 positive for WorkedHours.
Fixed effects (driver, weekday) included.

If you see “Condition number is large”: the script already:

centers/scales continuous covariates,

drops collinear dummies,

uses robust (HC1) standard errors,

logs a warning to outputs/model_warnings.log.

C. Service-time robustness (8/10/12/15 min)

This compares BAU (10) against other service-time assumptions.

python .\05_validation_service_time_robustness.py `
  --kpis ".\outputs\kpis_driverday_15_drive.csv" `
  --outdir ".\outputs"


Outputs:

service_time_summary.csv (means by service time: WorkedHours, cost_per_km, S_DOBI)

service_time_rank_corr.csv (Kendall’s τ rank stability for S_DOBI across service times)

service_time_rank_matrix.csv (pairwise τ matrix)

scenario_compare_service_minutes.csv (tidy deltas)

Validation check 4:

Signs and “what changes with service time” are intuitive.

Kendall’s τ for S_DOBI rankings ≥ 0.9 indicates robust ordering across service times.

D. Threshold sensitivity (12 vs 15 vs 18 km·h⁻¹)

Only S_DOBI (and its congestion component) should change when you vary the congestion speed threshold.

python .\06_threshold_sensitivity.py `
  --legs_12 ".\outputs\legs_enriched_12_drive.csv" `
  --legs_15 ".\outputs\legs_enriched_15_drive.csv" `
  --legs_18 ".\outputs\legs_enriched_18_drive.csv" `
  --outdir ".\outputs"


Outputs:

threshold_deltas_vs15.csv, threshold_deltas_vs15_long.csv

threshold_compare_summary.csv, threshold_compare_summary_tidy.csv

threshold_dobi_rank_correlation.csv (Kendall’s τ across thresholds)

Validation check 5: rank correlations are high; only the congestion component and S_DOBI move.

E. Frontiers (empirical)
python .\03_frontier_empirical.py `
  --kpis ".\outputs\kpis_driverday_15_drive.csv" `
  --outdir ".\outputs"


Outputs:

frontier_bau_points.csv, frontier_bau_only.csv

Plots: frontier_svc8.0_E_vs_C.png, frontier_svc8.0_E_vs_S.png, frontier_svc8.0_C_vs_S.png (and BAU variants)

Validation check 6:

Plots show a downward sloping trade-off in E vs C; E vs S and C vs S show plausible trade-offs.

The set of efficient points is non-dominated (the script asserts this).

7) Data & assumptions (for interpretation)

Scope: urban road last-mile in Hà Nội (Asia/Ho_Chi_Minh). Observed month; Sundays omitted.

Tour structure: Car-park → Depot (load) → Customers → Car-park.

No backhauls in this sample; no cancelled/failed stops present; non-delivery repositioning legs are included and flagged.

Drivers determine actual stop sequence on the day; sequence recorded by company tablet.

Service time = 10 min/stop is the SOP baseline; 8/12/15 min scenarios test realism/robustness.

Late cutoff = 18:30 by SOP/working hours (no early tolerance).

Geocoding confidence = geometry_type from Google Geocoding results (ROOFTOP > RANGE_INTERPOLATED > GEOMETRIC_CENTER > APPROXIMATE).

Clock drift: not testable here (no device/server raw timestamps). This is documented and mitigated by relying on leg-level distances and travel times (not wall-clock arrivals).

8) Common pitfalls & fixes

File not found: Make sure you run commands from the project root and paths in config.yml point to the local files.

PowerShell globbing: Always quote globs, e.g. ".\outputs\kpis_driverday_*_drive.csv".

Date parsing errors: The CSV is dd/mm/YYYY. Confirm date_format: "%d/%m/%Y" and dayfirst: true in config.yml.

Vehicle fuel map errors: If you see Vehicle '<name>' missing in config:l_per_100km, add the exact string to the l_per_100km map or use a canonicalization map inside the script (already included).

Collinearity warnings: Use 02_fit_models_safe_verbose.py (it centers/scales, prunes collinear dummies, uses HC1 SEs).

Zeros in low_speed_share: These indicate no legs below the threshold on that driver–day. This is acceptable and informative for sensitivity analyses.

9) Verification checklist (what I will grade)

Reconciliation (BAU):

Fuel ≈ 21,000,000 VND; Maintenance ≈ 3,600,000 VND across BAU driver–days.

Model signs and significance (H1–H4):

Low-speed share raises CO₂_g/km and WorkedHours; Stops raise WorkedHours; controls behave sensibly.

Fixed effects included; adj.R² and N consistent across thresholds/service times.

Robustness:

Service-time rank stability (Kendall’s τ ≥ 0.9).

Threshold sensitivity affects only S_DOBI & congestion component.

Frontiers:

Non-dominated sets recovered; plots saved; text summary lists 2–3 actionable deltas.

10) License, ethics, and ToS

Data are de-identified (anonymized driver_id; address geometry not exposed beyond district or as aggregated KPIs).

Travel time and geocoding outputs are exported artifacts; no API calls occur in this repo.

All analyses are for academic research only; please cite the dissertation when reusing the code.

11) How to re-run specific pieces

Only models (for one file):

python .\02_fit_models.py --kpis .\outputs\kpis_driverday_15_drive.csv --outdir .\outputs


Only service-time robustness:

python .\05_validation_service_time_robustness.py --kpis .\outputs\kpis_driverday_15_drive.csv --outdir .\outputs


Only threshold sensitivity:

python .\06_threshold_sensitivity.py `
  --legs_12 ".\outputs\legs_enriched_12_drive.csv" `
  --legs_15 ".\outputs\legs_enriched_15_drive.csv" `
  --legs_18 ".\outputs\legs_enriched_18_drive.csv" `
  --outdir ".\outputs"


Only frontiers (empirical):

python .\03_frontier_empirical.py --kpis .\outputs\kpis_driverday_15_drive.csv --outdir .\outputs

12) Contact

For any reproducibility issues, please open a GitHub issue with:

your OS and Python version,

the exact command you ran,

the console error,

the beginning of your config.yml.

Thanks for reviewing & validating!
