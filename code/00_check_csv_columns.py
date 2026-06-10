import pandas as pd

# Load your data
df = pd.read_csv(r"F:\delirium_model_v2.csv")

print("=" * 80)
print("CHECKING YOUR CSV COLUMNS")
print("=" * 80)

print(f"\nTotal columns: {len(df.columns)}")
print(f"Total rows: {len(df)}")

# Look for race-related columns
print("\n" + "=" * 80)
print("LOOKING FOR RACE/ETHNICITY COLUMNS:")
print("=" * 80)
race_cols = [col for col in df.columns if 'race' in col.lower() or 'ethnic' in col.lower()]
if race_cols:
    print(f"Found {len(race_cols)} race-related columns:")
    for col in race_cols:
        print(f"  - {col}")
        print(f"    Values: {df[col].value_counts().to_dict()}")
else:
    print("NO race/ethnicity columns found!")
    print("\nShowing all demographic-looking columns:")
    demo_cols = [col for col in df.columns if any(x in col.lower() for x in ['age', 'sex', 'gender', 'insurance', 'marital'])]
    for col in demo_cols:
        print(f"  - {col}")

# Check for dementia column
print("\n" + "=" * 80)
print("CHECKING DEMENTIA COLUMN:")
print("=" * 80)
if 'dementia_icd' in df.columns:
    print(f"✓ dementia_icd found")
    print(f"  Total: {df['dementia_icd'].sum()} ({100*df['dementia_icd'].mean():.1f}%)")
else:
    print("✗ dementia_icd NOT found")
    dementia_cols = [col for col in df.columns if 'dement' in col.lower()]
    if dementia_cols:
        print(f"  Found these dementia columns instead: {dementia_cols}")

# Check for outcome
print("\n" + "=" * 80)
print("CHECKING OUTCOME COLUMN:")
print("=" * 80)
if 'outcome_incident_cam_24h' in df.columns:
    print(f"✓ outcome_incident_cam_24h found")
    print(f"  Events: {df['outcome_incident_cam_24h'].sum()} ({100*df['outcome_incident_cam_24h'].mean():.1f}%)")
else:
    print("✗ outcome_incident_cam_24h NOT found")

print("\n" + "=" * 80)
print("FIRST 50 COLUMNS (for reference):")
print("=" * 80)
for i, col in enumerate(df.columns[:50], 1):
    print(f"{i:2d}. {col}")
