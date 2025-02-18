import os
import pandas as pd
import random
import tqdm
import sys

from table_utils import parse_table_to_text  

def process_mimic_data(input_dir, output_dir, train_ratio=0.7, valid_ratio=0.2):
    """Processes MIMIC-III data, splits into train/valid/test, and generates .data and .text files."""

    # Load CSV files and convert column names to lowercase
    admissions = pd.read_csv(os.path.join(input_dir, "mimic-iii-clinical-database-demo-1.4/ADMISSIONS.csv"))
    diagnoses = pd.read_csv(os.path.join(input_dir, "mimic-iii-clinical-database-demo-1.4/DIAGNOSES_ICD.csv"))
    icustays = pd.read_csv(os.path.join(input_dir, "mimic-iii-clinical-database-demo-1.4/ICUSTAYS.csv"))

    # Convert all column names to lowercase to avoid KeyError issues
    admissions.columns = admissions.columns.str.lower()
    diagnoses.columns = diagnoses.columns.str.lower()
    icustays.columns = icustays.columns.str.lower()

    # Merge data by patient admission ID
    merged = (
        admissions
        .merge(diagnoses, on="hadm_id", how="left")
        .merge(icustays[["hadm_id", "los"]], on="hadm_id", how="left")  # Only keep LOS from icustays
    )

    all_records = []

    # Generate data samples
    for _, row in tqdm.tqdm(merged.iterrows(), total=len(merged), desc="Processing MIMIC-III data"):
        text = generate_text(row)
        table = generate_table(row)
        all_records.append((text, table))

    # Shuffle before splitting
    random.shuffle(all_records)

    # Split into train, valid, test
    total_records = len(all_records)
    train_size = int(total_records * train_ratio)
    valid_size = int(total_records * valid_ratio)
    test_size = total_records - train_size - valid_size

    datasets = {
        "train": all_records[:train_size],
        "valid": all_records[train_size:train_size + valid_size],
        "test": all_records[train_size + valid_size:]
    }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Total records collected: {len(all_records)}")

    # Save to .data and .text files
    for split, records in datasets.items():
        print(output_dir)
        with open(os.path.join(output_dir, f"{split}.data"), "w") as data_f, \
             open(os.path.join(output_dir, f"{split}.text"), "w") as text_f:

            for text, table in records:
                print(data_f)
                data_f.write(table.replace("\n", " <NEWLINE> ").strip() + "\n")
                text_f.write(text + "\n")

    print(f"Data saved in {output_dir} (train: {train_size}, valid: {valid_size}, test: {test_size})")


def generate_text(row):
    """Generates a textual medical summary for a patient."""
    age = random.randint(30, 80)  # MIMIC-III does not contain exact age
    gender = random.choice(["Male", "Female"])
    diagnosis = row["icd9_code"] if pd.notna(row["icd9_code"]) else "Unknown Diagnosis"

    text = f"A {age}-year-old {gender} patient was admitted with a diagnosis of {diagnosis}. "

    if "los" in row and pd.notna(row["los"]):  # Check if LOS exists
        text += f"The patient stayed in the ICU for {int(row['los'])} days. "

    return text


def generate_table(row):
    """Generates structured tabular data for a patient."""
    table = {
        "admission id": row["hadm_id"],
        "age": random.randint(30, 80),
        "gender": random.choice(["male", "female"]),
        "diagnosis": row["icd9_code"] if pd.notna(row["icd9_code"]) else "unknown diagnosis",
        "icu stay": "yes" if "los" in row and pd.notna(row["los"]) else "no"
    }

    return parse_table_to_text([list(table.keys()), list(table.values())])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_mimic_data.py <input_dir> <output_dir>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    process_mimic_data(input_directory, output_directory)