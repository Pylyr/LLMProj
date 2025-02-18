import os
import json
import random
import tqdm
import sys

from table_utils import parse_table_to_text  


FINANCIAL_METRICS_KEYS = {
    "revenue": "Revenue",
    "netIncome": "Net Income",
    "operatingExpenses": "Operating Expenses",
    "cashFlow": "Cash Flow"
}

def extract_quarter_info(folder_name):
    """Extracts the year and quarter from the folder name (e.g., 2010.QTR1 -> 2010, 1)"""
    try:
        year, quarter = folder_name.split(".QTR")
        return int(year), int(quarter)
    except ValueError:
        return None, None  # If the folder structure is unexpected

def process_financial_reports(input_dir, output_dir, train_ratio=0.7, valid_ratio=0.2):
    """Processes all JSON files, splits them into train/valid/test, and generates required files"""
    
    all_records = []

    # Scan all folders and files
    for folder in tqdm.tqdm(sorted(os.listdir(input_dir)), desc="Processing Folders"):
        folder_path = os.path.join(input_dir, folder)

        if not os.path.isdir(folder_path):  # Skip files, process only directories
            continue

        year, quarter = extract_quarter_info(folder)
        if year is None:
            print(f"Skipping invalid folder: {folder}")
            continue

        for json_file in os.listdir(folder_path):
            if not json_file.endswith(".json"):
                continue

            json_path = os.path.join(folder_path, json_file)
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading {json_file}, skipping")
                    continue

            # Check if the JSON contains the required data
            if not isinstance(data, dict) or "symbol" not in data:
                continue

            company_name = data["symbol"]

            # Create the table
            table_data = [["Company", "Year", "Quarter"] + list(FINANCIAL_METRICS_KEYS.values())]
            row = [company_name, year, quarter] + [data.get(k, "N/A") for k in FINANCIAL_METRICS_KEYS.keys()]
            table_data.append(row)

            # Generate text format for the model
            text = "Company:\n{}\nMetrics:\n{}".format(
                parse_table_to_text(table_data[:1]),  # Headers
                parse_table_to_text(table_data[1:])   # Data
            )
            text = '\n'.join([line.strip() for line in text.splitlines() if len(line.strip()) > 0])

            summary = f"{company_name} reported revenue of {data.get('revenue', 'N/A')} in Q{quarter} {year}, " \
                      f"net income of {data.get('netIncome', 'N/A')}, and operating expenses of {data.get('operatingExpenses', 'N/A')}."

            all_records.append((text, summary))

    # Shuffle data before splitting
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

    # Save to .data and .text files
    for split, records in datasets.items():
        with open(os.path.join(output_dir, f"{split}.data"), "w") as data_f, \
             open(os.path.join(output_dir, f"{split}.text"), "w") as text_f:

            for text, summary in records:
                data_f.write(text.replace("\n", " <NEWLINE> ").strip() + "\n")
                text_f.write(summary + "\n")

    print(f"Data saved in {output_dir} (train: {train_size}, valid: {valid_size}, test: {test_size})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_financial_data.py <input_dir> <output_dir>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    process_financial_reports(input_directory, output_directory)


