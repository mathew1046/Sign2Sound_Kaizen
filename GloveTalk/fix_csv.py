import csv

CSV_FILENAME = "sign_language_dataset.csv"
CLEAN_FILENAME = "sign_language_dataset_clean.csv"

expected_columns = 20
fixed_count = 0

print(f"🧹 Scanning {CSV_FILENAME} for corrupted rows...")

with open(CSV_FILENAME, 'r') as infile, open(CLEAN_FILENAME, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    previous_valid_row = None
    
    for line_num, row in enumerate(reader, 1):
        if len(row) == expected_columns:
            writer.writerow(row)
            previous_valid_row = row
        else:
            print(f"⚠️ Corrupted data on line {line_num} (found {len(row)} columns). Fixing...")
            if previous_valid_row:
                writer.writerow(previous_valid_row) # Duplicate previous frame
            else:
                writer.writerow([0] * expected_columns)
            fixed_count += 1

print(f"\n✅ Done! Repaired {fixed_count} corrupted lines.")
print(f"➡️ The fixed data is saved as '{CLEAN_FILENAME}'.")
print(f"Delete your old CSV and rename '{CLEAN_FILENAME}' to '{CSV_FILENAME}'.")