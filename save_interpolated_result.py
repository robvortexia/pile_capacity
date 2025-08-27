#!/usr/bin/env python3

from app.interpolation import process_uploaded_cpt_data, format_cpt_data_for_download

# Read the AH Short.txt file
with open('AH Short.txt', 'r') as f:
    file_content = f.read()

print("Processing AH Short.txt and saving interpolated result...")

# Process the data
result, warning = process_uploaded_cpt_data(file_content)

# Format the data for saving
formatted_data = format_cpt_data_for_download(result)

# Save to file
output_filename = "AH_Short_Interpolated.txt"
with open(output_filename, 'w') as f:
    f.write(formatted_data)

print(f"✅ Interpolated data saved to: {output_filename}")
print(f"Original: 5 points → Interpolated: {len(result)} points")
print(f"Warning: {warning}")
print()
print("File contains tab-separated values with header:")
print("Depth (m)\tfs (kPa)\tqc (MPa)\tIc")
