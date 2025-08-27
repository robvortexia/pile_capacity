#!/usr/bin/env python3

from app.interpolation import process_uploaded_cpt_data, format_cpt_data_for_download

# Read the AH Short.txt file
with open('AH Short.txt', 'r') as f:
    file_content = f.read()

print("AH Short.txt - Complete Interpolated Result")
print("=" * 60)

# Process the data
result, warning = process_uploaded_cpt_data(file_content)

print(f"Original: 5 points (1m, 2m, 3m, 4m, 5m)")
print(f"Interpolated: {len(result)} points at 0.1m intervals")
print(f"Warning: {warning}")
print()

# Show the complete interpolated data
print("Complete Interpolated Data:")
print("Depth(m) | fs(kPa) | qc(MPa) | Ic")
print("-" * 45)

for row in result:
    print(f"{row[0]:7.1f} | {row[1]:7.3f} | {row[2]:7.3f} | {row[3]:2.1f}")

print()
print("=" * 60)
print("✅ Complete interpolated result shown above!")
print()
print("Note: Original data points are preserved exactly,")
print("      intermediate points are linearly interpolated.")
