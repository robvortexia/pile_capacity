#!/usr/bin/env python3

from app.interpolation import process_uploaded_cpt_data

# Read the AH Short.txt file
with open('AH Short.txt', 'r') as f:
    file_content = f.read()

print("Testing interpolation with AH Short.txt...")
print("=" * 50)

# Process the data
result, warning = process_uploaded_cpt_data(file_content)

print(f"Original data: 5 points (1m, 2m, 3m, 4m, 5m)")
print(f"Processed data: {len(result)} points")
print(f"Warning: {warning}")
print()

print("Original data points:")
print("Depth(m) | fs(kPa) | qc(MPa) | Ic")
print("-" * 35)
original_depths = [1, 2, 3, 4, 5]
for depth in original_depths:
    for row in result:
        if abs(row[0] - depth) < 0.001:  # Find exact matches
            print(f"{row[0]:7.1f} | {row[1]:7.3f} | {row[2]:7.3f} | {row[3]:2.1f}")
            break

print()
print("Sample interpolated points (every 0.5m):")
print("Depth(m) | fs(kPa) | qc(MPa) | Ic")
print("-" * 35)
for row in result:
    if row[0] % 0.5 == 0:  # Show every 0.5m
        print(f"{row[0]:7.1f} | {row[1]:7.3f} | {row[2]:7.3f} | {row[3]:2.1f}")

print()
print(f"Total interpolated points: {len(result)}")
print("✅ AH Short.txt interpolation test completed!")
