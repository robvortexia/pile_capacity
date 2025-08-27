#!/usr/bin/env python3

from app.interpolation import process_uploaded_cpt_data

# Test data with sparse depths
test_data = """1.0 0.151 4.6 19
2.0 0.765 7.5 19
3.0 0.114 1.2 19"""

print("Testing interpolation integration...")
print("=" * 50)

# Process the data
result, warning = process_uploaded_cpt_data(test_data)

print(f"Original data: 3 points")
print(f"Processed data: {len(result)} points")
print(f"Warning: {warning}")
print()

print("First 10 interpolated points:")
print("Depth(m) | fs(kPa) | qc(MPa) | Ic")
print("-" * 35)
for i, row in enumerate(result[:10]):
    print(f"{row[0]:7.1f} | {row[1]:7.3f} | {row[2]:7.3f} | {row[3]:2.1f}")

print()
print("Last 5 interpolated points:")
print("Depth(m) | fs(kPa) | qc(MPa) | Ic")
print("-" * 35)
for row in result[-5:]:
    print(f"{row[0]:7.1f} | {row[1]:7.3f} | {row[2]:7.3f} | {row[3]:2.1f}")

print()
print("✅ Interpolation integration test completed!")
