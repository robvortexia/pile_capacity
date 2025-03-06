from interpolation import process_uploaded_cpt_data, format_cpt_data_for_download

# Simulate a file upload with your data
file_content = """1 0.151 4.6 19
2 0.765 7.5 19
3 0.114 1.2 19
4 1.507 18 19
5 1.503 23.4 19"""

# Process the uploaded data
processed_data, original_interval = process_uploaded_cpt_data(file_content)

# Print information about the processing
print(f"Original interval between points: {original_interval:.3f}m")
print(f"Number of points in processed data: {len(processed_data)}")
print("\nFirst few points of processed data:")
for row in processed_data[:5]:
    print(f"Depth: {row[0]:.3f}m, fs: {row[1]:.3f}kPa, qc: {row[2]:.3f}MPa, Ic: {row[3]:.3f}")

# Save the processed data
formatted_data = format_cpt_data_for_download(processed_data)
with open('processed_cpt_data.txt', 'w') as f:
    f.write(formatted_data) 