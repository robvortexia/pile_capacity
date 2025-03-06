from interpolation import process_uploaded_cpt_data, format_cpt_data_for_download

# Simulate a file upload with specific depths
file_content = """1.0 0.151 4.6 19
2.0 0.765 7.5 19
3.1 0.114 1.2 19
3.2 1.507 18 19
3.3 1.503 23.4 19
4.0 1.507 18 19
4.5 1.503 23.4 19"""

# Process the uploaded data
processed_data, warning = process_uploaded_cpt_data(file_content)

# Print any warnings
print("Processing Result:", warning)

# Print the processed depths
print("\nProcessed depths:")
for row in processed_data:
    print(f"Depth: {row[0]:.1f}m")

# Save the processed data
formatted_data = format_cpt_data_for_download(processed_data)
with open('specific_depths_data.txt', 'w') as f:
    f.write(formatted_data) 