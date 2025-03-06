from interpolation import interpolate_cpt_data, format_cpt_data_for_download

# Your input data
input_data = [
    [1, 0.151, 4.6, 19],
    [2, 0.765, 7.5, 19],
    [3, 0.114, 1.2, 19],
    [4, 1.507, 18, 19],
    [5, 1.503, 23.4, 19]
]

# Interpolate the data
interpolated_data = interpolate_cpt_data(input_data, target_interval=0.1)

# Format and print the results
formatted_data = format_cpt_data_for_download(interpolated_data)
print(formatted_data)

# Save to file
with open('interpolated_cpt_data.txt', 'w') as f:
    f.write(formatted_data) 