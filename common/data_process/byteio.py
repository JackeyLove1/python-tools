from io import BytesIO

file_path = "test.txt"

with open(file_path, 'rb') as f:
    # Read the data from the file
    data = f.read()
    # Create a BytesIO object from the data
    byte_stream = BytesIO(data)
    modified_data = byte_stream.getvalue()
    print(modified_data)

# Write the modified data back to the file
'''
with open(file_path, 'wb') as f:
    f.write(modified_data)
'''
