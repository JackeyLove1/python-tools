import requests

url = "http://localhost:8080/upload/images"  # Replace with the URL of your upload endpoint
image_path = "img.png"  # Replace with the actual path to your image file

# Open the image file in binary mode
with open(image_path, "rb") as image_file:
    # Create a dictionary containing any additional form data required by the server
    payload = {"param1": "value1", "param2": "value2"}  # Replace with your form data if needed
    # Create a dictionary specifying the file name and content type
    files = {"image": (image_file.name, image_file, "image/jpeg")}  # Replace "image/jpeg" with the appropriate content type

    # Send the POST request with the image file and form data
    response = requests.post(url, files=files, data=payload)

# Check the response status
if response.status_code == 200:
    print("Image uploaded successfully.")
else:
    print("Failed to upload image. Status code:", response.status_code)