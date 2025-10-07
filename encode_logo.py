import base64

# The name of the image file you want to encode
image_file = "logo.png"

try:
    # Open the image file in binary read mode
    with open(image_file, "rb") as f:
        # Read the binary data
        binary_data = f.read()
        
        # Encode the binary data to Base64
        base64_encoded_data = base64.b64encode(binary_data)
        
        # Decode the Base64 bytes to a string for printing
        base64_string = base64_encoded_data.decode('utf-8')
        
        # Print the resulting string
        print(f"Your Base64 string is:\n\n{base64_string}")

except FileNotFoundError:
    print(f"Error: The file '{image_file}' was not found.")