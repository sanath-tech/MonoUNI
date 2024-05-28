from PIL import Image


for i in range(0,197):
    # Replace 'path/to/your/image.png' with the path to your PNG image
    file_number = f'{i:06}'
    input_path = f"/home/workstation/Documents/output_images/frame_{file_number}.png"
    #input_path = '/home/workstation/Documents/MonoUNI/test_field_data/image_2/mast_22_left.png'
    # Replace 'path/to/your/image.jpg' with the desired path for the output JPG image
    output_path = f"/home/workstation/Documents/output_images/frame_{file_number}.jpg"

    # Open the PNG image
    png_image = Image.open(input_path)

    # Convert PNG to JPG
    jpg_image = png_image.convert('RGB')

    # Save the image as JPG
    jpg_image.save(output_path, 'JPEG')

    print(f'Image successfully converted and saved to {output_path}')
