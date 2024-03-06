import cv2

def downscale_image(image_path, scale_factor):
    # Load the image
    image = cv2.imread(image_path)

    # Calculate the new dimensions
    new_width = int(image.shape[1] / scale_factor)
    new_height = int(image.shape[0] / scale_factor)

    # Resize the image
    downscaled_image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA)

    return downscaled_image

# Use the function
downscaled_image = downscale_image('/Users/vojtadeconinck/Downloads/Labyrinth.jpg', 20)

# Save the downscaled image
cv2.imwrite('downscaled_image.jpg', downscaled_image)