import os 

import cv2 

import matplotlib.pyplot as plt 

import numpy as np 

from skimage.exposure import rescale_intensity 

 

def identify_green_apples(image_path): 

 

    apple = cv2.imread(image_path) 

 

    # Convert the image to HSV color space 

    hsv_image = cv2.cvtColor(apple, cv2.COLOR_BGR2HSV) 

    plt.imshow(hsv_image) 

    # Define the lower and upper bounds for red color in HSV 

    lower_green = np.array([0, 127, 214]) 

    upper_green = np.array([255, 255, 255]) 

 

    # Create a mask for red pixels 

    green_mask = cv2.inRange(hsv_image, lower_green, upper_green) 

 

    # Find contours in the mask 

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

 

    # Draw the contours on the original image and count the apples 

    green_count = 0 

    for contour in contours: 

        area = cv2.contourArea(contour) 

        if area > 20:  # Adjust the threshold based on your image 

            cv2.drawContours(apple, [contour], -1, (0, 255, 0), 2) 

            green_count += 1 

 

    # Display the total number of red apples 

    cv2.putText(apple, f"Total green apples: {green_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) 

    print(f"Total green apples detected: {green_count}") 

 

    plt.imshow(cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)) 

    plt.show() 

    return green_count 

 

def identify_red_apples(image_path): 

 

    apple = cv2.imread(image_path) 

 

    # Convert the image to HSV color space 

    hsv_image = cv2.cvtColor(apple, cv2.COLOR_BGR2HSV) 

    plt.imshow(hsv_image) 

    # Define the lower and upper bounds for red color in HSV 

    # lower_red = np.array([0, 54, 54]) 

    # upper_red = np.array([20, 255, 255]) 

 

    lower_red = np.array([157, 35, 0]) 

    upper_red = np.array([255, 255, 255]) 

    # Create a mask for red pixels 

    red_mask = cv2.inRange(hsv_image, lower_red, upper_red) 

 

    # Find contours in the mask 

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

 

    # Draw the contours on the original image and count the apples 

    red_count = 0 

    for contour in contours: 

        area = cv2.contourArea(contour) 

        if area > 15:  # Adjust the threshold based on your image 

            cv2.drawContours(apple, [contour], -1, (0, 255, 0), 2) 

            red_count += 1 

 

    # Display the total number of red apples 

    cv2.putText(apple, f"Total red apples: {red_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) 

    print(f"Total red apples detected: {red_count}") 

 

    plt.imshow(cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)) 

    plt.show() 

    return red_count 

 

 

def read_text_file(file_path): 

    with open(file_path, 'r') as file: 

        return file.read() 

 

def identify_apples(image_path, color_mask_bounds): 

    apple = cv2.imread(image_path) 

    hsv_image = cv2.cvtColor(apple, cv2.COLOR_BGR2HSV) 

    lower_bound, upper_bound = color_mask_bounds 

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound) 

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

 

    apple_count = 0 

    for contour in contours: 

        area = cv2.contourArea(contour) 

        if area > 25:  #adjust this value its like a filter to adjust pixel values so it consideres it as a pixel 

            apple_count += 1 

 

    return apple_count 

 

def	process_apples_in_directory(directory,	color_mask_bounds,	count_txt_directory): 

    apple_counts = [] 

    actual_counts = [] 

    image_files = [file for file in os.listdir(directory) if file.endswith('.png')] 

 

    for image_file in image_files: 

        image_path = os.path.join(directory, image_file) 

        detected_count = identify_apples(image_path, color_mask_bounds) 

        apple_counts.append(detected_count) 

 

        base_name = os.path.splitext(image_file)[0] 

        txt_file_path = os.path.join(count_txt_directory, base_name + '.txt') 

        actual_count = int(read_text_file(txt_file_path)) 

        actual_counts.append(actual_count) 

 

        print(f"{base_name}: Detected {detected_count}, Actual {actual_count}") 

 

    return apple_counts, actual_counts 

 

def calculate_rms(detected_counts, actual_counts): 

    sum_of_squares = 0 

 

    for detected, actual in zip(detected_counts, actual_counts): 

        difference = detected - actual 

        sum_of_squares += difference ** 2 

    mean_of_squares = sum_of_squares / len(detected_counts) 

    return np.sqrt(mean_of_squares) 

 

def plot_accuracy(actual_counts, detected_counts, title, color): 

    plt.figure(figsize=(10, 6)) 

    plt.scatter(actual_counts, detected_counts, color=color) 

    plt.plot([0, max(actual_counts)], [0, max(actual_counts)], 'b--') 

    plt.xlabel('Actual Counts') 

    plt.ylabel('Detected Counts') 

    plt.title(title) 

    plt.legend(['Line of Perfect Accuracy', 'Detected Counts']) 

    plt.grid(True) 

    plt.show() 

 

 

# Directories 

count_txt_directory = '/content/drive/MyDrive/Colab Notebooks/detection/detection/count_labels/labels' 

green_image_directory = '/content/drive/MyDrive/Colab Notebooks/detection/detection/Accuracy/Accuracy/Green_Apples' 

red_image_directory = '/content/drive/MyDrive/Colab Notebooks/detection/detection/Accuracy/Accuracy/Red_Apples' 

 

# Color bounds 

green_bounds = (np.array([0, 127, 214]), np.array([255, 255, 255])) 

red_bounds = (np.array([157, 35, 0]), np.array([255, 255, 255])) 

 

# Process Green Apples 

green_counts, green_actuals = process_apples_in_directory(green_image_directory, green_bounds, count_txt_directory) 

green_rms = calculate_rms(green_counts, green_actuals) 

print(f"RMS Error for Green Apples: {green_rms}") 

plot_accuracy(green_actuals, green_counts, 'Accuracy Plot for Green Apples', color='green') 

 

 

# Process Red Apples 

red_counts, red_actuals = process_apples_in_directory(red_image_directory, red_bounds, count_txt_directory) 

red_rms = calculate_rms(red_counts, red_actuals) 

print(f"RMS Error for Red Apples: {red_rms}") 

plot_accuracy(red_actuals, red_counts, 'Accuracy Plot for Red Apples', color='red') 