# AIS_test

## Problem

A website uses Captchas on a form to keep the web-bots away. However, the captchas it generates, are quite similar each time. You are provided a set of twenty-five captchas, such that, each of the characters A-Z and 0-9 occur at least once in one of the Captchas' text. From these captchas, you can identify texture, nature of the font, spacing of the font, morphological characteristics of the letters and numerals, etc.

## Objective

Create a simple AI model or algorithm to identify the unseen captchas based on provided sample set.

## Solution

Solution of this problem is mainly comprised of two parts:
1. *Pattern recognition*: To create a dictionary that maps every character possibly to occur in captchas with a numpy array that represent its pixel-wise pattern. This is done by converting the given RGB matrix of each image into binary matrices with only 1s and 0s, and then extracting the pixel-wise pattern of each character. 
2. *Pattern matching*: For inference, new image is first converted to an RGB matrix and then compared with each character in its respective numpy array format. Matched characters will form the text of captcha. 

A set of functions are prepared to achieve this goal:

***Pattern recognition***
- *_read_image(filename)*: This function is to read the image .txt file and obtain each image as a 3D grid of pixel values. Each pixel is represented by 3 values representing the RGB components. It will return the image in a numpy array format. 
- *_binarize_image(image_raw)*: This function is to binarize the image with 1s representing image background and 0s representing the actual characters. 
- *_find_pattern()*: This function is to go through all training images and find pixel-wise patterns of each character. It will return a dictionary that maps characters from letter/number to numpy arrays.

***Pattern matching***
- *_match(image_binarized, pattern)*: This function is to slide across the binarized image and compare each region of it with a given character pattern. Comparison result is stored in a matrix with matrix index corresponding to the starting row and column index of each region on the image. If there are any matches, this function will return the starting column indices of matching regions. These column indices will be used for ordering all identified characters at last. 

## Usage

This is an example to further illustrate how unseen captchas could be identified:

This is the unseen captcha located at `input/input100.jpg`: ![This is an image](/input100.jpg). 

Then run the following code:
```
    idf_cap = Captcha()
    idf_cap('input/input100.jpg', 'output/final_result.txt')
```
It will generate a file named final_result.txt that contain a 5-character token YMB1Q in the output folder.

