# binarization, morphological operations, image filtrations  
Name: Erik Matovič  
Methods used: list of important methods

## Assignment
Load histology image from breast cancer dataset  - [image1](https://drive.google.com/file/d/15o6Dl25P6ern4JJkjArxpPdi8UPLcF6p/view), [image2](https://drive.google.com/file/d/1hHTTYJX6qyzY0BJbLQ21bx69Mj7LFrOv/view), [image3](https://drive.google.com/file/d/1UXCh_8nucjo5zA7-WqrJ_JNzmQkhO5am/view). 
Eliminate noise and binarize the image using morphological operations and contour analysis. 
Try to programmatically mark every cell and sum the total count. There are multiple solutions, use your imagination. Document each attempt, 
even if it is unsuccessful ([documentation example](https://sites.google.com/stuba.sk/vgg/computer-vision/solution-training-task?authuser=0)).
Use [OpenCV documentation](https://docs.opencv.org/4.7.0/), below are some tips you may try.
Optional Datasets: [Beer bubbles](https://drive.google.com/file/d/1jg_o5izpma-RUc8296SOjPau5ypruWnE/view), [red blood cells](https://drive.google.com/drive/folders/1FThJGItE_jSzne2LgcStj9Q4sLILPDWj)
Choose at least 2 images (of your choice) for this Assignment !

## Usage
To run Jupyter Notebook, you need OpenCV and matplotlib. You can install them using pip:  
```bash
pip install opencv-python matplotlib
```

## Solution
### 1. Load image and convert to grayscale
After loading images, we downsized histological images up to half their size and converted them to grayscale.

Function for resizing images:
```python
def resize_img(img: cv2.Mat, scale_percent: int) -> cv2.Mat:
    """
    Resizing images.
    :param: img - image
    :param: scale_percent - percent by which the image is resized
    :return: Resized image
    """
    # calculate the scale percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    img_resize = cv2.resize(img, dsize)
    return img_resize
```

Conversion to grayscale:
```python
img1_grayscale = cv2.cvtColor(img1_resize, cv2.COLOR_BGR2GRAY)
img2_grayscale = cv2.cvtColor(img2_resize, cv2.COLOR_BGR2GRAY)
img3_grayscale = cv2.cvtColor(img3_resize, cv2.COLOR_BGR2GRAY)
```

Original images:  
<p align="center">
	<img src="./outputs/images.png">
</p>

Grayscale images:  
<p align="center">
	<img src="./outputs/images_grayscale.png">
</p>

### 2. Image pre-processing  
Noise removal with blurring image:  
```python
img1_blur = cv2.blur(img1_grayscale, (5,5))
img1_gauss = cv2.GaussianBlur(img1_grayscale, (5,5), 0)
img1_median = cv2.medianBlur(img1_grayscale, 5)
img1_bilateral = cv2.bilateralFilter(img1_grayscale, 9, 75, 75)
```

<p align="center">
	<img src="./outputs/blur.png">
</p>


### 3. Binarization
Binarization of image 1 using thresholding & inrage:  
```python
retValue, img1_threshold = cv2.threshold(img1_gauss, 127, 255, cv2.THRESH_BINARY)
retValue, img1_otsu = cv2.threshold(img1_gauss, 0, 255, cv2.THRESH_OTSU)
img1_adaptiveThreshold = cv2.adaptiveThreshold(img1_gauss, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
img1_inrange = cv2.inRange(img1_gauss, 127, 255)
```

<p align="center">
	<img src="./outputs/binarization.png">
</p>

Binarization of image 1 using Sobel edge detection:  
```python
grad_x = cv2.Sobel(src=img1_gauss, ddepth=-1, dx=1, dy=0, ksize=3)
grad_y = cv2.Sobel(src=img1_gauss, ddepth=-1, dx=0, dy=1, ksize=3)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
```

Binarization of image 1 using Laplacian & Canny edge detection:  
```python
img1_laplac = cv2.Laplacian(src=img1_gauss, ddepth=-1, ksize=7)
img1_canny = cv2.Canny(img1_gauss, 30, 100)
```

Compare edge detection with adaptive thresholding:
<p align="center">
	<img src="./outputs/edge_detection.png">
</p>

### 4. Cell segmentation
Using morphological operations:
```python
img1_dilate = cv2.dilate(img1_canny,(-1, -1), 3)
img1_erode = cv2.erode(img1_canny, kernel=(3,3))
img1_distanceTransform = cv2.distanceTransform(src=img1_canny, distanceType=cv2.DIST_L2, maskSize=5)
```

<p align="center">
	<img src="./outputs/1_morphological_op.png">
</p>

Contours analysis of image 1 using:
```python
img1_result = img1_resize.copy()
img1_contours, img1_hierarchy = cv2.findContours(img1_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cellIndex_img1 = 0

for i in range(0, len(img1_contours)):
   if cv2.contourArea(img1_contours[i]) > 10:
      cellIndex_img1 += 1
      cv2.drawContours(img1_result, img1_contours, i, (0, 255, 0), 4)
   i += 1
```

<p align="center">
	<img src="./outputs/1_cells.png">
</p>

### Filtration

