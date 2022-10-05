# README - Lager, Pradipta

First, you have to prepare one directory for original images, and an empty directory for saving the aligned and cropped images.
By default, the directory for original images is named "Images" and the other directory is named "Cropped_Colored_Images".

## Main
Later, you can run the following commend for default setting and get the final image.
``` 
$python3 main.py
```
## Alignment and Cropped
If you only want to aligned and cropped the original images, run:
```
$python3 main.py -a
```
Further, you set the parameter by using:
```
$python3 main.py -a [original image directory] [max scale degree] [noise epsilon] [alignment epsilon] [directory for cropped images]
```
*note that all the parameter have to be specified.*

## Reconstruct and Tonemapping
If you only want to reconstruct the final image, run:
``` 
$python3 main.py -r
```
Also, you can specify your own parameter
```
$python3 main.py -r [-ng] [directory for cropped images] [pixel number] [g constraint] [Lambda] [high key] [L white]
```
or 
```
$python3 main.py -r [-wg] [directory for cropped images] [pixel number] [g constraint] [Lambda] [max iter] [high key] [L white]
```
*note that all the parameter have to be specified.*

## Parameter setting
* alignment
    1. max scale degree: In order to align the images efficiently, we compress the images into $2^{-(max\ scale\ degree)}$ and recursively align and extended them by $\times 2$.
    2. noise epsilon: The way we aligned the images is to construct the MTB of all images and then compute the correctness. Noise eplison is for us to neglect the pixel with value too close to the median, and thus reduce the noise.
    3. alignment epsilon: While align the images, we choose a reference image and move the other images for $0$~$alignment\ epsilon$ pixels from initial position and compute the correctness. Set the position with highest correctness as the initial position for the next recusion. 

        *note that we have to multiply the position by 2 because when doing the next recursion, we extend the image size.*
* reconstruct
    1. -ng or -wg: "-ng" means "no ghost removal" while "-wg" means "with ghost removal".
    2. pixel_num: While reconstructing the image, we have to select some reference pixels. In our implement, we randomly select $pixel\ num$ pixels.
    3. g_constraint: The response curve we get can be varified by apply some translation. Thus we use g_constraint to fix the curve. $g(g\_constraint) = 0$
    4. Lambda: While computing the response curve, Lambda is the weight for the smoothness of the curve.
    5. max_iter: If we choose "-wg", then we have to determine how many iterations we want for removing the ghost.
    6. high_key: This is a parameter for the luminace of our tonemapping function.
    7. L_white: This is a parameter for the luminace of our tonemapping function.