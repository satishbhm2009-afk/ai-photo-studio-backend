import cv2
import os


def enhance_image(input_path, output_path):

    image = cv2.imread(input_path)

    if image is None:
        return False


    # Noise reduction
    image = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        10,
        10,
        7,
        21
    )


    # Color enhancement
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )


    l,a,b = cv2.split(lab)


    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8,8)
    )


    l = clahe.apply(l)


    enhanced = cv2.merge(
        (l,a,b)
    )


    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )


    # Sharpen
    kernel = [
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ]


    enhanced = cv2.filter2D(
        enhanced,
        -1,
        kernel
    )


    cv2.imwrite(
        output_path,
        enhanced
    )


    return True
