import numpy as np
import matplotlib.pyplot as plt
import cv2


def gaussian_mask(length=3, sigma=1.):
    ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    return kernel / np.sum(kernel)


def convolute(image, kernel):
    s = kernel.shape + tuple(np.subtract(image.shape, kernel.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(image, shape=s, strides=image.strides * 2)
    return np.einsum('ij,ijkl->kl', kernel, subM)


def first_derivative_edge_detector(image):
    k_x = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])

    k_y = np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]])

    g_x = convolute(image, k_x)
    g_y = convolute(image, k_y)

    return g_x, g_y


def second_derivative_edge_detector(image):
    k_x = np.array([[1, -2, 1],
                    [1, -2, 1],
                    [1, -2, 1]])

    k_y = np.array([[1, 1, 1],
                    [-2, -2, -2],
                    [1, 1, 1]])

    g_x = convolute(image, k_x)
    g_y = convolute(image, k_y)

    return g_x, g_y


def sobel(image):
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    sobel_y_kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])

    s_x = convolute(image, sobel_x_kernel)
    s_y = convolute(image, sobel_y_kernel)

    return s_x, s_y


def prewitt(image):
    prewitt_x_kernel = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])

    prewitt_y_kernel = np.array([[1, 1, 1],
                                 [0, 0, 0],
                                 [-1, -1, -1]])

    p_x = convolute(image, prewitt_x_kernel)
    p_y = convolute(image, prewitt_y_kernel)

    return p_x, p_y


def non_max_suppression(mag, angle_degree):
    m, n = mag.shape
    non_max = np.zeros((m, n), dtype=np.uint8)
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            # Horizontal 0
            if (0 <= angle_degree[i, j] < 22.5) or (157.5 <= angle_degree[i, j] <= 180) or (
                    -22.5 <= angle_degree[i, j] < 0) or (
                    -180 <= angle_degree[i, j] < -157.5):
                b = mag[i, j + 1]
                c = mag[i, j - 1]
            # Diagonal 45
            elif (22.5 <= angle_degree[i, j] < 67.5) or (-157.5 <= angle_degree[i, j] < -112.5):
                b = mag[i + 1, j + 1]
                c = mag[i - 1, j - 1]
            # Vertical 90
            elif (67.5 <= angle_degree[i, j] < 112.5) or (-112.5 <= angle_degree[i, j] < -67.5):
                b = mag[i + 1, j]
                c = mag[i - 1, j]
            # Diagonal 135
            elif (112.5 <= angle_degree[i, j] < 157.5) or (-67.5 <= angle_degree[i, j] < -22.5):
                b = mag[i + 1, j - 1]
                c = mag[i - 1, j + 1]

                # Non-max Suppression
            if (mag[i, j] >= b) and (mag[i, j] >= c):
                non_max[i, j] = mag[i, j]
            else:
                non_max[i, j] = 0

    return non_max


def hysteresis_threshold(non_max_img, low):
    high = low * 2

    m, n = non_max_img.shape

    result = np.zeros((m, n), dtype=np.uint8)

    strong_i, strong_j = np.where(non_max_img >= high)
    zeros_i, zeros_j = np.where(non_max_img < low)
    weak_i, weak_j = np.where((non_max_img <= high) & (non_max_img >= low))

    # Set same intensity value for all edge pixels
    result[strong_i, strong_j] = 255
    result[zeros_i, zeros_j] = 0
    result[weak_i, weak_j] = 75

    # remove weak edges if not connected to a sure edge
    m, n = result.shape
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if result[i, j] == 75:
                if 255 in [result[i + 1, j - 1], result[i + 1, j], result[i + 1, j + 1], result[i, j - 1],
                           result[i, j + 1],
                           result[i - 1, j - 1], result[i - 1, j], result[i - 1, j + 1]]:
                    result[i, j] = 255
                else:
                    result[i, j] = 0

    return result


def canny():
    img = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
    plt.imshow(img, cmap="gray")
    plt.show()

    # 2) apply gauss-blur filter to remove noise

    gray_image_blur = convolute(img, gaussian_mask(sigma=0.5))
    plt.imshow(gray_image_blur, cmap="gray")
    plt.show()

    # 3) run sobel in x and y

    sobel_x, sobel_y = sobel(gray_image_blur)

    # 4) calculate the magnitude & the orientation

    magnitude = np.hypot(sobel_x, sobel_y)
    magnitude = magnitude / magnitude.max() * 255
    magnitude = np.uint8(magnitude)

    theta = np.arctan2(sobel_y, sobel_x)
    angle = np.rad2deg(theta)

    plt.imshow(magnitude, cmap="gray")
    plt.show()

    # 5) Non-Max Suppression

    non_max = non_max_suppression(magnitude, angle)

    # 6) Hysteresis Threshold

    output = hysteresis_threshold(non_max, 10)
    plt.imshow(output, cmap="gray")
    plt.show()

    output = hysteresis_threshold(non_max, 15)
    plt.imshow(output, cmap="gray")
    plt.show()
    output = hysteresis_threshold(non_max, 20)
    plt.imshow(output, cmap="gray")
    plt.show()
    


canny()
