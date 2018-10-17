import numpy as np

def conv(image, kernel):
    M = len(kernel)
    R = int((M-1)/2)
    H = len(image)
    W = len(image[0])

    conved_image = np.zeros((H,W))

    for y in range(len(image)):
        for x in range(len(image[y])):
            for yk in range(len(kernel)):
                yi = y + yk - R
                if not (yi < 0 or yi >= H):
                    for xk in range(len(kernel[yk])):
                        xi = x + xk - R
                        if not (xi < 0 or xi >= W):
                            conved_image[y][x] += kernel[yk][xk] * image[yi][xi]

    return conved_image

def downsample(image):
    H = len(image)
    W = len(image[0])

    downsampled_image = np.zeros((int(H/2),int(W/2)))

    for y in range(len(downsampled_image)):
        for x in range(len(downsampled_image[0])):
            downsampled_image[y][x] = image[y*2][x*2]

    return downsampled_image

def gaussianPyramid(image, W, k):
    G = [image]
    for i in range(k-1):
        G.append(conv(downsample(G[i]),W))
    return G

def upsample(image):
    H = len(image)
    W = len(image[0])

    upsampled_image = np.zeros((H*2,W*2))

    for y in range(len(image)):
        for x in range(len(image[0])):
            upsampled_image[y*2][x*2] = image[y][x]
    return upsampled_image

def laplacianPyramid(G, W):
    L = []
    for i in range(len(G)-1):
        L.append(G[i] - conv(upsample(G[i+1]), 4*W) + 128)
    return L
