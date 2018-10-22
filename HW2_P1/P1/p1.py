import numpy as np

def conv(image, kernel):
    M = len(kernel)
    R = int((M-1)/2)
    H = len(image)
    W = len(image[0])

    conved_image = np.zeros((H,W))

    for y in range(H):
        for x in range(W):
            low = R - y
            high = y + R + 1 - H
            left = R - x
            right = x + R + 1 - W

            window = image[max(y-R, 0):min(y+R+1,H),max(x-R, 0):min(x+R+1,W)]
            k = kernel[max(0, 0+low):min(M, M-high), max(0, 0+left):min(M, M-right)]
            conved_image[y][x] = min(np.multiply(window,k).sum(),255)

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
