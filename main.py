#importing the module needed
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc,ndimage
import math

# reading the image with imread("image directory")
image = cv2.imread("image.jpeg")

#get the size of image
size = image.shape
Area = size[0]* size[1]

# Splitting channels from image

fig, axes = plt.subplots(1, 4)
 
# image[:, :, 0] is R channel, replace the rest by 0.
imageR = image.copy()
imageR[:, :, 1:3] = 0
axes[0].set_title('R channel')
axes[0].imshow(imageR)
 
# image[:, :, 1] is G channel, replace the rest by 0.
imageG = image.copy()
imageG[:, :, [0, 2]] = 0
axes[1].set_title('G channel')
axes[1].imshow(imageG)
 
# image[:, :, 2] is B channel, replace the rest by 0.
imageB = image.copy()
imageB[:, :, 0:2] = 0
axes[2].set_title('B channel')
axes[2].imshow(imageB)

image_org = image.copy()
axes[3].set_title('merged channel')
axes[3].imshow(image_org)
#plt.show()
plt.savefig('RGB_color.png')
plt.clf()

# rgb splittig and merging
b, g, r = cv2.split(image)
#cv2.imshow("blue",b)
cv2.imwrite("channels/blue channel.jpg",b )
#cv2.imshow("green",g)
cv2.imwrite("channels/green channel.jpg",g )
#cv2.imshow("red",r)
cv2.imwrite("channels/red channel.jpg",r )
bgr = cv2.merge([b, g, r])
#cv2.imshow("RGB merged",bgr)
cv2.imwrite("channels/merged RGB.jpg",bgr )

# calculations
print ("image Area: ",Area)
print("image size: ","width:",size[0],"height:",size[1])

#changing image to gray scale image
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#showing the grayscaled image
#cv2.imshow("grayscale ",grayscale )
cv2.imwrite("grayscale.jpg", grayscale)



#showing image grayscale histogram
plt.hist(grayscale.ravel(), bins=256)
plt.savefig('histograms/histogram.png')
plt.clf()


#histogram normilized
def imhist(im):
  # calculates normalized histogram of an image
	m, n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[im[i, j]]+=1
	return np.array(h)/(m*n)

plt.hist(imhist(grayscale), bins=256)
plt.savefig('histograms/histogram_normalized.png')
plt.clf()


#histogram equalization
def histeq(im,nbr_bins=256):

   #get image histogram
   imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = np.interp(im.flatten(),bins[:-1],cdf)

   return im2.reshape(im.shape), cdf

plt.hist(histeq(grayscale), bins=256)
plt.savefig('histograms/histogram_equalized.png')
plt.clf()

# averaging 3x3
kernel_size3x3 = (3,3)
blur3x3 = cv2.blur(grayscale,kernel_size3x3)
#cv2.imshow("average filter 3x3",blur3x3 )
cv2.imwrite("average filter 3x3.jpg", blur3x3)

# averaging 5x5
kernel_size5x5 = (5,5)
blur5x5 = cv2.blur(grayscale,kernel_size5x5)
#cv2.imshow("average filter 5x5",blur5x5 )
cv2.imwrite("average filter 5x5.jpg", blur5x5)


# median blur 3x3
median3x3 = cv2.medianBlur(grayscale,3)
#cv2.imshow("median filter 3x3",median3x3 )
cv2.imwrite("median filter 3x3.jpg", median3x3)

# median blur 5x5
median5x5 = cv2.medianBlur(grayscale,5)
#cv2.imshow("median filter 5x5",median5x5 )
cv2.imwrite("median filter 5x5.jpg", median5x5)



# gaussian filter
gaussian = cv2.GaussianBlur(grayscale,(5,5),0)
#cv2.imshow("gaussian filter",gaussian )
cv2.imwrite("gaussian filter.jpg",gaussian )

list_output = [blur3x3,blur5x5,median3x3,median5x5,gaussian]
# laplacian edge detector
n= 0
for img in list_output:
    n+=1
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    cv2.imwrite("laplacian/laplacian_{}.jpeg".format(n),laplacian )
    #cv2.imshow("laplacian_{}".format(n),laplacian )

n= 0
for img in list_output:
    n+=1
    sobel = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)  
    cv2.imwrite("sobel/sobel_{}.jpeg".format(n),sobel )
    #cv2.imshow("sobel_{}".format(n),sobel )



# RGB TO HSI
def RGB_TO_HSI(img):

    with np.errstate(divide='ignore', invalid='ignore'):

        #Load image with 32 bit floats as variable type
        bgr = np.float32(img)/255

        #Separate color channels
        blue = bgr[:,:,0]
        green = bgr[:,:,1]
        red = bgr[:,:,2]


        #Calculate Intensity
        def calc_intensity(red, blue, green):
            return np.divide(blue + green + red, 3)

        #Calculate Saturation
        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)
            return saturation

        #Calculate Hue
        def calc_hue(red, blue, green):
            hue = np.copy(red)

            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                math.sqrt((red[i][j] - green[i][j])**2 +
                                        ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])

                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

            return hue

        #Merge channels into picture and return image
        H = calc_hue(red, blue, green)
        S = calc_saturation(red, blue, green)
        I = calc_intensity(red, blue, green)
        hsi = cv2.merge((H,S ,I ))

        return hsi ,H,S

HSI_Image,H,S = RGB_TO_HSI(image)
cv2.imshow("HSI Image.png",HSI_Image )
#cv2.imwrite("HSIImage.png", HSI_Image)
cv2.imshow("H",H )
#cv2.imwrite("H.png",H)
cv2.imshow("S",S )
#cv2.imwrite("S.png",S )


cv2.waitKey(0)