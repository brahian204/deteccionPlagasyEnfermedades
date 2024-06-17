import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from PIL import Image, ImageFilter,ImageEnhance

def filtros(img):   
    imagen = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flip_1 = cv2.flip(img,1)
    gaus= ndimage.median_filter(img, 3)
    image =serpia(img)
    lap = laplace(img)
    ap = apertura(img)
    img_invert_gray = cv2.bitwise_not(lap)
    rojo = color(imagen,2)
    verde = color(imagen,1)
    salt = salt_pepper(0.025,gray,img)
    nit = nitidez(img)
    his = histograma(img)
    war,cold = warmImage(img)
    ret = retro(img)
    brillo2,brillo1 = brillo(img)
    com = comic(lap)
    i,a = backgroud(gray)
    car,car1,ima = cartoon(img)
    new = ImgEnhace(img)
    # return [img,image,lap,brillo1,brillo2,ap,nit,his,war,cold,gaus,new]
    return [img,lap,brillo1,brillo2,ap,nit,his,cold, new]

def serpia(image):
    image = np.array(image, dtype=np.float64) # converting to float to prevent loss
    image = cv2.transform(image, np.matrix([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                   [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    image[np.where(image > 255)] = 255 # normalizing values greater than 255 to 255
    image = np.array(image, dtype=np.uint8)    
    return image  

def laplace(foto):
    w = 1/3
    foto = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
    foto = Image.fromarray(foto.astype(np.uint8))
    coeficientes = [1, 1, 1, 1, -8, 1, 1, 1, 1]
    datos_laplace = foto.filter(ImageFilter.Kernel((3,3), coeficientes, 1)).getdata()
    datos_imagen = foto.getdata()
    datos_nitidez = [datos_imagen[x] - (w * datos_laplace[x]) for x in range(len(datos_laplace))]   
    imagen_nitidez = Image.new('L', foto.size)
    imagen_nitidez.putdata(datos_nitidez)  
    imagen_nitidez = np.array(imagen_nitidez, dtype=np.uint8)
    imagen_nitidez = np.reshape(imagen_nitidez,(512,512,1))
    return imagen_nitidez 

def apertura(img):
    img = img.astype(np.uint8)
    img_to_yuv = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    contraste = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2RGB)
    return contraste

def salt_pepper(prob,gray,image):
    s_vs_p = 0.05
    output = np.copy(image)
    num_salt = np.ceil(prob * gray.size * (s_vs_p))
    coords = [np.random.randint(0, i-1, int(num_salt))for i in gray.shape]
    output[tuple(coords)] = 0
    num_salt = np.ceil(prob * gray.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i-1, int(num_salt))for i in gray.shape]
    output[tuple(coords)] = 1
    return output    

def nitidez(image):
     kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
     sharpness = cv2.filter2D(image,-1,kernel)
     return sharpness  

def histograma(image):
     image = image.astype(np.uint8)    
     img_to_yuv = cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
     img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])  
     img_to_yuv = cv2.cvtColor(img_to_yuv,cv2.COLOR_YUV2RGB)   
     return img_to_yuv

def warmImage(img):
    img = img.astype(np.uint8)
    def spreadLookupTable(x, y):
        spline = UnivariateSpline(x, y)
        return spline(range(256))
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 90, 150, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    warmImage = cv2.merge((red_channel, green_channel, blue_channel))   
    red_channel, green_channel, blue_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    coldImage = cv2.merge((red_channel, green_channel, blue_channel))
    return warmImage,coldImage 

def retro(img):
     rows, cols = img.shape[0:2]
     kernel_x = cv2.getGaussianKernel(cols,200)
     kernel_y = cv2.getGaussianKernel(rows,200)
     kernel = kernel_y * kernel_x.T
     filter = 255 * kernel / np.linalg.norm(kernel)  
     vintage_im = np.copy(img) 
     for i in range(3):
         vintage_im[:,:,i] = vintage_im[:,:,i] * filter 
     return vintage_im 

def brillo(image):
     cols, rows, channel = image.shape
     brightness = np.sum(image) / (255 * cols * rows * channel)
     minimum_brightness = 0.45
     alpha = brightness / minimum_brightness
     bright_img = cv2.convertScaleAbs(image, alpha = alpha, beta = 1 * (1 - alpha))
     bright_img1 = cv2.convertScaleAbs(image, alpha = 0.8, beta = 15)    
     return bright_img1,bright_img  

def color (image,number):
      image = image.astype(np.uint8)
      image = tone(image,number) 
      return image 

def exponential_function(channel, exp):
    table = np.array([min((i**exp), 255) for i in np.arange(0, 256)]).astype("uint8") # creating table for exponent
    channel = cv2.LUT(channel, table)
    return channel

def tone(image, number):
    for i in range(3):
        if i == number:
            image[:, :, i] = exponential_function(image[:, :, i], 0.85) # applying exponential function on slice
        else:
            image[:, :, i] = 0 # setting values of all other slices to 0
    return image

def comic(img):
     img_invert = cv2.bitwise_not(img)
     img_smoothing = cv2.GaussianBlur(img_invert, (21,21), sigmaX=0, sigmaY=0)
     def dodge(x,y):
         return cv2.divide(x, 255-y, scale=256)
     final_img=dodge(img, img_smoothing)   
     return final_img 

def cartoon(image):    
     image = image.astype(np.uint8)
     img_invert = cv2.bitwise_not(image)
     edges1 = cv2.bitwise_not(cv2.Canny(image, 110, 220)) 
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     gray = cv2.medianBlur(gray, 5) 
     edges2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3,3) 
     dst = cv2.edgePreservingFilter(image, flags=2, sigma_s=64, sigma_r=0.25) 
     cartoon1 = cv2.bitwise_and(dst, dst, mask=edges1)
     cartoon2 = cv2.bitwise_and(dst, dst, mask=edges2)
     return cartoon2,cartoon1,img_invert

def imadjust(img, In=(0,1.0), Out=(0,1.0), gamma=1.0):
    "J = low_out +(high_out - low_out).* ((I - low_in)/(high_in - low_in)).^ gamma"
    low_in,high_in = In
    low_out, high_out = Out
 
    low_in *= 255.0
    high_in *= 255.0
 
    low_out *= 255.0
    high_out *= 255.0    
    
    k = (high_out - low_out) / (high_in - low_in)
         # Gamma transformation table
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    h,w = img.shape[:2]
    imgOut = np.zeros((h,w), np.uint8)
    
    for r in range(h):
        for c in range(w):
            if img[r,c] <= low_in:
                imgOut[r,c] = low_out                
            elif img[r,c] > high_in:
                imgOut[r,c] = high_out
            else:
                res = int(k*(img[r,c]-low_in) + low_out)
                imgOut[r,c] = table[res]#Check table
               
    return imgOut    

def backgroud(img):
    kernel = np.ones((5,5),np.uint8)
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    I23 = img - background
    I23 = imadjust(I23,(0,1), (0,1), 0.5)  
    rangefiltG = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    rangefiltG = imadjust(rangefiltG,(0,1), (0,1), 0.5) 
    return I23,rangefiltG  

def ImgEnhace(img):
    img = img.astype(np.uint8)
    color = ImageEnhance.Color(Image.fromarray(img))
    color = color.enhance(0.8)
    color =ImageEnhance.Sharpness(color)
    color = color.enhance(0.8)
    color = ImageEnhance.Contrast(color)
    color = color.enhance(1.2)
    return np.array(color).astype('uint8')


