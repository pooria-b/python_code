import cv2
import numpy as np
from skimage.filters import roberts

class Filters:

  def __init__(self, img):
    self.img = img
  
  def HPF_Filter(self, r):
    rows, cols = self.img.shape
    crow, ccol = int(rows/2), int(cols/2)
    # matrix of ones to seperates frequencies spectrum
    # the '2' below, represents imaginary and real channels
    mask = np.ones((rows, cols, 2) , np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_center = (x - center[0])**2 + (y - center[1])**2 <= r*r
    mask[mask_center] = 0
    dft = cv2.dft(np.float32(self.img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    fshift = mask * dft_shift
    fshift_magnitude_spectrume = 20 * np.log((cv2.magnitude(fshift[:,:,0] , fshift[:,:,1])) + 1)
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = 20 * np.log((cv2.magnitude(img_back[:,:,0] , img_back[:,:,1])) + 1)

    return img_back


  def threshold(self, edge_detected_img):
    ret, thresh = cv2.threshold(edge_detected_img , 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh.astype('uint8')

  def edge_detector(self, threshold_img):
    edge_detected_img = roberts(threshold_img)
    return (edge_detected_img*255).astype('uint8')

  def LPF_Filter(self, r):
    
    rows, cols = self.img.shape
    crow, ccol = int(rows/2), int(cols/2)
    # matrix of ones to seperates frequencies spectrum
    # the '2' below, represents imaginary and real channels
    mask = np.ones((rows, cols, 2) , np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_center = (x - center[0])**2 + (y - center[1])**2 >= r*r
    mask[mask_center] = 0
    dft = cv2.dft(np.float32(self.img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    fshift = mask * dft_shift
    fshift_magnitude_spectrume = 20 * np.log((cv2.magnitude(fshift[:,:,0] , fshift[:,:,1])) + 1)
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = 20 * np.log((cv2.magnitude(img_back[:,:,0] , img_back[:,:,1])) + 1)
    return img_back