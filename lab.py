#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image

#LAB1 FUNCTIONS

def get_pixel(image, x, y):
    '''
    Gets a pixel at a specific location of the image representation and returns it.
    If the pixel requested is out-of-bounds for the image, the pixel value returned is the
    value of the closest existing pixel to it.  

    Parameters
    ----------
    image : (dict) a python representation of the image 
    x : (int) the x coordinate of the chosen pixel on the image
    y : (int) the y coordinate of the chosen pixel on the image


    Returns
    -------
    A float or an int that represents the pixel value of the chosen coordinates's pixel.

    '''
    #if the x coordinate is out of bounds, set it either to the highest or the lowest
    #possible x value
    if (x<0): 
        x = 0
    elif (x>(image['height']-1)): 
        x = image['height']-1
    #if the y coordinate is out of bounds, set it either to the highest or the lowest
    #possible y value
    if (y<0):
        y = 0
    elif (y> (image['width']-1)):
        y = image['width']-1
    
    return image['pixels'][(x)*image['width'] + y]


def set_pixel(image, x, y, c):
    '''
    Sets a pixel at a specific location of the image representation to a 
    chosen value.

    Parameters
    ----------
    image : (dict) a python representation of the original image 
    x : (int) the x coordinate of the chosen pixel on the image
    y : (int) the y coordinate of the chosen pixel on the image
    c : (float or int) the value the pixel should be set to

    Returns
    -------
    None.

    '''
    image['pixels'][(x)*image['width'] + y] = c



def apply_per_pixel(image, func):
    '''
    Applies a function to every single pixel of an image and returns a copy 
    of it.

    Parameters
    ----------
    image : (dict) a python representation of the original image
    func : (function) the location of the function that will be applied to the pixels

    Returns
    -------
    result : (dict) a copy of the original image dictionary but with all of 
    its pixels having had func applied to them

    '''
    #creates a new image representation to be returned with the same height and
    #width as of the original image, but with all of its pixel values equal to 0 as placeholders
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [0 for _ in range(image['height']*image['width']) ],
    }
    #iterates over every pixel value
    for x in range(image['height']):
        for y in range(image['width']):
            
            #gets a pixels color and applies a function to it
            color = get_pixel(image, x, y)
            newcolor = func(color)
            
            set_pixel(result, x, y, newcolor) #changes that pixel to its new value with the function applied
            
    return result


def inverted(image):
    '''
    Returns an inverted version of the image, with all of its pixel values
    reflected about the middle grey value. 
    
    Parameter:
        * image (dict) = A python representation of the original image to be inverted
    
    Returns
        * An inverted representation of the image without changing the original one
    '''
    return apply_per_pixel(image, lambda c: 255-c)


# HELPER FUNCTIONS

def correlate(image, kernel):
    """
    Compute the result of correlating the given image with the given kernel.

    The output of this function should have the same form as a 6.009 image (a
    dictionary with 'height', 'width', and 'pixels' keys), but its pixel values
    do not necessarily need to be in the range [0,255], nor do they need to be
    integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    
    A list that just all of the kernel values in a row (same format as the pixels)
    
    """
    #creates a new image representation to be returned with the same height and
    #width as of the original image, but with a copied version of the list of pixel values
    newPix = image['pixels'][:]
    newImage = {
        'height': image['height'],
        'width': image['width'],
        'pixels': newPix
    }

    length = int((len(kernel))**0.5) #the square root of the kernel gives its length
    midCords = int(length/2.0 -0.5) #the x and y coordinate of the middle number

    #iterates over all pixel values
    for x in range(image['height']):
        for y in range(image['width']):
            tempColor = 0 #will store the value for the final pixel value at this location
            
            #for each pixel value, applies all kernel values to it
            for xKer in range(length):
                for yKer in range(length):
                        
                    kerValue = kernel[xKer*length + yKer]
                    
                    #finds the coordinates for the pixel the kernel is multiplying by
                    tempPixX = x - midCords + xKer
                    tempPixY = y - midCords + yKer
                    
                    tempColor += get_pixel(image, tempPixX, tempPixY)*kerValue
            
            #sets the pixel value after all the kernel values were applied to it
            set_pixel(newImage, x, y, tempColor)
    
    return newImage

    
    


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    #checks all pixel values on the list
    for pix in range(image['width']*image['height']):
        image['pixels'][pix] = round(image['pixels'][pix]) #rounds the pixels to ints
        
        #ensures the lowest possible value is 0 and highest 255
        if (image['pixels'][pix] < 0): 
            image['pixels'][pix] = 0
        elif (image['pixels'][pix] > 255):
            image['pixels'][pix] = 255


# FILTERS

def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
 
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    val = 1/(n*n)
    kernel = []
    for i in range(n*n):
        kernel.append(val)


    # then compute the correlation of the input image with that kernel

    newImage = correlate(image, kernel)
    

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    round_and_clip_image(newImage)
    return newImage

def sharpened(image, n):
    '''
    It subtracts a blurred version of the image, making it sharper. 

    Parameters
    ----------
    * image (dict) = A python representation of the original image
    * n (int) = the size of the blur kernel that should be used to generate the blurred copy of the image

    Returns
    -------
    finalImage (dict) : a sharpened version of the image.

    '''
    #uses the blurred function to create the blurred copy of the image
    newImageblurred = blurred(image, n)
    #makes a list filled with 0 so that 
    newPix = [0 for _ in range(len(newImageblurred['pixels'])) ]
    
    #adds to the pixel list values of the formula S = 2I - B
    for i in range(len(newPix)):
        newPix[i] = 2* image['pixels'][i] - newImageblurred['pixels'][i]
    
    #makes a new image dic that will be returned
    finalImage = {
        'height': image['height'],
        'width': image['width'],
        'pixels': newPix
    }
    round_and_clip_image(finalImage)
    return finalImage

def edges(image):
    '''
    Implements a filter called Solbert Operator, which helps in edge detection 
    by emphasizing the edges on the image. 

    Parameters
    ----------
    * image (dict) = A python representation of the original image

    Returns
    -------
    finalImage (dict): a python representation of an image with its edges
    emphasized

    '''
    #inputs the two given kernels that will be utilized
    Kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    Ky = [-1, -2, -1, 0, 0, 0, 1,  2,  1]
    
    #corellates each kernel with the image and returns a copy
    Ox = correlate(image, Kx)
    Oy = correlate(image, Ky)
    
    Oxpix = Ox['pixels']
    Oypix = Oy['pixels']
    
    #makes an empty list so no values have to be appended, just changed
    newPix = [0 for _ in range(len(Ox['pixels'])) ]
    
    #combines the two correlated kernel images with the given formula
    for i in range(len(Oxpix)):
        newPix[i] = ((Oxpix[i]**2)+(Oypix[i]**2))**0.5
    
    finalImage = {
        'height': image['height'],
        'width': image['width'],
        'pixels': newPix
    }
    round_and_clip_image(finalImage)
    return finalImage


# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


###############################################
#LAB2 FUNCTIONS

# VARIOUS FILTERS


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    def filter_applicator(colored_image):
        #returns a tuple with the 3 colors separated
        grey_tuple = greyscale_image_from_color_image(colored_image)
        
        #applies the selected filter (filt) to the 3 separated colors
        greyR = filt(grey_tuple[0])
        greyG = filt(grey_tuple[1])
        greyB = filt(grey_tuple[2])
        
        #returns a tuple with the 3 separated colors with filters applied
        return color_from_grey((greyR, greyG, greyB))
        
        
    return filter_applicator


def make_blur_filter(n):
    '''
    Makes blurred filter on the image, returns a function
    '''
    def blur_filter(image):
        '''
        Should receive a grey image and will return it blurred

        '''
        return blurred(image, n)
    
    return blur_filter


def make_sharpen_filter(n):
    '''
    Makes a sharpened filter on the image, returns a function
    '''
    def sharpen_filter(image):
        '''
        Should receive a grey image and will return it sharpened

        '''
        return sharpened(image, n)
    
    return sharpen_filter


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    
    def cascading(image):
        '''
        Should receive a grey image, will return the image with 
        the list of filters applied to it

        '''
        imageNew = image
        
        for each_filter in filters:
            #the image will change with each filter applied
            imageNew = each_filter(imageNew)
            
        return imageNew
        
    
    return cascading
        



def greyscale_image_from_color_image(image):
    '''
    Transforms a colored image into 3 greyscale images

    Parameters
    ----------
    image : the original colored image

    Returns
    -------
    imageR : greyscale representation of the red values
    imageG : greyscale representation of the green values
    imageB : greyscale representation of the blue values

    '''
    #list of each color pixels
    pixelsR = []
    pixelsG = []
    pixelsB = []
    
    #adds the corresponding pixel color value to its corresponding color list
    for pix in range(len(image['pixels'])):
        pixelsR.append(image['pixels'][pix][0])
        pixelsG.append(image['pixels'][pix][1])
        pixelsB.append(image['pixels'][pix][2])
    
    #the image representation of each 
    imageR = {'height': image['height'], 'width': image['width'], 'pixels': pixelsR }
    imageG = {'height': image['height'], 'width': image['width'], 'pixels': pixelsG }
    imageB = {'height': image['height'], 'width': image['width'], 'pixels': pixelsB }
    
    return (imageR, imageG, imageB)
      

def color_from_grey(grey_tup):
    '''
    Takes the 3 grey images representing each color and puts them together
    into one colorful image. 

    Parameters
    ----------
    grey_tup (tuple) : a tuple with 3 images representing the red, blue and
    green colors

    Returns
    -------
    (dict) A colored image by combining the 3 given colored images

    '''
    newPix = []
    
    #puts the 3 separate images together into one 
    #(3 values per pixel together in a tuple)
    for pix in range(len(grey_tup[0]['pixels'])):
        pixR = grey_tup[0]['pixels'][pix]
        pixG = grey_tup[1]['pixels'][pix]
        pixB = grey_tup[2]['pixels'][pix]
        
        newPix.append((pixR, pixG, pixB))
    
    #returns the colored image
    return {'height': grey_tup[0]['height'], 'width': grey_tup[0]['width'], 'pixels': newPix }



# SEAM CARVING

# Main Seam Carving Implementation

def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image.
    """
    #function works recursively, repeating for each value of ncols
    if ncols == 0:
        return image
    
    #transforms the image into 3 grey tuples representing its colors
    greyImage = color_to_grey(image)
    
    #compute energy (edges)
    energyMap = compute_energy(greyImage)
    
    #compute cumulativ energy map
    cumulativeMap = cumulative_energy_map(energyMap)
    
    #find minimum energy seam
    seam = minimum_energy_seam(cumulativeMap)
    
    finalImage = image_without_seam(image, seam)
    
    return seam_carving(finalImage, ncols-1)    


# Optional Helper Functions for Seam Carving

def color_to_grey(image):
    '''
    Receives a colored image and return a grey version of it 
    following the formula v = round(.299×r+.587×g+.114×b)

    Parameters
    ----------
    image : original colored image

    Returns
    -------
    (dict): the dictionary description of a grey version of the image

    '''
    newPix = []

    #iterates over each pixel value and applies the formula v=round(.299×r+.587×g+.114×b)
    for each_pixel in range(len(image['pixels'])):
        r = image['pixels'][each_pixel][0]
        g = image['pixels'][each_pixel][1]
        b = image['pixels'][each_pixel][2]
        newPix.append(round(.299*r + .587*g + .114*b))
        
    return {'height': image['height'], 'width': image['width'], 'pixels': newPix }
        
        

def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy function),
    computes a "cumulative energy map" as described in the lab 2 writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    newPix = energy['pixels'][:]
    
    #iterates over each pixel in all rows except for the first one
    for row in range(1, energy['height']):
        for pixInRow in range(energy['width']):
            
            #find the value of the current pixel
            thisPix = energy['pixels'][(row)*energy['width'] + pixInRow]
            
            #finds the value for the middle adjecent pixel on the row above
            midAdj = newPix[(row-1)*energy['width'] + pixInRow]
            
            #if the current pixel is at the left corner of the image, 
            #dont compute the left adjecent and just copy midAdj for that value
            if (pixInRow == 0):
                leftAdj = midAdj
                #finds the value for the right adjecent pixel on the row above
                rightAdj = newPix[(row-1)*energy['width'] + pixInRow+1]
            
            #if the current pixel is at the right corner of the image, 
            #dont compute the right adjecent and just copy midAdj for that value
            elif (pixInRow == energy['width']-1):
                leftAdj = newPix[(row-1)*energy['width'] + pixInRow-1]
                rightAdj = midAdj
            
            else:
                leftAdj = newPix[(row-1)*energy['width'] + pixInRow-1]
                rightAdj = newPix[(row-1)*energy['width'] + pixInRow+1]
                
            #finds the smallest of the adjecent values
            min_adj = min([leftAdj, midAdj, rightAdj])
            
            #adds the current pixel value to that of the smallest adjecent
            newPix[(row)*energy['width'] + pixInRow] = thisPix + min_adj
    
    
    return {'height': energy['height'], 'width': energy['width'], 'pixels': newPix}


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    #will store the seam coordinates
    seamCords = []
    
    #makes a copy of the bottom row of the image pixels
    botLane = cem['pixels'][len(cem['pixels'])-cem['width']::]

    #finds the bottom row's smallest value and index
    lowestAtBot = min(botLane)
    xIndex = botLane.index(lowestAtBot)        
    
    #adds that pixel's coordinates to the seam list
    seamCords.append((cem['height']-1)*cem['width'] + xIndex)

        
    newPix = cem['pixels']
    
    #iterates from the bottom up, skipping the bottom row
    for row in range(1, cem['height'])[::-1]:

        #finds the value for the middle adjecent pixel on the row above
        midAdj = newPix[(row-1)*cem['width'] + xIndex]
        
        #if the current pixel is at the left corner of the image, 
        #dont compute the left adjecent 
        if (xIndex == 0):
            
            #finds the value for the right adjecent pixel on the row above
            rightAdj = newPix[(row-1)*cem['width'] + xIndex+1]
            
            #only has to compare the right and the middle and saves the index 
            #of the smallest
            if (rightAdj < midAdj):
                xIndex +=1
            else:
                xIndex = xIndex
        
        #if the current pixel is at the right corner of the image, 
        #dont compute the right adjecent 
        elif (xIndex == cem['width']-1):
            #finds the value for the left adjecent pixel on the row above
            leftAdj = newPix[(row-1)*cem['width'] + xIndex-1]
            
            #only has to compare the left and the middle and saves the index 
            #of the smallest
            if(midAdj < leftAdj):
                xIndex = xIndex
            else: 
                xIndex -=1
            
        else:
            leftAdj = newPix[(row-1)*cem['width'] + xIndex-1]
            rightAdj = newPix[(row-1)*cem['width'] + xIndex+1]
            
            #compares all adjecent pixels and saves the value of the smallest one's index
            if (leftAdj <= midAdj) and (leftAdj <= rightAdj):
                xIndex -=1
            elif (rightAdj < midAdj) and (rightAdj < leftAdj):
                xIndex += 1
        
        #uses the found index to save the coordinate of the adjecent pixel found
        seamCords.append((row-1)*cem['width'] + xIndex)

                
            
            
    return seamCords


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    newPix = image['pixels'][:]
    
    #iterates over the pixels list and removes all the ones in the given coordinates
    for element in seam:
        newPix.pop(element)
    
    return {'height': image['height'], 'width': image['width']-1, 'pixels': newPix}

def my_creation(image):
    '''Receives an image and switches around its color values.
    Returns a colored image
    
    '''
    colors_tup = greyscale_image_from_color_image(image)
    
    return color_from_grey((colors_tup[1], colors_tup[2], colors_tup[0]))

# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    # blur_filter = lab.make_blur_filter(ker_size)
    # assert callable(blur_filter), 'make_blur_filter should return a function.'
    # color_blur = lab.color_filter_from_greyscale_filter(blur_filter)
    # result = color_blur(im)
    
    filter1 = color_filter_from_greyscale_filter(edges)
    filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
    filt = filter_cascade([filter1, filter1, filter2, filter1])
    
    i = load_color_image('test_images/mushroom.png')
    #i = {'width': 9, 'height': 4, 'pixels': [160, 160, 0, 28, 0, 28, 0, 160, 160, 415, 218, 10, 22, 14, 22, 10, 218, 415, 473, 265, 40, 10, 28, 10, 40, 265, 473, 520, 295, 41, 32, 10, 32, 41, 295, 520]}

    save_color_image(my_creation(i), 'myCreation.png')
    #save_color_image(seam_carving(i, 100), 'minutesss.png')

    #blurry3 = color_filter_from_greyscale_filter(make_sharpen_filter(7))(i)
    #save_color_image(blurry3, 'sh.png')
