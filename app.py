import pygame, sys
from pygame.locals import *
import numpy as np 
import keras
from keras.models import load_model
import cv2 


win_size_x = 1000
win_size_y = 500

BOUNDARY = 4
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
green = (0, 255, 0)
blue = (0, 0, 128)

IMAGESAVE = False
MODEL = load_model("bestmodel.h5")
LABLES = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four", 
         5:"five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}

#initialize our pygame
pygame.init()

DISPLAYSURF = pygame.display.set_mode((win_size_x, win_size_y), pygame.RESIZABLE)
FONT = pygame.font.Font("freesansbold.ttf", 18)
FONT1 = pygame.font.Font("freesansbold.ttf", 24)

pygame.display.set_caption("                   Welcome to computeried recognition of handwriting")

image = pygame.image.load('th.jfif')
img_count = 1
iswriting = False
PRIDICT = True
num_x_cord = []
num_y_cord = []

text = FONT1.render('*      PROJECT EXHIBITION 2    *', True, green, blue)
text1 = FONT1.render('* Project has successfully run  *', True, green, blue)
text2 = FONT1.render('* * * * * * * * * * * * * * * * * * * * * * * *', True, green, blue)
 
# create a rectangular object for the
# text surface object
 
# set the center of the rectangular objec
while True:
    DISPLAYSURF.blit(text2, (500,10))
    DISPLAYSURF.blit(text, (500,30))
    DISPLAYSURF.blit(text1, (500,50))
    DISPLAYSURF.blit(text2, (500,75))

    DISPLAYSURF.blit(image, (0, 0))
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswriting:
            x_cord, y_cord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (x_cord, y_cord), 4, 0)
            num_x_cord.append(x_cord)
            num_y_cord.append(y_cord)
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            num_x_cord = sorted(num_x_cord)
            num_y_cord = sorted(num_y_cord)

            rect_min_x, rect_max_x = max(num_x_cord[0]-BOUNDARY, 0), min(win_size_x, num_x_cord[-1]+BOUNDARY)
            rect_min_y, rect_max_y = max(num_y_cord[0]-BOUNDARY, 0), min(num_y_cord[-1]+BOUNDARY, win_size_y)

            num_x_cord = []
            num_y_cord = []

            img_array = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                img_count +=1
            
            if PRIDICT:
                try:
                    img = cv2.resize(img_array, (28,28))
                    img = np.pad(img, (10,10), 'constant', constant_values=0)
                    img = cv2.resize(img, (28,28))/255
                except:
                    print("error denied")

                lable = str(LABLES[np.argmax(MODEL.predict(img.reshape(1,28,28,1)))])

                textSurface = FONT.render(lable, True, RED, WHITE)
                textRecobj = textSurface.get_rect()
                textRecobj.right, textRecobj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurface, textRecobj)

            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURF.fill(BLACK)

        pygame.display.update()

