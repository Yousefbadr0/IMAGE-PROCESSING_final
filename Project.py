import cv2                                  #* DECLARATION
import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
img = None
img_display = None
label = None
window = None
bg = "#4d243d"
fg = "white"
font= ("MV Boli",20,"bold")
fonts= ("MV Boli",5,"bold")
color2 = "#ac8295"
color3 = "#4D243D"

def load_image():                           #* LOAD IMAGE
    global img, img_display, label
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Image File",
                filetypes=(("Image files", (".jpg", ".jpeg", ".png", ".bmp")), ("All files", ".")))
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (500, 500))
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_display = Image.fromarray(img_display)
        img_display = ImageTk.PhotoImage(img_display)
        label.configure(image=img_display, padx=0, pady=0)
        label.image = img_display
        # show_screen2()

def apply_lpf_on_slider_change(value):      #! LOW PASS FILTER
        kernel_size = int(value)
        apply_lpf(kernel_size)
def apply_lpf(kernel_size):
    global img
    lpf_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    display_image(lpf_img)

def apply_hpf_on_slider_change(value):      #! HIGH PASS FILTER
        kernel_size = int(value)
        apply_hpf(kernel_size)
def apply_hpf(kernel_size):
    global img
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    hpf_img = cv2.subtract(gray_image, blurred_image)
    display_image(hpf_img)

def apply_mean_on_slider_change(value):     #! MEAN FILTER
        kernel_size = int(value)
        apply_mean_filter(kernel_size)
def apply_mean_filter(kernel_size):
    global img
    filtered_img = cv2.blur(img, (kernel_size, kernel_size))
    display_image(filtered_img)

def apply_median_on_slider_change(value):   #! MEDIAN FILTER
        kernel_size = int(value)
        apply_mean_filter(kernel_size)
def apply_median_filter(kernel_size):
    global img
    filtered_img = cv2.medianBlur(img, kernel_size)
    display_image(filtered_img)

def display_image(image):                   #* DIDISPLAY IMAGE
    global img_display, label
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_display = Image.fromarray(image_rgb)
    img_display = ImageTk.PhotoImage(img_display)
    label.configure(image=img_display,padx=0,pady=0)
    label.image = img_display

def apply_roberts_edge_detection():         #? ROBERT EDGE DETECTION
    global img
    roberts_image = cv2.Canny(img, 100, 200)
    display_image(roberts_image)

def apply_prewitt_edge_detection():         #? PREWITT EDGE DETECTION
    global img
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prewitt_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    prewitt_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    prewitt_image = np.sqrt(prewitt_x*2 + prewitt_y*2)
    prewitt_image = cv2.normalize(prewitt_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    display_image(prewitt_image)

def apply_sobel_edge_detection():           #? SOBEL EDGE DETECTION
    global img
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_img = np.sqrt(sobel_x*2 + sobel_y*2).astype(np.uint8)
    display_image(sobel_img)

def apply_erosion_on_slider_change(value):  #! EROSION
    kernel_size = int(value)
    apply_erosion(kernel_size)
def apply_erosion(kernel_size):
    global img
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_img = cv2.erode(img, kernel, iterations=1)
    display_image(eroded_img)

def apply_dilation_on_slider_change(value): #! DILATION
    kernel_size = int(value)
    apply_dilation(kernel_size)
def apply_dilation(kernel_size):
    global img
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_img = cv2.dilate(img, kernel, iterations=1)
    display_image(dilated_img)

def apply_opening_on_slider_change(value):  #! OPENING
    kernel_size = int(value)
    apply_opening(kernel_size)
def apply_opening(kernel_size):
    global img
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    display_image(opened_img)

def apply_closing_on_slider_change(value):  #! CLOSING
    kernel_size = int(value)
    apply_closing(kernel_size)
def apply_closing(kernel_size):
    global img
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    display_image(closed_img)

def hough_circle():                         #! HOUGH
    global img
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1,
                            minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        hough_image = img.copy()
        for i in circles[0, :]:
            cv2.circle(hough_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(hough_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            display_image(hough_image)

def split_merge(threshold):

    if img.shape[0] < threshold or img.shape[1] < threshold:
        return img

    top_left = img[:img.shape[0]//2, :img.shape[1]//2]
    top_right = img[:img.shape[0]//2, img.shape[1]//2:]
    bottom_left = img[img.shape[0]//2:, :img.shape[1]//2]
    bottom_right = img[img.shape[0]//2:, img.shape[1]//2:]

    mean_intensity = [np.mean(top_left), np.mean(top_right), np.mean(bottom_left), np.mean(bottom_right)]

    if np.std(mean_intensity) > threshold:
        top_left = split_merge(top_left)
        top_right = split_merge(top_right)
        bottom_left = split_merge(bottom_left)
        bottom_right = split_merge(bottom_right)
        result = np.vstack((np.hstack((top_left, top_right)), np.hstack((bottom_left, bottom_right))))

    return result.astype(np.uint8)
def apply_merge_on_slider_change(value):
        threshold = int(value)
        segmented_image = split_merge(threshold)
        display_image(segmented_image)


def apply_seg_on_slider_change(value):
        threshold = int(value)
        threshold_segmentation(threshold)
def threshold_segmentation(threshold):
    global img
    _, thresholded_image = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    display_image(thresholded_image)

def make_button(master, text, command, x=0, y=0):
    button = Button(master,
                    text=text,
                    command=command,
                    bg="#4d243d",
                    relief=RAISED,
                    fg="white",
                    borderwidth=5,
                    font=font,
                    activebackground="#ac8295",
                    activeforeground="white")
    button.grid(row=x,column=y)

def make_slider(master, text, command, from_, to,x=0,y=0):
    # label = Label(master,text=text,)
    # label.pack(side=LEFT)
    slider = Scale(master,label=text,
                from_=from_,
                to=to,
                orient=HORIZONTAL,
                length=200,
                fg=color3,
                foreground="Black",
                troughcolor=color2)
    slider.set((from_+to)//2)
    slider.grid(row=x,column=y,padx=30,pady=20)
    slider.config(command=command)

functions = [apply_lpf_on_slider_change,
            apply_hpf_on_slider_change,
            apply_mean_on_slider_change,
            apply_median_on_slider_change,
            apply_erosion_on_slider_change,
            apply_dilation_on_slider_change,
            apply_opening_on_slider_change,
            apply_closing_on_slider_change,
            apply_merge_on_slider_change,
            apply_seg_on_slider_change,
            hough_circle,
            apply_roberts_edge_detection,
            apply_prewitt_edge_detection,
            apply_sobel_edge_detection]
window = Tk()
window.title("Practical DIP")
window.geometry("1366x766")
window.configure(bg="white")

label = Label(window, bd=0)
label.grid(column=0,row=0)

frame = Frame(window,background="white")
frame.grid(row=0, column=1)
make_button(window, "load_image",load_image,1,0)
make_button(window, "Original",lambda: display_image(img),1,1)
i,j = 0,0
for fun in functions:
    txt = str(fun).split("_")
    if((i+1)*2)+j-1<=10:
        make_slider(frame,txt[1],fun,0,20,i,j)
    elif (i+1)*2+j-1 == 1:
        make_button(frame,"circle",fun,i,j)
    else:
        make_button(frame,txt[1],fun,i,j)
    if j == 0:
        j = 1
    elif j == 1:
        i+=1
        j = 0
    # print(txt[1])


window.mainloop()