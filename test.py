import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
img = None
img_display = None
label = None
window = None




def load_image():
    global img, img_display, label
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Image File",
                                           filetypes=(("Image files", (".jpg", ".jpeg", ".png", ".bmp")), ("All files", ".")))
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (400, 500))
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_display = Image.fromarray(img_display)
        img_display = ImageTk.PhotoImage(img_display)
        label.configure(image=img_display, padx=0, pady=0)
        label.image = img_display
        show_screen2()

def apply_lpf_on_slider_change(value):
        kernel_size = int(value)
        apply_lpf(kernel_size)

def apply_lpf(kernel_size):
    global img
    lpf_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    display_image(lpf_img)


def apply_hpf_on_slider_change(value):
        kernel_size = int(value)
        apply_hpf(kernel_size)
def apply_hpf(kernel_size):
    global img
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    hpf_img = cv2.subtract(gray_image, blurred_image)
    display_image(hpf_img)

def apply_mean_on_slider_change(value):
        kernel_size = int(value)
        apply_mean_filter(kernel_size)

def apply_mean_filter(kernel_size):
    global img
    filtered_img = cv2.blur(img, (kernel_size, kernel_size))
    display_image(filtered_img)

def apply_median_on_slider_change(value):
        kernel_size = int(value)
        apply_mean_filter(kernel_size)

def apply_median_filter(kernel_size):
    global img
    filtered_img = cv2.medianBlur(img, kernel_size)
    display_image(filtered_img)

def display_image(image):
    global img_display, label
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_display = Image.fromarray(image_rgb)
    img_display = ImageTk.PhotoImage(img_display)
    label.configure(image=img_display,padx=0,pady=0)
    label.image = img_display

def apply_roberts_edge_detection():
    global img
    roberts_image = cv2.Canny(img, 100, 200)
    display_image(roberts_image)

def apply_prewitt_edge_detection():
    global img
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prewitt_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    prewitt_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    prewitt_image = np.sqrt(prewitt_x*2 + prewitt_y*2)
    prewitt_image = cv2.normalize(prewitt_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    display_image(prewitt_image)

def apply_sobel_edge_detection():
    global img
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_img = np.sqrt(sobel_x*2 + sobel_y*2).astype(np.uint8)
    display_image(sobel_img)
def apply_erosion_on_slider_change(value):
        kernel_size = int(value)
        apply_erosion(kernel_size)
def apply_erosion(kernel_size):
    global img
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_img = cv2.erode(img, kernel, iterations=1)
    display_image(eroded_img)
def apply_dilation_on_slider_change(value):
        kernel_size = int(value)
        apply_dilation(kernel_size)
def apply_dilation(kernel_size):
    global img
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_img = cv2.dilate(img, kernel, iterations=1)
    display_image(dilated_img)
def apply_opening_on_slider_change(value):
        kernel_size = int(value)
        apply_opening(kernel_size)
def apply_opening(kernel_size):
    global img
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    display_image(opened_img)
def apply_closing_on_slider_change(value):
        kernel_size = int(value)
        apply_closing(kernel_size)
def apply_closing(kernel_size):
    global img
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    display_image(closed_img)

def hough_circle():
    global img
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
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

window = tk.Tk()
window.title("Practical DIP")
window.geometry("1366x766")
window.configure(bg="beige")

title_label = tk.Label(window, text="Image Processing App", font=("Consolas", 16), bg="black", fg="white")
title_label.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

screen1 = tk.Frame(window, background="beige")
screen2 = tk.Frame(window, background="beige")
screen3 = tk.Frame(window, background="beige")
screen4 = tk.Frame(window, background="beige")
screen5 = tk.Frame(window, background="beige")

def show_screen1():
    screen1.grid()
    screen2.grid_forget()
    screen3.grid_forget()
    screen4.grid_forget()
    screen5.grid_forget()


def show_screen2():
    screen2.grid()
    screen1.grid_forget()
    screen3.grid_forget()
    screen4.grid_forget()
    screen5.grid_forget()


def show_screen3():
    screen3.grid()
    screen1.grid_forget()
    screen2.grid_forget()
    screen4.grid_forget()
    screen5.grid_forget()


def show_screen4():
    screen4.grid()
    screen1.grid_forget()
    screen2.grid_forget()
    screen3.grid_forget()
    screen5.grid_forget()

def show_screen5():
    screen5.grid()
    screen1.grid_forget()
    screen2.grid_forget()
    screen3.grid_forget()
    screen4.grid_forget()

load_button = tk.Button(screen1, text="Load Image", width=20, height=2, font=("Script MT Bold", 50), bg="DarkRed", fg="white",
                     command=load_image)
load_button.grid(row=0, column=0,  pady=200, sticky="s")

back = tk.Button(screen2, text="Go back", width=15, height=2, font=("Lucida Handwriting", 16), bg="black", fg="white", command=show_screen1)
back.grid(row=0, column=0, padx=20, pady=10, sticky="w")
back2 = tk.Button(screen3, text="Go back", width=15, height=2, font=("Lucida Handwriting", 16), bg="black", fg="white", command=show_screen2)
back2.grid(row=0, column=0, padx=20, pady=10, sticky="w")
back3 = tk.Button(screen4, text="Go back", width=15, height=2, font=("Lucida Handwriting", 16), bg="black", fg="white", command=show_screen3)
back3.grid(row=0, column=0, padx=20, pady=10, sticky="w")
back4 = tk.Button(screen5, text="Go back", width=15, height=2, font=("Lucida Handwriting", 16), bg="black", fg="white", command=show_screen2)
back4.grid(row=0, column=0, padx=20, pady=10, sticky="w")

original_button1 = Button(screen2, text="Original Image", width=15, height=2, font=("Script MT Bold", 14), bg="#CCCCFF",
                         fg="black", command=lambda: display_image(img))
original_button1.grid(row=2, column=1, padx=0, pady=10, sticky="w")

original_button2 = Button(screen3, text="Original Image", width=15, height=2, font=("Script MT Bold", 14), bg="#CCCCFF",
                         fg="black", command=lambda: display_image(img))
original_button2.grid(row=2, column=1, padx=20, pady=10, sticky="w")

original_button3 = Button(screen4, text="Original Image", width=15, height=2, font=("Script MT Bold", 14), bg="#CCCCFF",
                         fg="black", command=lambda: display_image(img))
original_button3.grid(row=2, column=1, padx=20, pady=10, sticky="w")
original_button4 = Button(screen5, text="Original Image", width=15, height=2, font=("Script MT Bold", 14), bg="#CCCCFF",
                         fg="black", command=lambda: display_image(img))
original_button4.grid(row=2, column=1, padx=20, pady=10, sticky="w")

label = Label(window, bg="beige")
label.grid(row=1, column=0, columnspan=4, sticky="nsew")



erosion_size_slider = tk.Scale(screen2, from_=1, to=20, orient=HORIZONTAL, label="Erosion",
                            background="Teal", foreground="white", length=200,command=apply_erosion_on_slider_change)
erosion_size_slider.set(1)
erosion_size_slider.grid(row=0, column=1, padx=20, pady=10, sticky="w")
erosion_size_slider.config(command=apply_erosion_on_slider_change)



dilation_size_slider =tk.Scale(screen2, from_=1, to=20, orient=HORIZONTAL, label="Dilation",
                             background="Teal", fg="white", length=200,command=apply_dilation_on_slider_change)
dilation_size_slider.set(1)
dilation_size_slider.grid(row=0, column=2, padx=20, pady=10, sticky="w")
dilation_size_slider.config(command=apply_dilation_on_slider_change)



opening_size_slider = tk.Scale(screen2, from_=1, to=20, orient=HORIZONTAL, label="Opening",
                            background="Teal", fg="white", length=200,command=apply_opening_on_slider_change)
opening_size_slider.set(1)
opening_size_slider.grid(row=0, column=3, padx=20, pady=10, sticky="w")
opening_size_slider.config(command=apply_opening_on_slider_change)


closing_size_slider = tk.Scale(screen2, from_=1, to=20, orient=HORIZONTAL, label="Closing",
                            background="Teal", fg="white", length=200,command=apply_closing_on_slider_change)
closing_size_slider.set(1)
closing_size_slider.grid(row=0, column=4, padx=20, pady=10, sticky="w")
closing_size_slider.config(command=apply_closing_on_slider_change)

hough_button = tk.Button(screen2, text="Apply Hough", width=15, height=2, font=("Tahoma", 14),
                         bg="Teal", fg="white", command=hough_circle)
hough_button.grid(row=0, column=5, padx=20, pady=30, sticky="w")

filters = tk.Button(screen2, text="Filters", width=15, height=2, font=("Lucida Handwriting", 16),
                    bg="Maroon", fg="white", command=show_screen3)
filters.grid(row=2, column=5, padx=20, pady=10, sticky="w")
seg = tk.Button(screen2, text="Segmentation", width=15, height=2, font=("Lucida Handwriting", 16),
                    bg="Maroon", fg="white", command=show_screen5)
seg.grid(row=2, column=4, padx=20, pady=10, sticky="w")


lpf_size_slider = tk.Scale(screen3, from_=1, to=20, orient=HORIZONTAL, label="Low Pass filter", background="Teal", fg="white",
                         length=200,command=apply_lpf_on_slider_change)
lpf_size_slider.set(1)
lpf_size_slider.grid(row=0, column=4, padx=20, pady=10, sticky="w")
lpf_size_slider.config(command=apply_lpf_on_slider_change)


hpf_size_slider = tk.Scale(screen3, from_=1, to=20, orient=HORIZONTAL, label="High Pass filter", background="Teal", fg="white",
                         length=200,command=apply_hpf_on_slider_change)
hpf_size_slider.set(5)
hpf_size_slider.grid(row=0, column=3, padx=20, pady=40, sticky="w")
hpf_size_slider.config(command=apply_hpf_on_slider_change)



mean_size_slider = tk.Scale(screen3, from_=1, to=20, orient=HORIZONTAL, label="Mean filter", background="Teal", fg="white",
                          length=200,command=apply_mean_on_slider_change)
mean_size_slider.set(1)
mean_size_slider.grid(row=0, column=2, padx=20, pady=10, sticky="w")
mean_size_slider.config(command=apply_mean_on_slider_change)



median_size_slider = tk.Scale(screen3, from_=3, to=21, showvalue=True, orient=HORIZONTAL, label="Median filter", background="Teal",
                           fg="white", length=200,command=apply_median_on_slider_change)
median_size_slider.set(3)
median_size_slider.grid(row=0, column=1, padx=20, pady=10, sticky="w")
median_size_slider.config(command=apply_median_on_slider_change)

#  edge detectors

edge_button = tk.Button(screen3, text="Edge Detectors", width=15, height=2, font=("Lucida Handwriting", 16), bg="Maroon", fg="white",
                         command=show_screen4)
edge_button.grid(row=0, column=5, padx=20, pady=10, sticky="w")

sobel_button = tk.Button(screen4, text="Apply Sobel", width=15, height=2, font=("Tahoma", 14), bg="Teal", fg="white",
                          command=apply_sobel_edge_detection)
sobel_button.grid(row=0, column=1, padx=20, pady=40, sticky="w")

robert_button = tk.Button(screen4, text="Apply Robert", width=15, height=2, font=("Tahoma", 14), bg="Teal", fg="white",
                           command=apply_roberts_edge_detection)
robert_button.grid(row=0, column=2, padx=20, pady=10, sticky="w")

perwit_button = tk.Button(screen4, text="Apply Perwit", width=15, height=2, font=("Tahoma", 14), bg="Teal", fg="white",
                          command=apply_prewitt_edge_detection)
perwit_button.grid(row=0, column=3, padx=20, pady=10, sticky="w")
# segmentation

merge_segmentation_slider = tk.Scale(screen5, from_=1, to=255, orient=HORIZONTAL, label="Split and Merge Segmentation",
                            background="Teal", fg="white", length=200,command=apply_merge_on_slider_change)
merge_segmentation_slider.set(1)
merge_segmentation_slider.grid(row=0, column=2, padx=20, pady=10, sticky="w")
merge_segmentation_slider.config(command=apply_merge_on_slider_change)



threshold_segmentation_slider = tk.Scale(screen5, from_=1, to=255, orient=HORIZONTAL, label="Apply segmentation",
                            background="Teal", fg="white", length=200,command=apply_seg_on_slider_change)
threshold_segmentation_slider.set(1)
threshold_segmentation_slider.grid(row=0, column=1, padx=20, pady=10, sticky="w")
threshold_segmentation_slider.config(command=apply_seg_on_slider_change)

window.grid_rowconfigure(1, weight=1)
window.grid_columnconfigure(0, weight=1)

show_screen1()

window.mainloop()