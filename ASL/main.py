import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ctypes
import sys
import cv2
from tkinter import *
from tkinter.tix import *
from time import sleep
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
from detect import detect_frame  # Ensure `detect_frame` is correctly implemented

# Load the trained model
model = load_model('sequencial_model.h5')
print(model.summary())
print("Model Loaded...")
classes = os.listdir('dataset/train')  # Assuming dataset has 'train' folder

# Define global variables
current_character = ""
cap = None

def HomePage():
    global cntct, about, predict_stream, cap
    try:
        cntct.destroy()
    except:
        pass
    try:
        about.destroy()
    except:
        pass
    try:
        predict_stream.destroy()
    except:
        pass
    try:
        cap.release()
    except:
        pass
    
    # Create the main window
    window = Tk()
    img = Image.open("Images\\HomePage.png")
    img = ImageTk.PhotoImage(img)
    panel = Label(window, image=img)
    panel.pack(side="top", fill="both", expand="no")

    # Get screen dimensions for centring
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    lt = [w, h]
    a = str(lt[0]//2-446)
    b = str(lt[1]//2-383)

    window.title("HOME - American Sign Language")
    window.geometry("1214x680+" + a + "+" + b)
    window.resizable(0, 0)

    # Define buttons for navigation
    def contactus():
        ContactUsPage(window)

    def aboutus():
        AboutPage(window)

    def exit_app():
        result = tkMessageBox.askquestion("American Sign Language", "Are you sure you want to exit?", icon="warning")
        if result == 'yes':
            sys.exit()

    def start_video_mode():
        VideoModePage(window)

    # Add buttons to the homepage
    aboutusbtn = Button(window, text="About Project", font=("Arial Narrow", 16, "bold"), width=12,
                        relief=FLAT, bd=0, borderwidth='0', bg="#787A91", fg="#0F044C",
                        activebackground="#EEEEEE", activeforeground="#141E61", command=aboutus)
    aboutusbtn.place(x=784, y=40)

    contactusbtn = Button(window, text="About Us", font=("Arial Narrow", 16, "bold"), width=10,
                          relief=FLAT, bd=0, borderwidth='0', bg="#787A91", fg="#0F044C",
                          activebackground="#EEEEEE", activeforeground="#141E61", command=contactus)
    contactusbtn.place(x=925, y=40)

    exitbtn = Button(window, text="Exit", font=("Arial Narrow", 16, "bold"), width=10,
                     relief=FLAT, bd=0, borderwidth='0', bg="#787A91", fg="#0F044C",
                     activebackground="#EEEEEE", activeforeground="#141E61", command=exit_app)
    exitbtn.place(x=1050, y=40)

    livestream = Button(window, text="LIVE STREAM", font=("Arial Narrow", 18, "bold"), width=20,
                        relief=FLAT, bd=1, borderwidth='1', bg="#787A91", fg="#0F044C",
                        activebackground="#EEEEEE", activeforeground="#141E61", command=start_video_mode)
    livestream.place(x=225, y=350)

    window.mainloop()


def AboutPage(prev_window=None):
    if prev_window:
        prev_window.destroy()

    about = Tk()
    img = Image.open("Images\\AboutUs.png")
    img = ImageTk.PhotoImage(img)
    panel = Label(about, image=img)
    panel.pack(side="top", fill="both", expand="yes")

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    lt = [w, h]
    a = str(lt[0]//2-446)
    b = str(lt[1]//2-383)

    about.title("About Project - American Sign Language")
    about.geometry("1214x680+" + a + "+" + b)
    about.resizable(0, 0)

    homebtn = Button(about, text="HomePage", font=("Arial Narrow", 16, "bold"), width=10,
                     relief=FLAT, bd=0, borderwidth='0', bg="#787A91", fg="#0F044C",
                     activebackground="#EEEEEE", activeforeground="#141E61", command=HomePage)
    homebtn.place(x=800, y=40)

    exitbtn = Button(about, text="Exit", font=("Arial Narrow", 16, "bold"), width=10,
                     relief=FLAT, bd=0, borderwidth='0', bg="#787A91", fg="#0F044C",
                     activebackground="#EEEEEE", activeforeground="#141E61", command=sys.exit)
    exitbtn.place(x=1050, y=40)

    about.mainloop()


def ContactUsPage(prev_window=None):
    if prev_window:
        prev_window.destroy()

    cntct = Tk()
    img = Image.open("Images\\AboutTeam.png")
    img = ImageTk.PhotoImage(img)
    panel = Label(cntct, image=img)
    panel.pack(side="top", fill="both", expand="yes")

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    lt = [w, h]
    a = str(lt[0]//2-446)
    b = str(lt[1]//2-383)

    cntct.title("About Team - American Sign Language")
    cntct.geometry("1214x680+" + a + "+" + b)
    cntct.resizable(0, 0)

    homebtn = Button(cntct, text="HomePage", font=("Arial Narrow", 16, "bold"), width=10,
                     relief=FLAT, bd=0, borderwidth='0', bg="#787A91", fg="#0F044C",
                     activebackground="#EEEEEE", activeforeground="#141E61", command=HomePage)
    homebtn.place(x=800, y=40)

    exitbtn = Button(cntct, text="Exit", font=("Arial Narrow", 16, "bold"), width=10,
                     relief=FLAT, bd=0, borderwidth='0', bg="#787A91", fg="#0F044C",
                     activebackground="#EEEEEE", activeforeground="#141E61", command=sys.exit)
    exitbtn.place(x=1050, y=40)

    cntct.mainloop()


def VideoModePage(prev_window=None):
    global cap
    if prev_window:
        prev_window.destroy()

    predict_stream = Tk()
    img = Image.open("Images\\Loading.png")
    img = ImageTk.PhotoImage(img)
    panel = Label(predict_stream, image=img)
    panel.pack(side="top", fill="both", expand="yes")

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    lt = [w, h]
    a = str(lt[0] // 2 - 446)
    b = str(lt[1] // 2 - 383)

    predict_stream.title("Predict - American Sign Language")
    predict_stream.geometry("1214x680+" + a + "+" + b)
    predict_stream.resizable(0, 0)

    cap = cv2.VideoCapture(0)

    def show_frame():
        _, frame = cap.read()
        global current_character
        frame, current_character = detect_frame(frame, model, classes)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        display1.imgtk = imgtk
        display1.configure(image=imgtk)
        predict_stream.after(10, show_frame)

    imageFrame = Frame(predict_stream, width=600, height=500)
    imageFrame.place(x=300, y=100)
    display1 = Label(imageFrame)
    display1.grid(row=1, column=0)

    text_box = Text(predict_stream, height=3, width=25, font=("Arial Narrow", 16))
    text_box.place(x=300, y=550)

    def add_into_string():
        global current_character
        if current_character == "Space":
            current_character = " "
        text_box.insert(END, current_character)
        text_box.see(END)

    def remove_from_string():
        content = text_box.get("1.0", END)[:-2]
        text_box.delete("1.0", END)
        text_box.insert("1.0", content)

    Button(predict_stream, text="ADD", font=("Arial Narrow", 16, "bold"), width=12,
           relief=FLAT, bd=0, borderwidth='0', bg="#787A91", fg="#0F044C",
           activebackground="#EEEEEE", activeforeground="#141E61", command=add_into_string).place(x=600, y=600)

    Button(predict_stream, text="REMOVE", font=("Arial Narrow", 16, "bold"), width=10,
           relief=FLAT, bd=0, borderwidth='0', bg="#787A91", fg="#0F044C",
           activebackground="#EEEEEE", activeforeground="#141E61", command=remove_from_string).place(x=850, y=600)

    # Menu bar buttons
    aboutusbtn = Button(predict_stream, text="About Project", font=("Agency FB", 16, "bold"), width=12,
                        relief=FLAT, bd=0, borderwidth='0', bg="#787A91", fg="#0F044C",
                        activebackground="#EEEEEE", activeforeground="#141E61",
                        command=lambda: AboutPage(predict_stream))
    aboutusbtn.place(x=784, y=40)

    contactusbtn = Button(predict_stream, text="About Us", font=("Agency FB", 16, "bold"), width=10,
                          relief=FLAT, bd=0, borderwidth='0', bg="#787A91", fg="#0F044C",
                          activebackground="#EEEEEE", activeforeground="#141E61",
                          command=lambda: ContactUsPage(predict_stream))
    contactusbtn.place(x=925, y=40)

    exitbtn = Button(predict_stream, text="Exit", font=("Agency FB", 16, "bold"), width=10,
                     relief=FLAT, bd=0, borderwidth='0', bg="#787A91", fg="#0F044C",
                     activebackground="#EEEEEE", activeforeground="#141E61", command=sys.exit)
    exitbtn.place(x=1050, y=40)

    show_frame()
    predict_stream.mainloop()


def LoadingScreen():
    root = Tk()
    root.config(bg="white")
    root.title("Loading - American Sign Language")

    img = Image.open("Images\\Loading.png")
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.pack(side="top", fill="both", expand="yes")

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    lt = [w, h]
    a = str(lt[0]//2-446)
    b = str(lt[1]//2-383)

    root.geometry("1214x680+" + a + "+" + b)
    root.resizable(0, 0)

    def play_animation():
        for _ in range(40):
            sleep(0.07)
            root.update_idletasks()
        root.destroy()
        HomePage()

    play_animation()
    root.mainloop()


# Start the application
LoadingScreen()
