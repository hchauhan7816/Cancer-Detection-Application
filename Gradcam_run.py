import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import densenet121
from torchvision.models import alexnet
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import argparse
import Gradcam_utils
import os
import shutil
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog


def fetch_and_fit_model():
    model_ft = densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, 2)
    state = torch.load(
        'Models/Dense_net/densenetFeatureExtraction_l_0.001_m_0.9.pt')
    model_ft.load_state_dict(state['model_state_dict'])
    return model_ft


def alex_fetch_and_fit_model():
    model_ft = alexnet(pretrained=True)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, 2)
    state = torch.load(
        'Models/Alex_net/alexnetFeatureExtraction.pt')
    model_ft.load_state_dict(state['model_state_dict'])
    return model_ft


def upload_file():
    global img, img1, img2
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    shutil.copy(filename, "images/hello")

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_directory', action='store', default=os.getcwd(),
                        required=False, help='The root directory containing the image. Use "./" for current directory')
    args = parser.parse_args()
    input_directory = args.input_directory
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    temp = os.getcwd()
    temp = os.path.join(temp, "images")

    dataset = datasets.ImageFolder(root=temp, transform=transform)
    dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)
    images = []

    for filename in os.listdir(os.path.join(input_directory, 'images/hello/')):  # doubt
        if filename.endswith('.jpg'):
            images.append(filename)

    model_gradcam = fetch_and_fit_model()
    ds, img, scores, label = Gradcam_utils.run_inference(
        model_gradcam, dataloader)
    heatmap = Gradcam_utils.get_grad_cam(ds, img, scores, label)

    model_gradcam_2 = alex_fetch_and_fit_model()
    ds_2, img2, scores_2, label_2 = Gradcam_utils.run_inference(
        model_gradcam_2, dataloader, "AlexNet")
    heatmap_2 = Gradcam_utils.get_grad_cam(ds_2, img2, scores_2, label_2, True)

    for image in images:
        Gradcam_utils.render_superimposition(input_directory, heatmap, image)
        Gradcam_utils.render_superimposition(
            input_directory, heatmap_2, image, True)

        if(label == 1):
            text1 = "Densenet Image prediction :: malignant"
        else:
            text1 = "Densenet Image prediction :: benign"

        if(label_2 == 1):
            text2 = "Alexnet Image prediction ::  malignant"
        else:
            text2 = "Alexnet Image prediction :: benign"

        # Create an object of tkinter ImageTk
        img = ImageTk.PhotoImage(Image.open(
            input_directory + '/images/hello/' + image))
        img1 = ImageTk.PhotoImage(Image.open(
            input_directory + '/images/alex_superimposed_' + image))
        img2 = ImageTk.PhotoImage(Image.open(
            input_directory + '/images/superimposed_' + image))

        # Create a Label Widget to display the text or Image

        Label(frame, image=img).pack(side=LEFT, ipadx=10, ipady=10)
        Label(frame, image=img1).pack(side=LEFT, ipadx=10, ipady=10)
        Label(frame, image=img2).pack(side=LEFT, ipadx=10, ipady=10)
        #Label(frame,text="original",font="3").pack(side=LEFT,ipadx= 10,ipady=10)

        Label(win, font="3").pack(side=BOTTOM)
        Label(win, text=text1, font="3").pack(side=BOTTOM)
        Label(win, text=text2, font="3").pack(side=BOTTOM)

        directory = image
        parent = "images/hello/"
        path = os.path.join(parent, directory)
        os.remove(path)


if __name__ == "__main__":
    win = Tk()

    win.geometry("1900x650")

    w = Label(
        win, text='Explainable Artificial Intelligence(XAI) Model for Lung Cancer Detection', font="50")
    w.pack(ipadx=10, pady=10)

    frame = Frame(win)
    frame.pack()

    frame1 = Frame(win)
    frame1.pack()

    bottomframe = Frame(win)
    bottomframe.pack(side=BOTTOM)

    frame.place(anchor='center', relx=0.5, rely=0.5)
    win.title("Explainable AI Project on Cancer Detection")
    b1 = Button(win, text='Upload File',
                width=20, fg="red", command=upload_file)
    b1.pack(ipadx=20, pady=15)
    win.mainloop()
