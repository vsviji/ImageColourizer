import numpy as np
import cv2
import PySimpleGUI as sg
import os.path

prototxt = r'C:\Users\vgvij\Desktop\model/colorization_deploy_v2.prototxt'
model = r'C:\Users\vgvij\Desktop\model/colorization_release_v2.caffemodel'
points = r'C:\Users\vgvij\Desktop\model/pts_in_hull.npy'
points = os.path.join(os.path.dirname(__file__), points)
prototxt = os.path.join(os.path.dirname(__file__), prototxt)
model = os.path.join(os.path.dirname(__file__), model)
if not os.path.isfile(model):
    sg.popup_scrolled('Missing model file', 'You are missing the file "colorization_release_v2.caffemodel"',
                      'Download it',r'https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1')
    exit()
net = cv2.dnn.readNetFromCaffe(prototxt, model)   
pts = np.load(points)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
sg.theme_background_color('#008080')

def colorize_image(image_filename=None, cv2_frame=None):
    image = cv2.imread(image_filename) if image_filename else cv2_frame
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    return image, colorized

def convert_to_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    gray_3_channels = np.zeros_like(frame)  
    gray_3_channels[:, :, 0] = gray
    gray_3_channels[:, :, 1] = gray
    gray_3_channels[:, :, 2] = gray
    return gray_3_channels
left_col = [[sg.Text('Folder'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()],
            [sg.Listbox(values=[], enable_events=True, size=(40,20),key='-FILE LIST-')],
            [sg.CBox('Convert to gray first',key='-MAKEGRAY-')]]
images_col = [[sg.Text('Input file:'), sg.In(enable_events=True, key='-IN FILE-'), sg.FileBrowse()],
              [sg.Button('Colorize Photo', key='-PHOTO-'), sg.Button('Save File', key='-SAVE-'), sg.Button('Exit')],
              [sg.Image(filename='', key='-IN-'), sg.Image(filename='', key='-OUT-')],]
layout = [[sg.Column(left_col), sg.VSeperator(), sg.Column(images_col)]]
window = sg.Window('Photo Colorizer', layout, grab_anywhere=True)
prev_filename = colorized = cap = None

while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    if event == '-FOLDER-':        
        folder = values['-FOLDER-']
        img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")
        try:
            flist0 = os.listdir(folder)
        except:
            continue
        fnames = [f for f in flist0 if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith(img_types)]
        window['-FILE LIST-'].update(fnames)
    elif event == '-FILE LIST-':    
        try:
            filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            image = cv2.imread(filename)
            window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes())
            window['-OUT-'].update(data='')
            window['-IN FILE-'].update('')
            if values['-MAKEGRAY-']:
                gray_3_channels = convert_to_grayscale(image)
                window['-IN-'].update(data=cv2.imencode('.png', gray_3_channels)[1].tobytes())
                image, colorized = colorize_image(cv2_frame=gray_3_channels)
            else:
                image, colorized = colorize_image(filename)
            window['-OUT-'].update(data=cv2.imencode('.png', colorized)[1].tobytes())
        except:
            continue
    elif event == '-PHOTO-':        
        try:
            if values['-IN FILE-']:
                filename = values['-IN FILE-']
            elif values['-FILE LIST-']:
                filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            else:
                continue
            if values['-MAKEGRAY-']:
                gray_3_channels = convert_to_grayscale(cv2.imread(filename))
                window['-IN-'].update(data=cv2.imencode('.png', gray_3_channels)[1].tobytes())
                image, colorized = colorize_image(cv2_frame=gray_3_channels)
            else:
                image, colorized = colorize_image(filename)
                window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes())
            window['-OUT-'].update(data=cv2.imencode('.png', colorized)[1].tobytes())
        except:
            continue
    elif event == '-IN FILE-':    
        filename = values['-IN FILE-']
        if filename != prev_filename:
            prev_filename = filename
            try:
                image = cv2.imread(filename)
                window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes())
            except:
                continue
    
    elif event == '-SAVE-' and colorized is not None:  
        filename = sg.popup_get_file('Save colorized image.\nColorized image be saved in format matching the extension you enter.', save_as=True)
        try:
            if filename:
                cv2.imwrite(filename, colorized)
                sg.popup_quick_message('Image save complete', background_color='red', text_color='white', font='Any 16')
        except:
            sg.popup_quick_message('ERROR - Image NOT saved!', background_color='red', text_color='white', font='Any 16')
window.close()