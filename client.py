import base64
import requests
from tkinter.filedialog import askopenfilenames
url = 'https://7155bee98293b932fe.gradio.live/loraapi/v1/training'

def send_train_request(image_paths):
    images = []
    for image in image_paths:
        with open(image, "rb") as image_file:
            images.append(base64.b64encode(image_file.read()).decode('utf-8'))

    payload ={"train_images": images}
    resp = requests.post(url=url, json=payload) 

    print(resp.json()) 

def _main():
    images = askopenfilenames()

    send_train_request(images)
    return





if __name__ == '__main__':
    _main()