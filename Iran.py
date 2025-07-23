import streamlit as st

import torch

import torchvision.transforms as transforms

from PIL import Image



@st.cache_resource

def load_model():

    model = torch.load("model.pt", map_location="cpu")

    model.eval()

    return model

model = load_model()

class_names=['206','207','405','Dena','L90','Mazda-vanet','Naisan','Pars','Paykan-Vanet','Pride',
 'Pride_vanet','Quiek','Saina','Tiba','Truck-Benz','Truck-Renault','Unknown','Volvo-FH-FM',
 'Volvo-N10','Volvo-NH','samand']



transform=transforms.Compose([transforms.Resize((224,224)),
                             transforms.ToTensor(),
                             transforms.Normalize((.5),(.5))])




st.title("🧠 Image Classifier with Unknown Detection")

st.write('Please upload your image:')



img_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])



if img_file is not None:

    image = Image.open(img_file).convert("RGB")

    st.image(image, caption="Image selected", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        confidence = conf.item()
        predicted_class = pred.item()

        if confidence < 0.7:

            st.error("Unknown Class")

        else:
            st.success(f"✅class:{class_names[predicted_class]} ({confidence * 100:.2f}%)")