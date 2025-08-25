import streamlit as st
import pickle, joblib
import pandas as pd

model = joblib.load("laptop_price_model.pkl")
columns = pickle.load(open("train_columns.pkl", "rb"))

st.title("Laptop Price Predicting Application")

brand = st.selectbox(label="Brand", options=["Asus", "Lenovo", "Acer", "Avita", "HP", "Dell", "MSI", "Apple"], placeholder="Select a laptop brand", index=None)
processor_brand = st.selectbox(label="Brand of the Processor", options=["Intel", "AMD", "M1"], placeholder="Select a processor brand", index=None)
processor_name = st.selectbox(label="Processor Name", options=["Core i3", "Core i5", "Core i7", "Core i9", "M1", "Celeron Dual", "Ryzen 3", "Ryzen 5", "Ryzen 7", "Ryzen 9", "Pentium Q"], placeholder="Select a processor name", index=None)
processor_gnrtn = st.selectbox(label="Processor Generation", options=["4th", "7th", "8th", "9th", "10th", "11th", "12th"], placeholder="Select processor generation", index=None)

ram_gb = st.selectbox(label="RAM GB", options=["4 GB", "8 GB", "16 GB", "32 GB"], placeholder="Select RAM capacity", index=None)
ram_gb = int(ram_gb.split()[0]) if ram_gb else 0

ram_type = st.selectbox(label="RAM Type", options=["DDR3", "DDR4", "DDR5", "LPDDR3", "LPDDR4", "LPDDR4X"], placeholder="Select RAM type", index=None)

ssd = st.selectbox(label="SSD Capacity", options=["0 GB", "128 GB", "256 GB", "512 GB", "1024 GB", "2048 GB", "3072 GB"], placeholder="Select SSD capacity", index=None)
ssd = int(ssd.split()[0]) if ssd else 0

hdd = st.selectbox(label="HDD Capacity", options=["0 GB", "512 GB", "1024 GB", "2048 GB"], placeholder="Select HDD capacity", index=None)
hdd = int(hdd.split()[0]) if hdd else 0

os = st.selectbox(label="Operating System", options=["Windows", "DOS", "Mac"], placeholder="Select operating system", index=None)

os_bit = st.selectbox(label="Operating System bit", options=["32 bit", "64 bit"], placeholder="Select OS architecture", index=None)
os_bit = int(os_bit.split()[0]) if os_bit else 0

graphic_card_gb = st.selectbox(label="Graphic Card GB", options=["0 GB", "2 GB", "4 GB", "6 GB", "8 GB"], placeholder="Select graphics card memory", index=None)
graphic_card_gb = int(graphic_card_gb.split()[0]) if graphic_card_gb else 0

weight = st.selectbox(label="Weight", options=["Casual", "ThinNLight", "Gaming"], placeholder="Select weight category", index=None)
Touchscreen = st.selectbox(label="Touch Screen", options=["Yes", "No"], placeholder="Select touchscreen option", index=None)
msoffice = st.selectbox(label="MS Office", options=["Yes", "No"], placeholder="Select MS Office inclusion", index=None)

rating = st.selectbox(label="Rating for the Laptop", options=["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"], placeholder="Select rating", index=None)
rating = int(rating.split()[0]) if rating else 0

if st.button("Predict Laptop Price"):
    user_data = pd.DataFrame([{
        "brand": brand,
        "processor_brand": processor_brand,
        "processor_name": processor_name,
        "processor_gnrtn": processor_gnrtn,
        "ram_gb": ram_gb,
        "ram_type": ram_type,
        "ssd": ssd,
        "hdd": hdd,
        "os": os,
        "os_bit": os_bit,
        "graphic_card_gb": graphic_card_gb,
        "weight": weight,
        "Touchscreen": Touchscreen,
        "msoffice": msoffice,
        "rating": rating,       
    }])
    
    if None in user_data.values:
        st.error("Please select all fields before prediction")
    else:
        user_data = user_data.reindex(columns=columns, fill_value=0)
        
        price = model.predict(user_data)[0]
        st.success(f"Predicted Laptop Price: {price:.2f}")