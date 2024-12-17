import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('xgboost_model.pkl')
X_columns = [
    'total_sqft',
    'bath',
    'BHK',
    '1st Phase JP Nagar', '2nd Stage Nagarbhavi', '5th Phase JP Nagar', '6th Phase JP Nagar', '7th Phase JP Nagar',
    '8th Phase JP Nagar', '9th Phase JP Nagar', 'Abbigere', 'Akshaya Nagar', 'Ambalipura', 'Ambedkar Nagar', 
    'Amruthahalli', 'Anandapura', 'Ananth Nagar', 'Anekal', 'Anjanapura', 'Ardendale', 'Arekere', 'Attibele',
    'BTM 2nd Stage', 'BTM Layout', 'Babusapalaya', 'Balagere', 'Banashankari', 'Banashankari Stage II', 
    'Banashankari Stage III', 'Banashankari Stage VI', 'Banaswadi', 'Bannerghatta', 'Bannerghatta Road', 
    'Basavangudi', 'Basaveshwara Nagar', 'Battarahalli', 'Begur', 'Begur Road', 'Bellandur', 'Benson Town', 
    'Bhoganhalli', 'Billekahalli', 'Binny Pete', 'Bisuvanahalli', 'Bommanahalli', 'Bommasandra', 'Bommasandra Industrial Area', 
    'Brookefield', 'Budigere', 'CV Raman Nagar', 'Chamrajpet', 'Chandapura', 'Channasandra', 'Chikka Tirupathi', 
    'Chikkalasandra', 'Choodasandra', 'Cooke Town', 'Dasanapura', 'Dasarahalli', 'Devanahalli', 'Devarachikkanahalli', 
    'Dodda Nekkundi', 'Doddathoguru', 'Domlur', 'EPIP Zone', 'Electronic City', 'Electronic City Phase II', 
    'Electronics City Phase 1', 'Frazer Town', 'Garudachar Palya', 'Gottigere', 'Green Glen Layout', 'Gubbalala', 
    'Gunjur', 'HBR Layout', 'HRBR Layout', 'HSR Layout', 'Haralur Road', 'Harlur', 'Hebbal', 'Hebbal Kempapura', 
    'Hegde Nagar', 'Hennur', 'Hennur Road', 'Hoodi', 'Horamavu Agara', 'Horamavu Banaswadi', 'Hormavu', 'Hosa Road', 
    'Hosakerehalli', 'Hoskote', 'Hosur Road', 'Hulimavu', 'Iblur Village', 'Indira Nagar', 'JP Nagar', 'Jakkur', 'Jalahalli', 
    'Jigani', 'Judicial Layout', 'KR Puram', 'Kadugodi', 'Kaggadasapura', 'Kaggalipura', 'Kaikondrahalli', 
    'Kalena Agrahara', 'Kalyan nagar', 'Kambipura', 'Kammanahalli', 'Kammasandra', 'Kanakapura', 'Kanakpura Road', 
    'Kannamangala', 'Kasavanhalli', 'Kasturi Nagar', 'Kathriguppe', 'Kaval Byrasandra', 'Kenchenahalli', 'Kengeri', 
    'Kengeri Satellite Town', 'Kereguddadahalli', 'Kodichikkanahalli', 'Kodihalli', 'Kogilu', 'Koramangala', 'Kothannur', 
    'Kothanur', 'Kudlu', 'Kudlu Gate', 'Kumaraswami Layout', 'Kundalahalli', 'Lakshminarayana Pura', 
    'Lingadheeranahalli', 'Magadi Road', 'Mahadevpura', 'Mallasandra', 'Malleshpalya', 'Malleshwaram', 'Marathahalli', 
    'Margondanahalli', 'Munnekollal', 'Mysore Road', 'Nagarbhavi', 'Nagavara', 'Nagavarapalya', 'OMBR Layout', 
    'Old Airport Road', 'Old Madras Road', 'Padmanabhanagar', 'Pai Layout', 'Panathur', 'Parappana Agrahara', 
    'Poorna Pragna Layout', 'R.T. Nagar', 'Rachenahalli', 'Raja Rajeshwari Nagar', 'Rajaji Nagar', 'Ramagondanahalli', 
    'Ramamurthy Nagar', 'Rayasandra', 'Sahakara Nagar', 'Sanjay nagar', 'Sarjapur', 'Sarjapur Road', 
    'Sarjapura - Attibele Road', 'Sector 2 HSR Layout', 'Seegehalli', 'Singasandra', 'Somasundara Palya', 
    'Sonnenahalli', 'Subramanyapura', 'TC Palaya', 'Talaghattapura', 'Thanisandra', 'Thigalarapalya', 'Thubarahalli', 
    'Tumkur Road', 'Ulsoor', 'Uttarahalli', 'Varthur', 'Varthur Road', 'Vidyaranyapura', 'Vijayanagar', 'Vittasandra', 
    'Whitefield', 'Yelachenahalli', 'Yelahanka', 'Yelahanka New Town', 'Yeshwanthpur'
]

X_columns = pd.Series(X_columns)

st.title('Bengaluru House Price Prediction')

location = st.selectbox('Select Location', X_columns[3:])
sqft = st.number_input('Enter Total Square Feet', min_value=1)
bath = st.number_input('Enter Number of Bathrooms', min_value=1)
bhk = st.number_input('Enter Number of BHK', min_value=1)

def predict_price(location, sqft, bath, bhk, model, X_columns):
    x = np.zeros(len(X_columns))

    if location in X_columns.values:  
        loc_index = X_columns[X_columns == location].index[0]
        x[loc_index] = 1

    x[X_columns[X_columns == 'total_sqft'].index[0]] = sqft
    x[X_columns[X_columns == 'bath'].index[0]] = bath
    x[X_columns[X_columns == 'BHK'].index[0]] = bhk

    price = model.predict([x])[0]
    price = price * 100000
    return price


if st.button('Predict'):
    predicted_price = predict_price(location, sqft, bath, bhk, model, X_columns)
    st.write(f"Predicted Price: ₹{predicted_price:,.2f}")
