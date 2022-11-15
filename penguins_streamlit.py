import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import requests
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
import numpy as np
import cv2
import rembg
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
import joblib


@st.cache(persist=True)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    
    return r.json()
# # st.markdown(""" <style> .font { font-size:35px ; font-family: 'Cooper Black'; color: blue;}  </style> """, unsafe_allow_html=True)
# # st.markdown('<h1 class="font">Quantitative Analysis of Dopamine concentration...</h1>', unsafe_allow_html=True)
# # original_title = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Original image</p>'
st.markdown(f'<h1 style="color:blue;font-size:30px;">{"Quantitative Analysis of Dopamine concentration..."}</h1>',unsafe_allow_html=True)
# #st.title('WebApp to predict Dopamine concentration')
st.markdown(f'<h4 style="color:black;font-size:15px;">{"Use this app to determine change in dopamine concentration after adding Gold nanoparticles!"}</h4>',unsafe_allow_html=True)
# #st.markdown('Use this app to see accuracy score on dopamine concentration!')

#st.progress(10)
with st.spinner('Wait for it...'):
    time.sleep(10)
st.balloons()  
lottie_penguin =load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_cwgnwfat.json')
st_lottie(lottie_penguin, height=200)

@st.cache(persist=True)
def start():
    pc_pickle = 'pc_pickle.sav'
    fbunet=joblib.load(pc_pickle)
    #pc_pickle.close()

    output_pickle = 'output_pc.sav'
    #pickle.dump(target1, output_pickle)
    target1 = joblib.load(output_pickle)
   # output_pickle.close()

    pc_pi= 'pcanew.sav'
    pca=joblib.load(pc_pi)
   # pc_pi.close()
    return fbunet,target1,pca

fbunet,target1,pca= start()

features_unet=pd.DataFrame(fbunet)
train_features, test_features, \
    train_labels, test_labels = train_test_split(features_unet, target1, test_size=0.2, random_state=42)

def prediction():
    rf = 'rf.sav'
    model=joblib.load(rf)
    xg = 'xg.sav'
    model1=joblib.load(xg)
    re = 'sv.sav'
    regressor1=joblib.load(re)
    lm= 'lm.sav'
    lm2=joblib.load(lm)
    return model,model1,regressor1,lm2
model,model1,regressor1,lm2 = prediction()
def predict_image(df1):
    #model = RandomForestRegressor(n_estimators=300, random_state=0)
#     model=LinearRegression()
   # model = XGBRegressor()
  #  model.fit(train_features, train_labels.values.ravel())
   # modelt= RF()
    preds = model.predict(df1)
    x=round(preds.item(),3)
    return x
def get_RGB1u(img1):
   # img = Image.fromarray(img1)
    img2 = img1.resize((50, 50))
    pixel_values = list(img2.getdata())
    RGB_pixels = [pixel for image_pixels in pixel_values for pixel in image_pixels]   
    return RGB_pixels
def pca_decompositionu(img_rm):
    rgb = get_RGB1u(img_rm)
    return rgb
def read_images_testunet(path):   
    test_aya = []
    img = pca_decompositionu(path)
    test_aya.append(img)
    return test_aya
st.markdown(""" <style> .font { font-size:20px ; font-family: 'Cooper Black'; color: blue;}  </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Please Upload your photo here...</p>', unsafe_allow_html=True)
image_file = st.file_uploader("Dopamine image", type=["png","jpg","jpeg"])


if image_file is not None:
 #   imgn = Image.open(image_file)
    img = Image.open(image_file)
    remove = rembg.remove(img)
    df300u=read_images_testunet(remove)
    df_pc300u=pca.transform(np.array(df300u))

    dft300u=pd.DataFrame(df_pc300u)
 
   # st.markdown(f'<h4 style="color:black;font-size:15px;">{"Concentration :"}</h4>',unsafe_allow_html=True)
   # st.markdown('###### **Concentration :**')
    st.write(f"###### **Concentration is**  :  {predict_image(dft300u)}")
    

#     st.image(img)
#     st.image(remove)
else:
    
#     img2 = Image.open("image75")
#     remove=rembg.remove(img)
#     df_pc300u=pca1.transform(np.array(df300u))


#     dft300u=pd.DataFrame(df_pc300u)
    st.write("Concentration is: ")
#     st.image(img2)
#     st.image(remove2)
    

st.sidebar.subheader("Choose Regression Model")
predict = st.sidebar.selectbox("Select your model then press predict", ( "Random Forest","XGBRegressor","Support Vector Regressor (SVR)", "Linear Regression"))

if predict == "Random Forest":
   
    if st.sidebar.button("Predict", key = "predict"):
        st.subheader("Random Forest Results")

      #  modelrf=RF()
        preds = model.predict(test_features)
        # Evaluate the model
        mea = mean_absolute_error(test_labels, preds)
        score = model.score(test_features, test_labels.values.ravel()) * 100
        
        st.write("Accuracy:", score.round(3))
        st.write('MAE:', mea.round(3))
       # st.write('MSE : ',MSE(test_labels, preds).round(3))
        st.write('RMSE : ',np.sqrt(MSE(test_labels, preds)).round(3))
        fig, x_ax = plt.subplots(figsize=(8, 4))
        x_ax1 = range(len(test_labels))
        x_ax.scatter(x_ax1, test_labels, s=5, color="blue", label="original")
        x_ax.plot(x_ax1, preds, lw=0.8, color="purple", label="predicted")
        legend_labels=["original","predicted"]
        x_ax.legend(legend_labels,loc=2)
        x_ax.set_title("Scatterplot of RandomForest")
        st.pyplot(fig)


elif predict == "XGBRegressor":
   
    if st.sidebar.button("Predict", key="predict"):
        st.subheader("XGBRegressor Results")

        preds1 = model1.predict(test_features)
        # Evaluate the model
        mea1 = mean_absolute_error(test_labels, preds1)
        score1 = model1.score(test_features, test_labels.values.ravel()) * 100
        
        st.write("Accuracy:", score1)
        st.write('MAE:', mea1)
      #  st.write('MSE : ',MSE(test_labels, preds1))
        st.write('RMSE : ',np.sqrt(MSE(test_labels, preds1)))
        fig1, x_axx = plt.subplots(figsize=(8, 4))
        x_ax1 = range(len(test_labels))
        x_axx.scatter(x_ax1, test_labels, s=5, color="blue", label="original")
        x_axx.plot(x_ax1, preds1, lw=0.8, color="purple", label="predicted")
        legend_labels=["original","predicted"]
        x_axx.legend(legend_labels,loc=2)
        x_axx.set_title("Scatterplot of XGBRegressor ")
        st.pyplot(fig1)

       
elif predict == "Support Vector Regressor (SVR)":
   
    if st.sidebar.button("Predict", key="predict"):
        st.subheader("Support Vector Regressor Results")
  

        score_sv1=regressor1.score(test_features,test_labels.values.ravel() )*100

        v11=regressor1.predict(test_features)

        rmse = np.sqrt(MSE(test_labels, v11))

        mea_x = mean_absolute_error(test_labels, v11)
       
        st.write("Accuracy:", score_sv1.round(3))
        st.write('MAE:', mea_x.round(3))
      #  st.write('MSE : ',MSE(test_labels, v11).round(3))
        st.write('RMSE : ',rmse.round(3))
        fig4, x_ax4 = plt.subplots(figsize=(8, 4))
        x_ax11 = range(len(test_labels))
        x_ax4.scatter(x_ax11, test_labels, s=5, color="blue", label="original")
        x_ax4.plot(x_ax11, v11, lw=0.8, color="purple", label="predicted")
        legend_labels=["original","predicted"]
        x_ax4.legend(legend_labels,loc=2)
        x_ax4.set_title("Scatterplot of SVR")
        st.pyplot(fig4)

       
elif predict == "Linear Regression":
   
    if st.sidebar.button("Predict", key="predict"):
        st.subheader("Linear Regression")


        preds_lin = lm2.predict(test_features)

        meal = mean_absolute_error(test_labels, preds_lin)
        scorell = lm2.score(test_features, test_labels) * 100
        st.write("Accuracy:", scorell.round(3))
        st.write('MAE:', meal.round(3))
       # st.write('MSE : ',MSE(test_labels, preds_lin).round(3))
        st.write('RMSE : ',np.sqrt(MSE(test_labels, preds_lin).round(3)))
        regress=LinearRegression()
        regress.fit(test_labels, preds_lin)  
        y_fit = regress.predict(preds_lin) 
        reg_intercept = round(regress.intercept_[0],4)
        reg_coef = round(regress.coef_.flatten()[0],4)
        reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)
        fig9, x1 = plt.subplots(figsize=(8, 4))
        x1.scatter( test_labels, preds_lin, color="blue", label="data")
        
      #  legend_labels=["original","predicted",reg_label]
        x1.plot(preds_lin, y_fit, color='red', linewidth=2, label = 'Linear regression\n'+reg_label) 
        
        x1.legend()
        x1.set_xlabel('observed')
        x1.set_ylabel('predicted')
       # x1.legend(legend_labels,loc=2)
        x1.set_title('Linear Regression')
        st.pyplot(fig9)
       

