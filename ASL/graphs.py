import pickle
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

with open('sequencial_model_score.h5', 'rb') as score:
    fit_history = pickle.load(score)



plt.figure(1, figsize = (20,10)) 


plt.subplot(221)  
plt.plot(fit_history[1][1])  
plt.plot(fit_history[3][1])  
plt.title('Model Accuracy')  
plt.ylabel('Accuracy')
plt.legend(['Train', 'Valid']) 
    
plt.subplot(222)  
plt.plot(fit_history[0][1])  
plt.plot(fit_history[2][1])  
plt.title('Model Loss')  
plt.ylabel('loss')  
plt.legend(['Train', 'Valid'])  
plt.show()


