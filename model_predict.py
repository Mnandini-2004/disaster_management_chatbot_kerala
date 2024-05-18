
#X=[29.8,29.8,53,53,228,651,698,430,246,293,162,40,2925]
import pickle
import sklearn
def predict_class():
    X=[29.8,29.8,53,53,228,651,698,430,246,293,162,40,2925]
    with open('mlp_model.pkl', 'rb') as file:
        model = pickle.load(file)
    if model.predict([X])[0] == 1:
        return "High Chance of Flood"
    else:
        return "Low Chance of Flood"    
