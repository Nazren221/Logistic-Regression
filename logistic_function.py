import numpy as np
 

def sigmoid(omega, b, x,):
    z = np.dot(omega.T,x)+b
    y_pred = 1/(1+np.exp(-z))
    return y_pred 
def logistic_train(x,y,omega,b,iterations,alpha):
    m=len(y)
    def cost(omega, x, y):
        h = sigmoid(omega, b, x)
        J = -(np.dot(np.log(h), y)+np.dot(np.log(1-h), 1-y))
        J /= m
        return h, J

    def gradient(omega, x, y):
        
        h, J = cost(omega,x,y)
        domega = np.dot(x,h.T - y)
        domega /= m
        db = np.sum(h.T - y)
        db /= m
        return domega, db

    h, J = cost(omega, x, y)
    
    for i in range(iterations):
        h,J = cost(omega,x,y)
        domega, db = gradient(omega, x, y)
        omega = omega - alpha*domega 
        b = b - alpha*db
    
    return J,  omega, b

# you have to call training function firstly,  in order to get more accurate parameters
def logistic_testing(x, y, b,omega,length):

    def prediction(omega, b, x):
        y_pred = sigmoid(omega, b, x)
        for i in y_pred:
            for j in range(len(i)):
                if y_pred[0][j]>0.5:
                    y_pred[0][j]=1
                else:
                    y_pred[0][j]=0
        return y_pred
    hypothesis=prediction(omega, b,x)
    def number_of_correct_result(y,h):
        number=0
        for i in range(length):
            if h[0][i]==y[0][i]:
                number+=1
            else:
                pass
        return number
        
    def get_accuracy(number_of_correct_result, total_len):
        return (number_of_correct_result/total_len*100)
    number=number_of_correct_result(y,hypothesis)
    rate=get_accuracy(number , length)
    return rate
