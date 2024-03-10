import numpy as np
from tqdm import tqdm

#SVM objective=0.5*(np.linalg.norm(w)**2)+c*np.sum(np.max(0,1-y*(np.doy(X,w)+b)))
#margin=y*(np.dot(X,w)+b)
#lossFn=max(0,1-margin)
#update w,b (SGD)

class SVM_class:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        #implement this
        # initialize the parameters        
        self.b=0 #bias b in y(wx+b)
        self.w=np.zeros(X.shape[1]) #beta vector=w
        #pass

    def fit(
            self, X, y, 
            learning_rate:float,
            num_iters:int,
            C:float,            
    ) -> None:
        self._initialize(X)        
        #implement this
        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            id=np.random.randint(0,len(X))
            #margin=y*(np.dot(X,w)+b)
            xi=X[id]  
            yi=y[id]
            margin=yi*(np.dot(xi,self.w)+self.b)
            #update w,b (SGD) [margin>=1 loss=0 or 1-margin]            
            loss=np.maximum(0,1-margin)
            if loss>0:
                self.w-=learning_rate* (C*self.w-yi*xi)
                self.b-=learning_rate*yi*C                            
            else:
                self.w-=learning_rate*C* self.w
            #raise NotImplementedError"""
            """if margin>=1:
                loss=0
            else:
                loss= -yi*xi
            self.w-=learning_rate*(self.w)"""
            
    
    def predict(self, X) -> np.ndarray:
        # make predictions for the given data        
        #pred=np.dot(X,self.w)-self.b
        return  np.sign(np.dot(X,self.w)+self.b)
        #raise NotImplementedError
    def accuracy_score(self, X, y) -> float:
        #implement this
        #(((correct prediction)/Nsamples)x100) mean
        predictions=self.predict(X)
        accuracy=((np.mean(predictions==y)))        
        return accuracy
        #raise NotImplementedError
    
    def precision_score(self, X, y) -> float:
        #implement this
        #((TP)/(TP+FP))
        predictions=self.predict(X)
        TP=np.sum(np.logical_and(predictions==1,y==1))
        FP=np.sum(np.logical_and(predictions==1,y==-1))
        print(f"tp={TP} , FP={FP}")
        return ((TP/(TP+FP)) if TP+FP >0 else 0)
        #raise NotImplementedError
    
    def recall_score(self, X, y) -> float:
        #implement this
        #((TP)/(TP+FN))
        predictions=self.predict(X)
        TP=np.sum(np.logical_and(predictions==1,y==1))
        FN=np.sum(np.logical_and(predictions==-1,y==1))
        return((TP/(TP+FN) if TP+FN >0 else 0))
        #raise NotImplementedError
    
    def f1_score(self, X, y) -> float:
        #implement this
        #2((precision*recall)/(precision+recall))
        precision=self.precision_score(X,y)
        recall=self.recall_score(X,y)
        if (precision+recall)>0:
            f1_score=2*(precision*recall)/(precision + recall)            
        else:
            f1_score=0        
        return f1_score
        #raise NotImplementedError



    
