class LinearRegression():
    def __init__(self, learning_rate = 0.000005, epoch = 1000):
        self.learning_rate = learning_rate
        self.epoch = epoch

    #training
    def fit(self, x_train, y_train, z_train):
        
        self.n = len(x_train)
        self.m1  = 1
        self.m2  = 2
        self.b   = 0

        # gradient descent learning
        for i in range(self.epoch):
            self.update_weights(x_train,y_train,z_train)
        return self


    def update_weights(self,x_train,y_train,z_train):
        z_pred = self.predict(x_train,y_train)
        dm1=0
        dm2=0
        db=0
        # calculate gradients
        for i in range (self.n):
            diff = z_pred[i] - z_train[i]
            dm1 += diff * x_train[i]
            dm2 += diff * y_train[i]
            db += diff

        dm1 = 2 * dm1 / self.n
        dm2 = 2 * dm2 / self.n
        db = 2 * db / self.n
        self.m1 -= self.learning_rate * dm1
        self.m2 -= self.learning_rate * dm2
        self.b -= self.learning_rate * db
        return self


    def predict(self, x, y):
        z = []
        for i in range(len(x)):
            z.append(round(self.m1 * x[i] + self.m2 * y[i] + self.b)) # rounding to int.
        return z
    

model = LinearRegression()

    