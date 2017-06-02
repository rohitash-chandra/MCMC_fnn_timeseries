
#!/usr/bin/python


# MCMC Random Walk for Feedforward Neural Network for One-Step-Ahead Chaotic Time Series Prediction

#Data (Sunspot and Lazer). Taken' Theorem used for Data Reconstruction (Dimension = 4, Timelag = 2).
#Data procesing file is included.

# RMSE (Root Mean Squared Error)

#based on: https://github.com/rohitash-chandra/FNN_TimeSeries
#based on: https://github.com/rohitash-chandra/mcmc-randomwalk


# Rohitash Chandra, Centre for Translational Data Science
# University of Sydey, Sydney NSW, Australia.  2017 c.rohitash@gmail.conm
# https://www.researchgate.net/profile/Rohitash_Chandra






import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math

#An example of a class
class Network:

    def __init__(self, Topo, Train, Test ):
        self.Top  = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
    	np.random.seed()

     	self.W1 = np.random.randn(self.Top[0]  , self.Top[1])  / np.sqrt(self.Top[0] )
        self.B1 = np.random.randn(1  , self.Top[1])  / np.sqrt(self.Top[1] ) # bias first layer
    	self.W2 = np.random.randn(self.Top[1] , self.Top[2]) / np.sqrt(self.Top[1] )
        self.B2 = np.random.randn(1  , self.Top[2])  / np.sqrt(self.Top[1] ) # bias second layer

        self.hidout = np.zeros((1, self.Top[1] )) # output of first hidden layer
        self.out = np.zeros((1, self.Top[2])) #  output last layer


    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self,actualout):
        error = np.subtract(self.out, actualout)
        sqerror= np.sum(np.square(error))/self.Top[2]
        return sqerror

    def ForwardPass(self, X ):
         z1 = X.dot(self.W1) - self.B1
         self.hidout = self.sigmoid(z1) # output of first hidden layer
         z2 = self.hidout.dot(self.W2)  - self.B2
         self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired, vanilla):
        out_delta =   (desired - self.out)*(self.out*(1-self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1-self.hidout))

        self.W2+= (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2+=  (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1+=  (-1 * self.lrate * hid_delta)



    def decode(self, w):

        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size+ w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size+ w_layer2size:w_layer1size+ w_layer2size+self.Top[1]]
        self.B2 = w[w_layer1size+ w_layer2size+self.Top[1]:w_layer1size+ w_layer2size+self.Top[1]+self.Top[2]]


    def evaluate_proposal(self, data, w): # BP with SGD (Stocastic BP)

        self.decode(w) #method to decode w into W1, W2, B1, B2.

        size = data.shape[0]

        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for pat in xrange(0, size):

            Input[:]  =  data[pat,0:self.Top[0]]
            Desired[:] = data[pat,self.Top[0]:]

            self.ForwardPass(Input )
            fx[pat] = self.out

        return fx

#--------------------------------------------------------------------------

#-------------------------------------------------------------------


class MCMC:

    def __init__(self, samples, traindata, testdata, topology ):
        self.samples  = samples  # NN topology [input, hidden, output]
        self.topology = topology# max epocs
        self.traindata = traindata #
        self.testdata = testdata
        #----------------
    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self,  neuralnet, data,   w, tausq):
        y =    data[:,self.topology[0]]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.rmse(fx,y)
        loss =   -0.5*np.log(2*math.pi*tausq) - 0.5 * np.square(y -fx)/tausq
        return [np.sum(loss), fx, rmse]

    def sampler(self):

      #------------------- initialize MCMC
         testsize =  self.testdata.shape[0]
         trainsize = self.traindata.shape[0]
         samples = self.samples

         x_test = np.linspace(0, 1, num= testsize)
         x_train = np.linspace(0, 1, num= trainsize)

         netw = self.topology  # [input, hidden, output]
         y_test = self.testdata[:,netw[0]]
         y_train = self.traindata[:,netw[0]]
         print y_train.size
         print y_test.size

         w_size = (netw[0] * netw[1] )+ (netw[1] * netw[2] ) + netw[1] + netw[2]  # num of weights and bias

         pos_w = np.ones((samples, w_size)) #posterior of all weights and bias over all samples
         pos_tau = np.ones((samples,  1))

         fxtrain_samples = np.ones((samples, trainsize)) # fx of train data over all samples
         fxtest_samples = np.ones((samples, testsize))   # fx of test data over all samples
         rmse_train = np.zeros(samples)
         rmse_test = np.zeros(samples)

         w =  np.random.randn(w_size)
         w_proposal = np.random.randn(w_size)

         step_w=0.05;  # defines how much variation you need in changes to w
         step_eta=0.01;
     #--------------------- Declare FNN and initialize

         neuralnet = Network(self.topology, self.traindata, self.testdata  )
         print 'evaluate Initial w'

         pred_train = neuralnet.evaluate_proposal(self.traindata, w)
         pred_test =  neuralnet.evaluate_proposal(self.testdata, w)

         tautrain = np.var(pred_train - y_train)
         etatrain = np.log(tautrain)
         eta_pro_train = 0

         tautest = np.var(pred_test - y_test)
         etatest = np.log(tautest)
         eta_pro_test = 0

         [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet,  self.traindata,   w, tautrain)
         [likelihood_test, pred_test, rmsetest] = self.likelihood_func(neuralnet,  self.testdata,   w, tautest)



         print likelihood

         naccept = 0
         print 'begin sampling using mcmc random walk'
         plt.plot(x_train, y_train)
         plt.plot(x_train, pred_train )
         plt.title("Plot of Data vs Initial Fx")
         plt.savefig('mcmcresults/begin.png')
         plt.clf()

         plt.plot(x_train, y_train)

         for i in range(samples-1):



             w_proposal  =  w +   np.random.normal(0 , step_w , w_size)

             eta_pro_train  =  etatrain   + np.random.normal(0, step_eta, 1)
             eta_pro_test  =  etatest   + np.random.normal(0, step_eta, 1)
             tau_pro_train =  math.exp(eta_pro_train )
             tau_pro_test =  math.exp(eta_pro_test )



             [likelihood_proposal , pred_train, rmsetrain] = self.likelihood_func( neuralnet, self.traindata, w_proposal,   tau_pro_train )
             [likelihood_pro_test , pred_test, rmsetest] = self.likelihood_func( neuralnet, self.testdata, w_proposal,   tau_pro_test )




             diff = likelihood_proposal-likelihood

             mh_prob=min(1,math.exp(diff))

             u=random.uniform(0, 1)

             if u < mh_prob:
                 # Update position
                 print    i, ' is accepted sample'
                 naccept += 1
                 likelihood  = likelihood_proposal
                 w  = w_proposal
                 etatrain  = eta_pro_train
                 etatest  = eta_pro_test

                 print  likelihood , rmsetrain, rmsetest, w,    etatrain, etatest , 'accepted'

                 pos_w[i+1, ] =  w_proposal
                 pos_tau[i+1,] = tau_pro_train
                 fxtrain_samples[i+1,] = pred_train
                 fxtest_samples[i+1,] = pred_test
                 rmse_train[i+1,] = rmsetrain
                 rmse_test[i+1,] = rmsetest

                 plt.plot(x_train, pred_train)


             else:
                 pos_w[i+1,] = pos_w[i,]
                 pos_tau[i+1,] = pos_tau[i,]
                 fxtrain_samples[i+1,] = fxtrain_samples[i,]
                 fxtest_samples[i+1,] = fxtest_samples[i,]
                 rmse_train[i+1,] = rmse_train[i,]
                 rmse_test[i+1,] = rmse_test[i,]

                 #print i, 'rejected and retained'


         print naccept, ' num accepted'
         print naccept/(samples * 1.0), '% was accepted'
         accept_ratio = naccept/(samples * 1.0) * 100

         plt.title("Plot of Accepted Proposals")
         plt.savefig('mcmcresults/proposals.png')
         plt.savefig('mcmcresults/proposals.svg', format='svg', dpi=600)
         plt.clf()

         return ( pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)

def main():


        problem = 7 #  Lazer or Sunspot

        hidden = 5
        input = 4  #
        output = 1
        learnRate = 0.1
        mRate = 0.01
        MaxTime = 1000

        if problem == 1:
 	   traindata  = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/Lazer/test.txt") #
        if problem == 2:
           traindata  = np.loadtxt("Data_OneStepAhead/Sunspot/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/Sunspot/test.txt") #
        if problem == 3:
           traindata  = np.loadtxt("Data_OneStepAhead/Mackey/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/Mackey/test.txt") #
        if problem == 4:
           traindata  = np.loadtxt("Data_OneStepAhead/Lorenz/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/Lorenz/test.txt") #
        if problem == 5:
           traindata  = np.loadtxt("Data_OneStepAhead/Rossler/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/Rossler/test.txt") #
        if problem == 6:
           traindata  = np.loadtxt("Data_OneStepAhead/Henon/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/Henon/test.txt") #
        if problem == 7:
           traindata  = np.loadtxt("Data_OneStepAhead/ACFinance/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/ACFinance/test.txt") #







        print(traindata)




        topology = [input, hidden, output]


        MinCriteria = 0.005 #stop when RMSE reaches MinCriteria ( problem dependent)

        random.seed(time.time())

        numSamples = 20000  # need to decide yourself

        mcmc = MCMC(numSamples, traindata, testdata, topology ) #declare class

        [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler()
        print 'sucessfully sampled'

        burnin= 0.1* numSamples   # use post burn in samples

        pos_w = pos_w[int(burnin):,]
        pos_tau = pos_tau[int(burnin):,]


        fx_mu = fx_test.mean(axis=0)
        fx_high = np.percentile(fx_test, 95, axis=0)
        fx_low = np.percentile(fx_test, 5, axis=0)

        fx_mu_tr = fx_train.mean(axis=0)
        fx_high_tr = np.percentile(fx_train, 95, axis=0)
        fx_low_tr= np.percentile(fx_train, 5, axis=0)

        rmse_tr = np.mean(rmse_train[int(burnin):])
        rmsetr_std = np.std(rmse_train[int(burnin):])
        rmse_tes = np.mean(rmse_test[int(burnin):])
        rmsetest_std = np.std(rmse_test[int(burnin):])
        print rmse_tr, rmsetr_std, rmse_tes, rmsetest_std
	np.savetxt('mcmcresults/results.txt', (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')


        ytestdata = testdata[:,input]
        ytraindata = traindata[:,input]

        plt.plot(x_test, ytestdata, label ='actual')
        plt.plot(x_test, fx_mu, label = 'pred. (mean)')
        plt.plot(x_test, fx_low, label = 'pred.(5th percen.)')
        plt.plot(x_test, fx_high, label = 'pred.(95th percen.)')
        plt.fill_between(x_test, fx_low, fx_high,facecolor='g',alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Plot of Test Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/mcmcrestest.png')
        plt.savefig('mcmcresults/mcmcrestest.svg', format='svg', dpi=600)
        plt.clf()
#-----------------------------------------
        plt.plot(x_train, ytraindata, label ='actual')
        plt.plot(x_train, fx_mu_tr, label = 'pred. (mean)')
        plt.plot(x_train, fx_low_tr, label = 'pred.(5th percen.)')
        plt.plot(x_train, fx_high_tr, label = 'pred.(95th percen.)')
        plt.fill_between(x_train, fx_low_tr, fx_high_tr,facecolor='g',alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Plot of Train Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/mcmcrestrain.png')
        plt.savefig('mcmcresults/mcmcrestrain.svg', format='svg', dpi=600)

        plt.clf()

        plt.hist(rmse_train, bins=np.linspace(0, 0.1, num= 20))  # plt.hist passes it's arguments to np.histogram
        plt.title("RMSE train")
        plt.savefig('mcmcresults/rmsetrain.png')

        plt.clf()

        plt.hist(rmse_train, bins=np.linspace(0, 0.1, num= 20))  # plt.hist passes it's arguments to np.histogram
        plt.title("RMSE train")
        plt.savefig('mcmcresults/rmsetest.png')
        plt.clf()



if __name__ == "__main__": main()
