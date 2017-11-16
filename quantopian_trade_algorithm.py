"""
I am aiming to employ an ensemble learning based model for analyzing stock trends and computing a reliable prediction for the price at the end of the day. The architecture or pseudo code that I aim to follow is something similar to this:

1. Obtain price of equity within 1 minute of day opening. Store this floating point value as a global hyperparameter. It will be compared with the prediction in later stages.
2. Construct a Neural Network based on backpropagation. This is proven to be more reliable than genetic algorithms for NNs that have a low magnitude of hidden layers.
3. Using sklearn, initiate  a KNNRegressor and use this to create a model of prices.
4. Implement pipeline with relevant and 'nice' filters/factors to obtain well-performing equities and append the equities to a global hyperparameter. This will be a list of strings that the quantopian api can recognize as equities.  
5. Using the quantopian API, import data pertaining to the equity aimed to analyze. Establish a look-back range of at least 100 days. 
6. Plug in the technical indicator values of the particular day and plug them into the KNNRegressor and the NeuralNetwork. The outputs of both will be aggregated into one, average value. 
7. Based on a simple decision algorithm, buy or short the selected stock. Establish rules to ensure that portfolio is hedged.
""" 

""" QUANTOPIAN API """
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import Returns, RSI
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.filters import Q1500US
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.classifiers.morningstar import Sector # import an equity universe.

""" DATA SCIENCE/ML MODULES """
from sklearn.neighbors import KNeighborsRegressor
from scipy import optimize
import numpy as np
import pandas as pd

""" CREATE NEURAL NETWORK (inspired from Welch's NN Definition https://www.youtube.com/user/Taylorns34 """

""" START OF NEURAL NETWORK CLASS DEFINITION """
class Neural_Network(object):
    def __init__(self, Lambda = 0):
        # the hyperparameters (do not change) go over here
        self.input_size = 3 # units in number of neurons
        self.hidden_size = 4
        self.output_size = 1 
        #------------------------------------------------------------------
        # initialize random weight definitions
        self.weight_1 = np.random.randn(self.input_size, self.hidden_size)
        self.weight_2 = np.random.randn(self.hidden_size, self.output_size)
        # regularization param
        self.Lambda = Lambda
        
    def forward_propagation(self, X): # exactly the same as welch's definition
        # h denotes values at the hidden layer
        # H denotes values of hidden layer after sigmoid activation
        # O denotes values at output layer
        # ans denotes value of output after sigmoid activation
        self.h = np.dot(X, self.weight_1)
        self.H = self.sigmoid(self.h)
        self.O = np.dot(self.H, self.weight_2)
        ans = self.sigmoid(self.O)
        return ans    
    
    def cost_function(self, X, y):
        ''' Very important function: finds cost of the forward_propagation'''
        self.ans = self.forward_propagation(X)
#         print(self.ans)
#         print(y)
        cost = 0.5*np.sum(np.square(y - self.ans))/X.shape[0] \
                        + (self.Lambda/2)*(np.sum(np.square(self.weight_1))+np.sum(np.square(self.weight_2)))
        return cost
    
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward_propagation(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.O))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.H.T, delta3)/X.shape[0] + self.Lambda*self.weight_2

        delta2 = np.dot(delta3, self.weight_2.T)*self.sigmoidPrime(self.h)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.weight_1
        return dJdW1, dJdW2

    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.weight_1.ravel(), self.weight_2.ravel()))
        return params

    def setParams(self, params):
        weight_1_start = 0
        weight_1_end = self.hidden_size * self.input_size
        self.weight_1 = np.reshape(params[weight_1_start:weight_1_end], \
                             (self.input_size, self.hidden_size))
        weight_2_end = weight_1_end + self.hidden_size*self.output_size
        self.weight_2 = np.reshape(params[weight_1_end:weight_2_end], \
                             (self.hidden_size, self.output_size))    
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
        
    # -----------------------------------------------------
    # Utility functions
    # -----------------------------------------------------
    def sigmoid(self, to_convert):
        return 1/(1+np.exp(-to_convert))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
""" END OF NEURAL NETWORK CLASS DEFINITION """

""" START OF TRAINER CLASS DEFINITION """
# defining the trainer class; it inherits the neural network.
class Trainer(object):
    def __init__(self, inherited_NN, X, y):
        '''inherits using the NN object, also delivers values of X and y to each instance 
        of the object.'''
        self.NN = inherited_NN
        self.X = X
        self.y = y
    
    def costFunctionWrapper(self, params):
        self.NN.setParams(params)
        cost = self.NN.cost_function(self.X, self.y)
        gradient = self.NN.computeGradients(self.X, self.y)
        return cost, gradient
    
    def train(self):        
        params0 = self.NN.getParams()

        options = {'maxiter': 500, 'disp' : False}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 options=options)

        self.NN.setParams(_res.x)
        self.optimizationResults = _res
""" END OF TRAINER CLASS DEFINITION """
""" END OF NEURAL NETWORK CONSTRUCTION """

""" START OF TECHNICAL INDICATOR CALCULATION """

def find_SMA(pricing, window):
    df = pd.DataFrame()
    df['SMA'] = pricing.rolling(window=window,center=False).mean()
    df = df.dropna()
    return df['SMA']
    
def find_bolling_bandwidth(pricing, common_window):
    df = pd.DataFrame()
    df['SMA'] = find_SMA(pricing, common_window)
    df['STDev'] = pricing.rolling(window = common_window, center = False).std()
    upper_band = df['SMA'] + 2*df['STDev']; lower_band = df['SMA'] - 2*df['STDev']
    df['Bandwidth'] = ((upper_band - lower_band) / df['SMA']) * 100
    df = df.dropna()
    return df['Bandwidth']

def find_price_momentum(pricing, window):
    df = pd.DataFrame()
    df['close_price'] = pricing
    df['PM'] = pricing / pricing.shift(window)
    df = df.dropna()
    return df['PM']

def gather_technical_indicators(pricing_data, lookback_window):
    """ Gathers the calculations made by technical indicator functions initiated in the beginning of algorithm.
    Appends them to one large dataframe with price, and technical indicator side-by-side 
    """
    master_df = pd.DataFrame()
    master_df['Price'] = pricing_data
    master_df['SMA'] = find_SMA(pricing_data, lookback_window)
    master_df['Bandwidth'] = find_bolling_bandwidth(pricing_data, lookback_window)
    master_df['PM'] = find_price_momentum(pricing_data, lookback_window)
    return master_df.dropna()    

""" END OF TECHNICAL INDICATOR CALCULATION """

def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    context.lookback_window = 3 # look back window for technical indicator calculations.
    context.data_history_window = 150 # we will accumulate data from the past 1000 days.
    context.price_current = [] # initialize empty current price list. This is a float.
    context.equities = [] # list of type string. Will contain the names of all equities traded by the algorithm
    context.technical_indicator_data = pd.DataFrame() # initialize an empty pandas dataframe that can hold data of three technical indicators
    context.leverage_buffer = 0.95 # limits leverage, or equities brought on credit. This will solve problems of overshorting our portfolio 
    context.weight = {}
    context.ENSEMBLE_PREDICTION = 0.0
 
    # gather the technical indicators at 10 AM each stock-trading day        
    schedule_function(get_initial_price, 
                      date_rules.every_day(),
                      time_rules.market_open(minutes = 1))
    # conduct calculations using the NN and KNN
    schedule_function(predict_and_weigh,
                      date_rules.every_day(), 
                      time_rules.market_open(minutes = 2))
    # trade
    schedule_function(trade, 
                      date_rules.every_day(),
                      time_rules.market_open(minutes = 3))
    # close all the positions made in previous day
    schedule_function(close_position,
                      date_rules.every_day(),
                      time_rules.market_close(minutes = 1))
    # Record tracking variables at the end of each day. Variables will include the correlation of training data
    schedule_function(my_record_vars, 
                      date_rules.every_day(), 
                      time_rules.market_close())
    # Create our dynamic stock selector.
    attach_pipeline(make_pipeline(), 'my_pipeline')

        
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """
    base_universe = Q1500US()
    sector = Sector()    
    # screen is based off of returns
    returns = Returns(window_length = 2)
    # check if stock price has good strength, but not necessarily overbought
    rsi = RSI() 
    price = USEquityPricing.close.latest
    # creating filter by specifying the type of returns desired
    top_return_stocks = returns.top(1,mask=base_universe, groupby=sector)
    pipe = Pipeline(
        columns = {
            'rsi': rsi,
            'price': price
        },
        # filter top return stocks, and stocks that are not being overbought
        # but are not too oversold either
        screen = base_universe & top_return_stocks & (20 < rsi < 80)
        # the above is equivalent to: choose stocks from the base universe that have had the top returns in their sectors and have a good RSI value
    )
    return pipe
 
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    # These are the securities that we are interested in trading each day.
    context.output = pipeline_output('my_pipeline')
    context.equities = context.output.index.tolist()
    log.info("Stocks today") 
    print(context.equities)
        
def get_initial_price(context, data):
    """
    IMPORTANT NOTES: TRY TO IMPLEMENT A DICTIONARY WITH KEYS AS THE EQUITY NAMES AND VALUES AS THEIR CURRENT PRICE 
    THIS WAY WE CAN AVOID ASSUMING THE THE ORDER OF THE INTIAL VALUES IN A LIST IS THE SAME AS THE ORDER OF EQUITY 
    NAMES IN ANOTHER LIST. THIS RUNS 30 MINUTE FROM MARKET OPENING. (10AM)
    """
    context.current_prices = {}
    for equity in context.equities:
        context.current_prices[equity] = data.current(equity, 'price')
        #print("{} {}".format(str(equity), context.current_prices[equity]))      
        
def predict_and_weigh(context,data):
    """
    Called every day at 10:01 AM.
    """
    context.predicted_prices = {}
    context.weights = {}
    for equity in context.equities:
        # initialize the data_history. 
        equity_price_data = pd.Series(data.history(equity, 'price', context.data_history_window, '1d'))
        # gets price data every day for the 100 previous days 
        # this ensures the the data for the current day is not appended. data.history in made in a way that it concatenates the current price as well. 
        context.technical_indicator_data = gather_technical_indicators(equity_price_data, context.lookback_window)
        # Now we have the technical data for the stock defined by the for loop
        """ 
        CREATE A NEURAL NETWORK OBJECT THAT IS TRAINABLE 
        """
        NN = Neural_Network(Lambda = 0.000001) # class definition at beginning of python script
        # We use a small regulatization parameter as we are dealing with a high amount of data found. 
        # Once can imagine this number acting as sort of an error factor. If there are more data points, then
        # the regularization constant will approach the point of UNDERfitting, where the prediction is not even
        # close to the train/test batch.
        KNN = KNeighborsRegressor(n_neighbors = 5, weights = 'distance') # Initiated KNN obtained from scikit_learn
        """
        END OF NEURAL NETWORK AND TRAINER OBJECT INSTANTIATION
        """
        # We can now split the technical indicator data into 2 parts. 
        # First, the columns that actually contain the technical indicator data
        # and second, the single column that contains the prices. These are X and y, respectively.
        yTrain = context.technical_indicator_data.ix[:, 0].values[:-1] # numpy ndarray of prices
        yTrain = np.reshape(yTrain, (yTrain.shape[0],1)) # Has proper shape (# rows, 1)
        XTrain = context.technical_indicator_data.ix[:, [1,2,3]].values[:-1] # numpy ndarray of technical indicators
        ymax = np.max(yTrain) # Very important! needed later for de-normalization purposes 
        Xmax = np.max(XTrain)
        # Normalization of data...
        XTrain = XTrain / np.max(XTrain)
        yTrain = yTrain / ymax
        """ 
        RUN TRAINING/FITTING TECHNIQUES
        """
        T = Trainer(NN, XTrain, yTrain)
        T.train()
        KNN.fit(XTrain, yTrain)
        # Introduce the testing data
        XTest = context.technical_indicator_data.ix[:, [1,2,3]].values[:-1]
        XTest = XTest / Xmax # Make sure the most current row of data is being used to test, and that it is normalized using the XMax of the training batch
        NN_predict = T.NN.forward_propagation(XTest)
        KNN_predict = KNN.predict(XTest)
        context.prediction_arr = np.array([NN_predict, KNN_predict])
        # Find the average of the NN and KNN prediction
        context.ENSEMBLE_PREDICTION = np.average(context.prediction_arr)
        """ 
        END OF TRAINING/FITTING TECHNIQUES
        """
        
        # I will base the weights on the percent change that the stock will experience, based off of the 
        # prediction made by the stock. I will make them inversely proportionate.
        context.predicted_prices[equity] = context.ENSEMBLE_PREDICTION * ymax
        #print("Prediction is, ", context.ENSEMBLE_PREDICTION * ymax)
    """
    START OF WEIGHT DISTRIBUTION
    """
    total_delta = np.sum(np.abs(np.array(context.current_prices.values()) - np.array(context.predicted_prices.values())))

    # Find the total differences between predictions and initial prices. This will be used to find the weights for
    # each, individual equity.
    for equity in context.equities:
        difference = np.abs(context.current_prices[equity] - context.predicted_prices[equity])
        context.weights[equity] = 1 - float(difference)/total_delta
        #print('weight for ', equity, 'is', context.weights[equity])
        
    sum_weights = np.sum(context.weights.values())
    for equity in context.equities:
        context.weights[equity] = context.weights[equity] / (sum_weights / 0.5)
        #print('regulated weight for ', equity, 'is', context.weights[equity])
        
def trade(context, data):
    """
    INITIATE TRADING DECISIONS
    """
    # We already know that predicted prices and weights dictionaries are filled. 
    for equity in context.equities:
        if(context.predicted_prices[equity] < context.current_prices[equity] * 0.95):
            # Trade if there is more than a 1% decrease (SHORT)
            order_target_percent(equity, (-1 * context.weights[equity]))
        elif(context.predicted_prices[equity] >= context.current_prices[equity] * 1.05):
            # Trade if there is more than a 1% increase (LONG)
            order_target_percent(equity, float(context.weights[equity]))
        else:
            continue
    """
    END OF TRADING DECISIONS
    """
    
def close_position(context, data):
    for equity in context.portfolio.positions:
        order_share = context.portfolio.positions[equity].amount
        order(equity,-order_share)
        
def my_record_vars(context, data):
    """
    PLOT VARIABLES AT END OF EACH DAY
    """
    # Check how many long and short positions we have.
    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        if position.amount < 0:
            shorts += 1
    # Record and plot the leverage of our portfolio over time as well as the
    # number of long and short positions. Even in minute mode, only the end-of-day
    # leverage is plotted.
    record(leverage = context.account.leverage, long_count=longs, short_count=shorts)
    """
    END OF PLOTTING VARIABLES 
    """
    
    
            
           
           
           
           
           

