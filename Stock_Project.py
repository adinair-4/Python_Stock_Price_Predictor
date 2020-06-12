import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import linear_model
from scipy import stats
import datetime
import csv
import warnings
warnings.filterwarnings('ignore')

# Method to accept company name from user
def accept_company_name():

    found_company = False
# Run until a company is selected

    while found_company == False:

        with open("C:/Users/Adi/OneDrive/Documents/MSc BA/Programming for Analytics/Project/Final/companylist.csv") as f:

            company_reader = csv.reader(f)

            company_name = input("Please enter the company name to analyse the stock data: \n")

            for row in company_reader:

                if company_name in row[1].lower():

                    print('Company name: ', row[1])

                    print('Tickr Code: ', row[0])

                    found_company = True

        if not found_company:

            print("\nSorry we could not find the company with the entered name.")
    

# Method to accept company tickr code from user


def accept_tickr_code():

    tickr_code = None

    found_tickr = False

    while not found_tickr:

        print("\nPlease select company tickr code from the list above: ")

        tickr_code = input().upper()

        tickr_list = []

        with open("C:/Users/Adi/OneDrive/Documents/MSc BA/Programming for Analytics/Project/Final/companylist.csv") as f:

            for row in f:

                tickr_list.append(row.split(',')[0])

        if tickr_code in tickr_list:

            print("\nEntered company tickr code is correct")

            found_tickr = True

        else:

            print("\nPlease enter the correct company tickr code !")

    return tickr_code

# Method to accept start and end dates
def accept_start_end_dates():

    start_flag = False

    while not start_flag:

        start_input = input("\nPlease enter the Start Date in YYYY-MM-DD format: ")

        start_year, start_month, start_day = map(int, start_input.split('-'))

        start_flag = validate_date(start_year, start_month, start_day)

        if not start_flag:

            print("\nPlease enter a valid date in the specified format")

        else:

            print("\nThank you. The start date entered is Valid")

    end_flag = False

    while not end_flag:

        end_input = input("\nPlease enter the end Date to see the descriptive analysis of the company in YYYY-MM-DD format: ")

        end_year, end_month, end_day = map(int, end_input.split('-'))

        end_flag = validate_date(end_year, end_month, end_day)

        if not end_flag:

            print("Please enter a valid date in the specified format")

        else:

            if end_input < start_input or end_input == start_input:

                print("\nInvalid end date. End date should be greater than start date !")

                end_flag = False

    return start_year, start_month, start_day, end_year, end_month, end_day

# Method to check if the user has enetered a valid date
def validate_date(year, month, day):

    date_time = False

    try:

        datetime.datetime(year, month, day)

        date_time = True

    except Exception:

        date_time = False

    return date_time

# Method to define Trend Line

def trend_line(stock_df):
    plt.style.use('dark_background')
    trend_data0 = stock_df.copy()

    trend_data0['Date'] = ((stock_df.index.date - stock_df.index.date.min())).astype('timedelta64[D]')

    trend_data0['Date'] = trend_data0['Date'].dt.days + 1

   # High trend line
    trend_data1 = trend_data0.copy()

    while len(trend_data1) > 3:

        regression = stats.linregress(x=trend_data1['Date'],y=trend_data1['High'], )

        trend_data1 =  trend_data1.loc[trend_data1['High'] > regression[0] * trend_data1['Date'] + regression[1]]

    regression = stats.linregress(x=trend_data1['Date'],y=trend_data1['High'],)

    trend_data0['high_trend'] = regression[0] * trend_data0['Date'] + regression[1]

    # Low Trend line

    trend_data1= trend_data0.copy()

    while len(trend_data1) > 3:

        regression = stats.linregress(x= trend_data1['Date'],y= trend_data1['Low'],)

        trend_data1 =  trend_data1.loc[ trend_data1['Low'] < regression[0] *  trend_data1['Date'] + regression[1]]

    regression = stats.linregress(x= trend_data1['Date'],y= trend_data1['Low'],)

    trend_data0['Low_trend'] = regression[0] * trend_data0['Date'] + regression[1]

    trend_data0['Close'].plot()

    trend_data0['high_trend'].plot()

    trend_data0['Low_trend'].plot()
    
    plt.grid(True, linewidth=0.40, color='#ff0000', linestyle='-')
             
    plt.legend()

    plt.xlabel('Date')

    plt.ylabel('Stock Price')

    plt.suptitle(tickr_code+' Stock Price')

    plt.show()
    

# Method to plot the moving average graph
    
def moving_average(stock_df):
    plt.style.use('dark_background')

    n_value= input("Enter the n value for Moving Average: ")

    stock_df['Moving_Average'] = stock_df['Close'].rolling(window=int(n_value),min_periods=0). mean()

    ma_plot = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)

    ma_plot.plot(stock_df.index, stock_df['Close'], color= "red", label = "Closing Price")

    ma_plot.plot(stock_df.index, stock_df['Moving_Average'], color="green", label = "Moving Average")
    
    plt.grid(True, linewidth=0.40, color='#ff0000', linestyle='-')

    plt.xlabel('Date')

    plt.ylabel('Stock Price')
    
    plt.legend()

    plt.suptitle(tickr_code+' Stock Price')

    plt.show()

# Method to predict prices using linear regression
def predict_price(ordinal_dates, price_list, x):

    # Defining the linear regression model

    reg_model = linear_model.LinearRegression()

    ordinal_dates = np.reshape(ordinal_dates, (len(ordinal_dates), 1))  # Convert to matrix of n X 1

    price_list = np.reshape(price_list, (len(price_list), 1))

    reg_model.fit(ordinal_dates, price_list)  # Putting the data points in the model

    price = reg_model.predict([[x]])

    return price[0][0], reg_model.coef_[0][0], reg_model.score(ordinal_dates, price_list)
    
# Start of the "Main Program"

print("\n *** Welcome to Python Project! ***\n")

main_run = True

while main_run:

    # Accepting company name displaying tickr codes

    accept_company_name()

    # Accepting the TICKR code from the user

    tickr_code = accept_tickr_code() #returns the tickr code 

    # Accepting and validating start and end dates

    start_year, start_month, start_day, end_year, end_month, end_day = accept_start_end_dates() 
    
    # Store start and end dates

    start_date = datetime.datetime(start_year, start_month, start_day) # This method will return start date object and store in start_date 
    

    end_date = datetime.datetime(end_year, end_month, end_day)  # This method will return end date object and store in end_date 

    # Using yahoo service to fetch company stock details

    stock_df = web.DataReader(tickr_code, "yahoo", start_date, end_date)   # Getting the data for the selected company online and storing in df

    # Saving data frame into a csv file

    stock_df.to_csv('company_details.csv') # All details from the web is fetched and stored in the csv file

    # Showing the descriptive analysis of the selected company

    print("\nDescriptive analytics of the selected company") 

    # General discription of the dataframe using describe method

    print(stock_df.describe())

    closing_val_variation = stats.variation(stock_df['Close'], axis=0) # Variation is calculated using the closing values

    print("\nVariation=", closing_val_variation)

    # Graphical visualization of the data and making prediction for the selected company
    while True:

        print("\nMENU:  ")

        print("1. Trend Line Plot")

        print("2. Raw Time-Series and Moving Averages Plot")

        print("3. Raw Time-Series and Weighted Moving Averages Plot")

        print("4. MACD Plot")

        print("5. Predict the stock price")

        print("6. Quit for current company")

        user_choice = input("\nPlease choose an option between 1 and 6: ")

        if user_choice == "1":

            trend_line(stock_df)  # stock_df is dataframe which we got from web(Yahoo)

        elif user_choice == "2":

            moving_average(stock_df)

        elif user_choice == "3":
            plt.style.use('dark_background')
            print("\nEnter the n value for Weighted Moving Average:")
           
            n_value = input()

            stock_df['ewma'] = stock_df['Close'].ewm(span=int(n_value)).mean() # Addding column and storing in dataframe stock_df['emwa']

            wma_plot = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

            wma_plot.plot(stock_df.index, stock_df['Close'], color="red", label='Closing Price')

            wma_plot.plot(stock_df.index, stock_df['ewma'], color="green", label='WMA')
            
            plt.grid(True, linewidth=0.40, color='#ff0000', linestyle='-')

            plt.xlabel('Date')

            plt.ylabel('Stock Price')

            plt.legend()

            plt.suptitle(tickr_code+' Stock Price')

            plt.show()

        elif user_choice == "4":
            plt.style.use('dark_background')

            x = 26

            y = 12

            stock_df['26 ema'] = stock_df['Close'].ewm(span=int(x)).mean()

            stock_df['12 ema'] = stock_df['Close'].ewm(span=int(y)).mean()

            stock_df['MACD'] = (stock_df['12 ema'] - stock_df['26 ema'])

            macd_plot = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

            macd_plot.plot(stock_df.index, stock_df['Close'],color="red", label='Closing Price')

            macd_plot.plot(stock_df.index, stock_df['MACD'],color="green", label='MACD')
            
            plt.grid(True, linewidth=0.40, color='#ff0000', linestyle='-')

            plt.xlabel('Date')

            plt.ylabel('Stock Price')

            plt.legend()

            plt.suptitle(tickr_code+' Stock Price')

            plt.show()

        elif user_choice == "5":
        
            predict_date_flag = False

            while not predict_date_flag:

                query_date = input("\nEnter the date for which you want to predict the stock price in yyyy-mm-dd format: ")

                pred_year, pred_month, pred_day = map(int, query_date.split('-'))

                predict_date_flag = validate_date(start_year, start_month, start_day)

                if not predict_date_flag:

                    print("\nEnter the correct date")

                else:

                    # calculate the ordinal value of the prediction date
                    
                    to_predict_date = datetime.date(year=pred_year, month=pred_month, day=pred_day).toordinal() ##date object using datetime library and then converting it into ordinal value

                    days = []

                    months = []

                    years = []

                    prices = []

                    ordinal_days = []

                    with open('company_details.csv', 'r') as company_file:

                        company_file_reader = csv.reader(company_file)

                        next(company_file_reader)  # Skipping column names

                        for row in company_file_reader: # Reading the company_file total rows(iterable sequence) until the end

                            d_values = row[0].split('-')

                            days.append(int(d_values[2]))

                            months.append(int(d_values[1]))

                            years.append(int(d_values[0]))

                            prices.append(float(row[4]))

                    # Storing all the ordinal values of dates into an array

                    for index in range(len(days)):  # Looping the total number of days (Procedural Programming)

                        day_val = datetime.date(year=years[index], month=months[index], day=days[index])

                        day_val = day_val.toordinal()

                        ordinal_days.append(day_val)

                    predicted_price, coeff, score = predict_price(ordinal_days, prices, to_predict_date)
                   
                    print("The stock predicted price for "+str(pred_year) + "-" +

                          str(pred_month)+"-"+str(pred_day)+" is: $", str(predicted_price))

                    print("The regression coefficient is ", str(coeff))
                    print("The R^2 score is",str(score))
                    
        
        elif user_choice == "6":

            break

        else:

            print("Incorrect entry. Please try again !")

    print("\nDo you want to check for another company 1. Yes  2. No")

    user_choice2 = input()

    if int(user_choice2) == 1:

        main_run = True

    else:

        main_run = False

        print("\nThank you!!")



