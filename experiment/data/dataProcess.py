import pandas as pd
from datetime import datetime

# Load the CSV file into a DataFrame

try:

    #df = pd.read_csv('output_9_13_.csv')
    #df = pd.read_csv('output_14_18_.csv')
    df = pd.read_csv('output_9_18.csv')
    # Specify the column containing the date strings
    date_column = 'Time'
    time_of_day = None
    semester = None
    # Function to extract month, year, and time from date string
    def extract_date_info(date_str):

        date_obj = datetime.strptime(date_str, '%m/%d/%Y %H:%M')

        # Extract time part as string in HH:MM format
        time_str = date_obj.strftime('%H:%M')

        # Determine time of day based on time string
        if time_str >= '12:00' and time_str <= '16:59':
            time_of_day = 1 #'afternoon'
        elif time_str >= '17:00' and time_str <= '20:59':
            time_of_day = 2 #'evening'
        elif time_str >= '21:00' or time_str < '05:00':
            time_of_day = 3 #'night'
        else:
            time_of_day = 0 #'morning'

        print(f"The time {time_str} is classified as {time_of_day}")

    #to determine semester based on month
        if date_obj.month >=8:
            semester =1
        else:
            semester =2

        return date_obj.month, date_obj.year, date_obj.strftime("%H:%M"), date_obj.weekday(), time_of_day, semester

    # Apply the function to the specified column
    df[['Month', 'Year','Extracted_Time', 'WeekDay','TimeOfDay', 'Semester']] = df[date_column].apply(lambda x: pd.Series(extract_date_info(x)))

    # Specify the output CSV file where you want to add the new columns
    #output_filename = 'output_9_13_5.csv'
    output_filename = 'output_9_18_5.csv'

    # Append the extracted columns to the existing CSV file
    #df.to_csv(output_filename, mode='a', header=True, index=False)
    df.to_csv(output_filename, index=False)

except pd.errors.ParserError as e:
    print(f"ParserError: {e}")

