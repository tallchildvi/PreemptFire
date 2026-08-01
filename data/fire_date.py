import pandas as pd
import requests
from dotenv import load_dotenv
import os
from datetime import datetime, date

load_dotenv()
MAP_KEY = os.getenv("FIRMS_API_KEY")

class Date():
    def __init__(self, fire_date: date):
        self.fire_date = fire_date.strftime("%Y-%m-%d")

    def get_fires(self):
        load_dotenv()
        MAP_KEY = os.getenv("FIRMS_API_KEY")
        url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/" + MAP_KEY + "/VIIRS_SNPP_SP/-141,41.5,-52.0,83.5/1/" + self.fire_date

        try:
            df_area = pd.read_csv(url)
            print(df_area)
            return df_area

        except Exception as e:
            print(e)
            print ("There is an issue with the query. \nTry in your browser: %s" % url)

if __name__ == "__main__":
    test_date = date(2023, 6, 15)
    date1 = Date(test_date)
    print(date1.get_fires())
