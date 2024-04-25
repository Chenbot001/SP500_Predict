## Setup ##
from mkdManager import MarketDataManager

# to be moved to separate file
host = 'localhost'
port = '5433'
dbname = 'market_data'
user = 'postgres'
pwd = 'Er!c374226'

## Connect to database ##
mdm = MarketDataManager(host,port,dbname,user,pwd)

## Insert data ##
try:
    mdm.update_sp500_ticker()
except Exception as e:
    print(str(e))

del mdm

