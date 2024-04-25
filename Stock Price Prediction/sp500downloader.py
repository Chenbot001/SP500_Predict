#%%
## Setup ##
from mkdManager import MarketDataManager

## Main ####################################################################

# to be moved to separate file
host = 'localhost'
port = '5433'
dbname = 'market_data'
user = 'postgres'
pwd = 'Er!c374226'

## Connect to database ##
mdm = MarketDataManager(host,port,dbname,user,pwd)
mdm.refresh_dbars()

#%%
del mdm