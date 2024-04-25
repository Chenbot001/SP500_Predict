import psycopg2 as ppg2
from datetime import date, timedelta
import pandas as pd
import yfinance as yf

class MarketDataManager:
    def __init__(self, host: str, port: int, dbname: str, 
                 user: str, pwd: str, batch_size: int = 1000):

        self.batch_size = batch_size

        conn_string = '''        
            host={} 
            port={} 
            dbname={} 
            user={} 
            password={} 
            sslmode=prefer 
            connect_timeout=10
            '''.format(host,port,dbname,user,pwd)

        self.conn = ppg2.connect(conn_string)
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.cursor.close()
        self.conn.close()

# fetch operations
    def get_sec_info(self) -> list[tuple[int, str, date]]:
        sql = '''
            select b.sec_id, s.ticker, max(b.date) last_date 
	        from stocks s
	        left join daily_bars b on b.sec_id = s.id
	        group by b.sec_id, s.ticker
            order by b.sec_id
        '''
        self.cursor.execute(sql)
        sec_info = self.cursor.fetchall()
        return sec_info

    def fetch_dbars_yf(self, ticker, start, end) -> pd.DataFrame:
        dbars = yf.download(ticker, start=start, end=end)
        return dbars

    def fetch_dbars_db(self, col, ticker: str, start: date, 
                         end:date) -> pd.DataFrame:
        sql = '''select date, %s from daily_bars 
        where sec_id = %s
        and date > %s
        and date < %s'''%(col, ticker,start,end)
        self.cursor.execute(sql)
        dbars = self.cursor.fetchall()
        return dbars

# insert operations
    def update_sp500_ticker(self) -> None:
        sp500_tickers = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        )[0]['Symbol'].tolist()
        
        sql = 'INSERT INTO stocks (ticker) values'
        for i in range(len(sp500_tickers)):
            sql = f"{sql} ('{sp500_tickers[i]}'),"
                
        sql = sql.rstrip(sql[-1])
        try:
            self.cursor.execute(sql)
            print('Update successful')
        except Exception as e:
            print('Update unsuccessful with exception ',str(e))
            pass
        self.conn.commit()

    def insert_dbars(self, sec_id:int, bars:pd.DataFrame) -> None:
        sql = 'INSERT INTO daily_bars (date, sec_id, open, high, low, close, adj_close, volume) values'
        for index, row in bars.iterrows():
            sql = f"{sql} ('{index.date()}',{sec_id},{row['Open']},{row['High']},{row['Low']},{row['Close']},{row['Adj Close']},{row['Volume']}),"
            
        sql = sql.rstrip(sql[-1])
        try:
            self.cursor.execute(sql)
        except Exception as e:
            print(str(e))


    def insert_dbars_batch(self, sec_id, bars:pd.DataFrame) -> None:
        start = 0
        end = None
        numRows = len(bars)

        while start < numRows:
            end = start + self.batch_size
            if end > numRows:
                end = numRows
            
            self.insert_dbars(sec_id, bars[start:end])
            start = end
        

    def refresh_dbars(self) -> None:
        sec_info = self.get_sec_info()
        count = 1
        for s in sec_info:
            print(s[1],' ',count,'/503')
            count = count + 1
            try:
                bars = self.fetch_dbars_yf(s[1], start = s[2] + timedelta(days=1), end = date.today() + timedelta(days=1))
                self.insert_dbars(s[0],bars)
            except Exception:
                pass            
        self.conn.commit()          
#######################################################################################