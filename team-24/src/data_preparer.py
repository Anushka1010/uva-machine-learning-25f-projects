#Dataset: https://www.kaggle.com/datasets/rdolphin/financial-news-with-ticker-level-sentiment
#pip install pandas yfinance

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import json
import warnings
warnings.filterwarnings('ignore')

class FinancialNewsAnalyzer:

    def __init__(self, input_file, output_file="financial_news_with_prices.csv"):
        self.input_file = input_file
        self.output_file = output_file
        self.cache = {}
        self.cache_file = "price_cache.json"
        self.load_cache()

    def load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
            print(f"Loaded {len(self.cache)} cached price entries")
        except:
            self.cache = {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
        print(f"Saved {len(self.cache)} price entries to cache")

    def load_data(self):
        print("Loading financial news dataset from JSON...")

        with open(self.input_file, 'r') as f:
            self.raw_data = json.load(f)

        print(f"Loaded {len(self.raw_data)} articles from JSON")

        return self.raw_data

    def parse_data(self):
        print("\nParsing JSON news data...")

        parsed_rows = []

        for idx, article in enumerate(self.raw_data):
            if idx % 1000 == 0:
                print(f"Processing article {idx}/{len(self.raw_data)}")

            # Extract headline/title
            headline = article.get('title', '')

            # Extract date
            date_str = article.get('published_utc', '')

            # Extract tickers (array format in JSON)
            tickers = article.get('tickers', [])


            if not headline or not date_str or not tickers:
                continue

            # Parse date
            try:
                date_obj = pd.to_datetime(date_str)
                date_only = date_obj.date()
            except:
                continue

            # Create row for each ticker
            for ticker in tickers:
                if ticker and isinstance(ticker, str):
                    parsed_rows.append({
                        'headline': headline,
                        'date': date_only,
                        'stock_ticker': ticker.upper(),
                    })

        self.parsed_df = pd.DataFrame(parsed_rows)
        print(f"\nParsed into {len(self.parsed_df)} rows")
        print(f"Date range: {self.parsed_df['date'].min()} to {self.parsed_df['date'].max()}")
        print(f"Unique tickers: {self.parsed_df['stock_ticker'].nunique()}")

        return self.parsed_df

    def get_price_change(self, ticker, date):
        # Create cache key
        cache_key = f"{ticker}_{date}"

        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Fetch data from yfinance
            start_date = datetime.combine(date, datetime.min.time())
            end_date = start_date + timedelta(days=5)

            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)

            if len(hist) == 0:
                result = {
                    'open': None,
                    'close': None,
                    'change': None,
                    'pct_change': None
                }
            else:
                # Get the first available trading day
                open_price = float(hist['Open'].iloc[0])
                close_price = float(hist['Close'].iloc[0])
                change = close_price - open_price
                pct_change = (change / open_price * 100) if open_price != 0 else 0

                result = {
                    'open': round(open_price, 2),
                    'close': round(close_price, 2),
                    'change': round(change, 2),
                    'pct_change': round(pct_change, 2)
                }

            # Cache the result
            self.cache[cache_key] = result

            return result

        except Exception as e:
            return {
                'open': None,
                'close': None,
                'change': None,
                'pct_change': None
            }

    def enrich_with_prices(self, max_rows=None, delay=0.05):
        print("\nEnriching data with stock prices...")

        if max_rows:
            df = self.parsed_df.head(max_rows).copy()
            print(f"Processing first {max_rows} rows for testing")
        else:
            df = self.parsed_df.copy()

        opens = []
        closes = []
        changes = []
        pct_changes = []

        total = len(df)
        start_time = time.time()

        for idx, row in df.iterrows():
            if idx % 100 == 0 and idx > 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                remaining = (total - idx) / rate if rate > 0 else 0
                print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%) - "
                      f"ETA: {remaining/60:.1f} minutes")

            ticker = row['stock_ticker']
            date = row['date']

            price_data = self.get_price_change(ticker, date)

            opens.append(price_data['open'])
            closes.append(price_data['close'])
            changes.append(price_data['change'])
            pct_changes.append(price_data['pct_change'])

            # Save cache periodically
            if idx % 100 == 0:
                self.save_cache()

            time.sleep(delay)

        # Add columns to dataframe
        df['open_price'] = opens
        df['close_price'] = closes
        df['price_change'] = changes
        df['price_change_pct'] = pct_changes

        self.save_cache()

        with_data = df['price_change'].notna().sum()
        print(f"\nCompleted enrichment!")
        print(f"Total rows: {total}")
        print(f"Rows with price data: {with_data} ({with_data/total*100:.1f}%)")
        print(f"Rows without data: {total - with_data}")

        self.final_df = df
        return df
    
    def save_results(self):
        print(f"\nSaving results to {self.output_file}...")

        output_columns = ['headline', 'date', 'stock_ticker', 'price_change']
        optional_columns = ['open_price', 'close_price', 'price_change_pct']
        for col in optional_columns:
            if col in self.final_df.columns:
                output_columns.append(col)

        output_df = self.final_df[output_columns]
        output_df.to_csv(self.output_file, index=False)

        print(f"Saved {len(output_df)} rows to {self.output_file}")

        return output_df

    def run(self, max_rows=None):
        print("=" * 70)
        print("FINANCIAL NEWS STOCK PRICE ANALYSIS")
        print("=" * 70)
        self.load_data()
        self.parse_data()
        self.enrich_with_prices(max_rows=max_rows)
        self.save_results()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)

def main():
    INPUT_FILE = "polygon_news_sample.json"
    OUTPUT_FILE = "financial_news_with_prices.csv"
    MAX_ROWS = 100  # set to 100 for code demo, None for all rows. 
    analyzer = FinancialNewsAnalyzer(INPUT_FILE, OUTPUT_FILE)
    analyzer.run(max_rows=MAX_ROWS)

if __name__ == "__main__":
    main()
