
## Web Scraping Financial Data from stockanalysis.com

### This project aims to extract financial data from the stockanalysis.com website using web scraping techniques. The extracted data can be used for financial analysis and modeling.

1) Installation
To install the required libraries, run:

pip install -r requirements.txt

2) Usage

To run the web scraper, run:

main_stock_finantial.py in order to extract .xlsx file with datasets for each important aspect of the companie or stock that youre interested in. the Symbol variable is where you should put the convention, for example EC for Ecopetrol data.

This .xlsx file can then be read by eda_generator.py that automatically proceeds to create EDA jupyter notebook files for each sheet of the .xlsx file you just generated.

scrapping_canvas.py is a file that can extract important chat in PNG using the same symbol that you used in main_stock_finantial.  the EDA jupyter notebooks that are in this repository where created using eda_generator.py and implement sweetviz librarie for data analysis creating an HTML report for each excel file.

The scraped data will be saved to a .xlsx file in the data/ directory.