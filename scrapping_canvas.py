from selenium import webdriver
from selenium.webdriver.common.by import By

# Set up the Selenium webdriver
driver = webdriver.Chrome()

symbol = 'TSLA'  #symbol, company or stock to analyze

#This is the web with images for scraping canvas
driver.get(f"https://stockanalysis.com/stocks/{symbol}/forecast/")

driver.implicitly_wait(10) # wait for up to 10 seconds for the page to load

# Find the canvas element, here we use find_element of selenium with css_selector and class name that contains canvas.
canvas1 = driver.find_element(By.CSS_SELECTOR, 'div[data-test="forecast-price-chart"] canvas') 
canvas2 = driver.find_element(By.CSS_SELECTOR, 'div[data-test="forecast-estimategrowth-charts"] canvas')
canvas3 = driver.find_element(By.CSS_SELECTOR, 'div[data-test="forecast-estimate-charts"] canvas')

# Set the background color to white
driver.execute_script("arguments[0].style.backgroundColor = 'white';", canvas1)
driver.execute_script("arguments[0].style.backgroundColor = 'white';", canvas2)
driver.execute_script("arguments[0].style.backgroundColor = 'white';", canvas3)

# Get the base64-encoded JPG image of the canvas
canvas_base64_1 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);", canvas1)
canvas_base64_2 = driver.execute_script("return arguments[0].toDataURL('image/jpg').substring(23);", canvas2)
canvas_base64_3 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);", canvas3)


# here we decode the base64-encoded image and save it to a file
import base64
with open("graph1.png", "wb") as f:
    f.write(base64.b64decode(canvas_base64_1))

with open("graph2.png", "wb") as f:
    f.write(base64.b64decode(canvas_base64_2))

with open("graph3.png", "wb") as f:
    f.write(base64.b64decode(canvas_base64_3))


# Close the browser
driver.quit()