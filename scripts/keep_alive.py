#!/usr/bin/env python3
"""
Streamlit Keep-Alive Script
This script uses Selenium to visit a Streamlit app and wake it up if it's sleeping.
"""

import os
import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def setup_driver():
    """Set up and return a headless Chrome WebDriver."""
    print("Setting up Chrome WebDriver...")
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(60)
    
    print("Chrome WebDriver setup complete.")
    return driver


def check_and_wake_app(url):
    """
    Visit the Streamlit app URL and wake it up if it's sleeping.
    
    Args:
        url (str): The Streamlit app URL to visit
        
    Returns:
        bool: True if successful, False otherwise
    """
    driver = None
    
    try:
        driver = setup_driver()
        
        print(f"Visiting URL: {url}")
        driver.get(url)
        
        # Wait a moment for the page to load
        time.sleep(3)
        
        print("Page loaded successfully.")
        
        # Check if the wake-up button is present
        try:
            # Look for the "Yes, get this app back up!" button
            # Streamlit uses different possible selectors for this button
            wait = WebDriverWait(driver, 10)
            
            # Try multiple strategies to find the button
            button_found = False
            
            # Strategy 1: Look for button with exact text
            try:
                button = wait.until(
                    EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Yes, get this app back up!')]"))
                )
                button_found = True
                print("✓ Wake-up button found (exact text match)!")
            except:
                pass
            
            # Strategy 2: Look for button with partial text match
            if not button_found:
                try:
                    button = driver.find_element(By.XPATH, "//button[contains(text(), 'get this app back up')]")
                    button_found = True
                    print("✓ Wake-up button found (partial text match)!")
                except:
                    pass
            
            # Strategy 3: Look for any button containing "back up"
            if not button_found:
                try:
                    button = driver.find_element(By.XPATH, "//button[contains(., 'back up')]")
                    button_found = True
                    print("✓ Wake-up button found (generic match)!")
                except:
                    pass
            
            if button_found:
                # Click the button
                print("Clicking the wake-up button...")
                button.click()
                
                # Wait for the app to start loading
                time.sleep(5)
                
                print("✓ Button clicked successfully! App is waking up...")
                return True
            else:
                print("✓ App is already awake (no wake-up button found).")
                return True
                
        except Exception as e:
            # If button not found, the app is likely already awake
            print("✓ App is already awake (no wake-up button detected).")
            return True
            
    except Exception as e:
        print(f"✗ Error occurred: {str(e)}", file=sys.stderr)
        return False
        
    finally:
        if driver:
            print("Closing browser...")
            driver.quit()
            print("Browser closed.")


def main():
    """Main function to run the keep-alive script."""
    print("=" * 60)
    print("Streamlit Keep-Alive Script")
    print("=" * 60)
    
    # Get the Streamlit app URL from environment variable
    app_url = os.getenv("STREAMLIT_APP_URL")
    
    if not app_url:
        print("✗ Error: STREAMLIT_APP_URL environment variable not set!", file=sys.stderr)
        sys.exit(1)
    
    print(f"Target URL: {app_url}")
    print("-" * 60)
    
    # Check and wake the app
    success = check_and_wake_app(app_url)
    
    print("-" * 60)
    if success:
        print("✓ Keep-alive check completed successfully!")
        sys.exit(0)
    else:
        print("✗ Keep-alive check failed!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
