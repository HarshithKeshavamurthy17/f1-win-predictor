# Streamlit Keep-Alive Setup Instructions

This document provides step-by-step instructions for setting up the Selenium-based keep-alive mechanism for your Streamlit application.

## Overview

The keep-alive system consists of:
- **Python Script** ([scripts/keep_alive.py](scripts/keep_alive.py)): Uses Selenium to visit your Streamlit app and click the wake-up button if needed
- **GitHub Actions Workflow** ([.github/workflows/keep_alive.yml](.github/workflows/keep_alive.yml)): Runs the script every 15 minutes automatically

## Setup Instructions

### 1. Add the STREAMLIT_APP_URL Secret

You need to add your Streamlit app URL as a secret in your GitHub repository:

1. Go to your GitHub repository: `https://github.com/YOUR_USERNAME/YOUR_REPO`
2. Click on **Settings** (top menu)
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click the **New repository secret** button
5. Fill in the details:
   - **Name**: `STREAMLIT_APP_URL`
   - **Secret**: Your Streamlit app URL (e.g., `https://your-app.streamlit.app`)
6. Click **Add secret**

### 2. Push the Code to GitHub

Commit and push the new files to your repository:

```bash
# Add the new files
git add scripts/keep_alive.py
git add .github/workflows/keep_alive.yml
git add KEEP_ALIVE_SETUP.md

# Commit the changes
git commit -m "Add Selenium-based keep-alive for Streamlit app"

# Push to GitHub
git push origin main
```

> **Note**: Replace `main` with your default branch name if different (e.g., `master`).

### 3. Verify the Workflow

After pushing, verify that the workflow is set up correctly:

1. Go to your GitHub repository
2. Click on the **Actions** tab
3. You should see "Streamlit Keep-Alive" in the workflows list
4. The workflow will run automatically every 15 minutes, or you can trigger it manually:
   - Click on "Streamlit Keep-Alive"
   - Click **Run workflow** button
   - Select the branch and click **Run workflow**

### 4. Monitor the Workflow

To check if the workflow is running successfully:

1. Go to the **Actions** tab in your repository
2. Click on a workflow run to see the details
3. Check the logs for each step
4. Look for messages like:
   - `✓ Wake-up button found!` (if app was sleeping)
   - `✓ App is already awake` (if app was running)
   - `✓ Keep-alive check completed successfully!`

## How It Works

### Script Behavior

The `keep_alive.py` script:
1. Sets up a headless Chrome browser using Selenium
2. Visits your Streamlit app URL
3. Waits for the page to load
4. Checks for the "Yes, get this app back up!" button using multiple detection strategies
5. Clicks the button if found (waking up the app)
6. Reports success if the app is already awake or was successfully woken up

### Workflow Schedule

The GitHub Actions workflow:
- Runs automatically every 15 minutes (`cron: "*/15 * * * *"`)
- Can be triggered manually via the GitHub Actions UI
- Uses Ubuntu with Python 3.9 and Chrome browser
- Passes the `STREAMLIT_APP_URL` secret to the script

## Troubleshooting

### Workflow Not Running

- **Check the Actions tab**: Ensure workflows are enabled for your repository
- **Verify the secret**: Make sure `STREAMLIT_APP_URL` is set correctly in repository secrets
- **Check branch**: Ensure the workflow file is on your default branch

### Script Failing

Check the workflow logs for error messages:
- **"STREAMLIT_APP_URL environment variable not set"**: The secret is not configured properly
- **Timeout errors**: The app might be taking too long to load; the script waits up to 60 seconds
- **Button not found**: The script will still report success if the app is already awake

### Testing Locally

You can test the script locally (requires Chrome installed):

```bash
# Install dependencies
pip install selenium webdriver-manager

# Set the environment variable and run
export STREAMLIT_APP_URL="https://your-app.streamlit.app"
python scripts/keep_alive.py
```

## Customization

### Change the Schedule

To run more or less frequently, edit the cron expression in [.github/workflows/keep_alive.yml](.github/workflows/keep_alive.yml):

```yaml
schedule:
  - cron: "*/15 * * * *"  # Every 15 minutes
  # - cron: "*/10 * * * *"  # Every 10 minutes
  # - cron: "*/30 * * * *"  # Every 30 minutes
  # - cron: "0 * * * *"     # Every hour
```

### Adjust Timeouts

Edit the timeout values in `scripts/keep_alive.py`:

```python
driver.set_page_load_timeout(60)  # Page load timeout (seconds)
time.sleep(3)                      # Initial wait after page load
time.sleep(5)                      # Wait after clicking button
```

## Notes

- GitHub Actions has a limit on workflow runs; running every 15 minutes should be well within limits
- The script uses a headless browser, so it won't open any visible windows
- Multiple detection strategies ensure the button is found even if Streamlit changes its UI slightly
- The workflow will continue running even if individual runs fail

## Support

If you encounter issues:
1. Check the workflow logs in the Actions tab
2. Verify your Streamlit app URL is correct and accessible
3. Ensure the secret is set properly in repository settings
4. Test the script locally to isolate GitHub Actions issues
