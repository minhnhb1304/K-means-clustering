## ⚙️ Installation & Setup

Follow these steps to set up and run the project locally.

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system. You can check this by running:
```bash
python --version
```
### 2. Create a Virtual Environment (Optional)
This step is recommended to isolate dependencies. **If you already have a Python environment set up or prefer global installation , you can skip this step.**

For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

For macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
Install all required libraries listed in requirements.txt:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Launch the Streamlit web app:
```bash
streamlit run app.py
```
After running this command, the app should automatically open in your default web browser at http://localhost:8501.
If this command not working, you can try `python -m streamlit run app.py` instead.
