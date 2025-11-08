# Default Credit Card of Clients

This repository contains a small classification pipeline that trains a logistic regression model to predict default of credit card clients.

Quick start

1. Create a virtual environment (recommended):

   Windows (cmd.exe):
   ```cmd
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Configure environment variables: create a local `.env` in the project root with these values (do NOT commit it):

   ```properties
   DB_USER=root
   DB_PASSWORD=your_db_password_here
   DB_HOST=127.0.0.1
   DB_PORT=3306
   DB_NAME=creditCard_db
   # Optional: CSV_PATH to override the default CSV location
   # CSV_PATH=path\to\default of credit card clients.csv
   ```

   A `.env.sample` is provided for reference.

3. Run the pipeline:

   ```cmd
   python credit_Card.py
   ```

Outputs

- `outputs/` will contain generated artifacts: model `.joblib` files, `model_report.json`, CSVs and `manifest.json`.

Notes

- `.env` is excluded from git via `.gitignore` to avoid leaking secrets. Use `.env.sample` to create your local `.env`.
- If you want `load_dotenv()` to automatically load `.env`, ensure `python-dotenv` is installed (it's included in `requirements.txt`).

License

- No license specified. Add a LICENSE if you plan to share this repo publicly.
