## Project Setup and Reproducibility

This project uses both local data files and MySQL as part of the data preparation workflow.

The raw datasets are stored in the `data/raw` folder. During the data preparation stage, these files are cleaned, merged and transformed into final analytical datasets. The processed outputs are saved in the `data/processed` folder and are also written to a local MySQL database for structured storage and querying.

The project uses a `.env` file to store local database credentials and API keys. This file is intentionally excluded from GitHub for security reasons.

To run the full project from the beginning:

1. Download or clone the GitHub repository.
2. Create a local `.env` file.
3. Create a MySQL database called `ca2_agriculture`.
4. Run the data preparation notebook first. This reads the raw data, creates processed datasets, saves them to `data/processed`, and writes the final tables to MySQL.
5. Run the statistics, machine learning, sentiment analysis and dashboard notebooks after the data preparation step.

The processed CSV files are also included in the repository so that the final analytical datasets can be inspected without requiring the database to be recreated manually.

## Dashboard

The interactive farmer-focused dashboard was developed using Plotly Dash. It presents export performance, product demand, country benchmarking, sentiment signals and farmer-focused recommendations.

Live dashboard link: To be added after deployment.

The dashboard can also be run locally from the dashboard notebook or deployment `app.py` file.
