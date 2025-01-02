# Sql-analysis# SQL Query Analysis Platform

A Streamlit application that allows users to analyze data using natural language queries, which are automatically converted to SQL. The application uses Google's Gemini AI to generate SQL queries and provide insights from the data.

## Features

- Upload Excel/CSV files for analysis
- Natural language to SQL query conversion
- Automatic topic extraction from datasets
- Dynamic topic matching and analysis
- Interactive query results with explanations
- Results export functionality

## Setup

1. Clone this repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Environment Variables

Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Upload your Excel/CSV file
2. Enter your analysis question in natural language
3. View the generated SQL query and results
4. Export results if needed

## Deployment

This application is deployed on Streamlit Community Cloud.

## License

MIT License