[Home](../README.md)
# Data Processing

## Data Extraction

Activate the virtual environment for your OS.  For example on Mac run the following
command:
`source .venv/bin/activate`

Change to the data directory:
`cd /data`

Ensure that you have a Firecrawl API key stored in a file named `.env' in the
following format:


`
FIRECRAWL_API_KEY=<YOUR_FIRECRAWL_API_KEY>
`

Run the following command to execute the data scraper script:

`python3 data_scraper.py`

## Post Processing - optional

The data that is downloaded is approximately 200 mb, to review the data it needs to be formatted, otherwise it will all be on a single line.

Run the following command from your terminal:

`cat scout_information.json | python -m json.tool > pretty_scout_information.json`

After reviewing the data for completeness delete the pretty_scout_information.json file as it is not needed for processing.
