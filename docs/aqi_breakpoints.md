# AQI Breakpoints — Nepal DoE Standard

Nepal Department of Environment AQI standard follows the US EPA 6-category structure.
These breakpoints are used in `code/01_data_prep/label_aqi.py`.

| Class | Label | PM2.5 (µg/m³) | AQI Range | Health Advisory |
|---|---|---|---|---|
| 0 | Good | 0.0 – 12.0 | 0 – 50 | Air quality is satisfactory |
| 1 | Moderate | 12.1 – 35.4 | 51 – 100 | Unusually sensitive people should consider reducing prolonged outdoor exertion |
| 2 | Unhealthy for Sensitive Groups | 35.5 – 55.4 | 101 – 150 | Sensitive groups should reduce prolonged outdoor exertion |
| 3 | Unhealthy | 55.5 – 150.4 | 151 – 200 | Everyone should reduce prolonged outdoor exertion |
| 4 | Very Unhealthy | 150.5 – 250.4 | 201 – 300 | Everyone should avoid prolonged outdoor exertion |
| 5 | Hazardous | 250.5+ | 301 – 500 | Everyone should avoid all outdoor exertion |

## Source
Nepal Department of Environment: https://pollution.gov.np
US EPA AQI Basics: https://www.airnow.gov/aqi/aqi-basics/
