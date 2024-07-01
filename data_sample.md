
The following samples are produced from the jupyter notebook in: `src/notebooks/view_data_preprocessing.ipynb`. To run the notebook, please download the data first.

### SE Indirect 1D

```
{
    'input': '+92% ',
    'output': 'property_value',
    'input_type': 'text',
    'augmentation_status': 'original'
}
```

### SE Indirect 2D

For better viewing, the newline token from the data has been replaces with actual newline tokens.

Input:

```
| 0                     | 1       | 2       | 3        |
|-----------------------|---------|---------|----------|
|                       | FY2021  | FY2022  | % change |
| Scope 2 emissions     | 85,341  | 85,050  | -0.3%    |
| Scope 1 emissions     | 27,716  | 28,500  | 2.8%     |
| Scope 1 & 2 emissions | 113,057 | 113,550 | 0.4%     |
```

Output:

```
| 0        | 1              | 2              | 3              |
|----------|----------------|----------------|----------------|
| empty    | time_value     | time_value     | unit_value     |
| property | property_value | property_value | property_value |
| property | property_value | property_value | property_value |
| property | property_value | property_value | property_value |
```


### SE Direct

For better viewing, the newline token and the sep from the data has been replaced with actual newline tokens.

Input:

```
| 0                                                               | 1       | 2       | 3            |
|-----------------------------------------------------------------|---------|---------|--------------|
| Australia - Scope 1                                             | 2021    | 2020    | 2022         |
| Emissions from fuel consumption (tCO$_{2}$e)                    | 17,957  | 14,574  | 19,561       |
| Emissions from Killara Feedlot cattle (tCO$_{2}$e)              | 37,462  | -       | 44,826       |
| Fuel consumption (GJ)                                           | 255,873 | 207,569 | 278,969      |
| Australia - Scope 2                                             |         |         |              |
| Electricity consumption from the grid (GJ)                      | 24,201  | 41,051  | 29,104       |
| Total renewable electricity (MWh)                               | -       | -       | 8,084 (100%) |
| Voluntary LGCs                                                  | -       | -       | 6,581        |
| Electricity consumption from the grid (MWh)                     | 6,722   | 11,403  | 8,084        |
| Mandatory LGCs 1                                                | -       | -       | 1,503        |
| Emissions from purchased electricity (tCO$_{2}$e) 2             | 4,982   | 8,946   | 5,801        |
| Total energy (Australia)                                        |         |         |              |
| Total energy consumption (GJ)                                   | 280,074 |         | 308,073      |
| China - Scope 2                                                 |         |         |              |
| Electricity consumption from the grid (GJ)                      | 2,466   | -       | 2,580        |
| Scope 2 GHG emissions from electricity consumption (tCO$_{2}$e) | 427     | -       | 385          |
```

Output (excerpt):

```
| property                                                           | property_value   | unit   | subject   | subject_value   |
|--------------------------------------------------------------------|------------------|--------|-----------|-----------------|
| Australia - Scope 1 : Emissions from fuel consumption (tCO$_{2}$e) | 17,957           |        |           |                 |
| time                                                               | 2021             |        |           |                 |

| property                                                           | property_value   | unit   | subject   | subject_value   |
|--------------------------------------------------------------------|------------------|--------|-----------|-----------------|
| Australia - Scope 1 : Emissions from fuel consumption (tCO$_{2}$e) | 14,574           |        |           |                 |
| time                                                               | 2020             |        |           |                 |

| property                                                           | property_value   | unit   | subject   | subject_value   |
|--------------------------------------------------------------------|------------------|--------|-----------|-----------------|
| Australia - Scope 1 : Emissions from fuel consumption (tCO$_{2}$e) | 19,561           |        |           |                 |
| time                                                               | 2022             |        |           |                 |
```