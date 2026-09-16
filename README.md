# YuvaIntern Week 1 – Logistics Data Cleaning & Transformation

## Overview

This project was completed as part of my YuvaIntern internship as a
Junior Data Analyst – Logistics.

The project focuses on cleaning, validating, transforming and categorizing
a logistics delivery dataset using Python and Pandas.

## Objectives

- Inspect logistics data quality
- Identify missing values and duplicates
- Investigate inconsistent data representations
- Transform timestamp-like delivery time fields
- Investigate anomalies and outliers
- Validate categorical and numerical data
- Create distance-based categories
- Produce a cleaned dataset for further analysis

## Dataset

The dataset contains 25,000 logistics delivery records and includes
information about:

- Delivery partners
- Package types
- Vehicle types
- Delivery modes
- Regions
- Weather conditions
- Distance
- Package weight
- Delivery time
- Expected delivery time
- Delivery status
- Ratings
- Delivery cost

## Tools

- Python
- Pandas
- NumPy
- Jupyter Notebook

## Key Data Quality Findings

- 25,000 records analyzed
- 0 missing values
- 0 exact duplicate rows
- 0 negative numerical values
- Timestamp-like hour fields transformed into integer hours
- Potential delivery-time outliers investigated and retained when plausible
- Source delivery status fields validated for internal consistency
- Distance categories created: Short, Medium and Long

## Distance Categorization

| Category | Distance | Records |
|---|---|---:|
| Short | <= 100 km | 8,317 |
| Medium | 101–200 km | 8,415 |
| Long | > 200 km | 8,268 |

## Repository Structure

```text
data/
notebook/
report/
README.md
