# Kerr-Salpeter Cosmological Model: Pulsar Timing Dashboard

This repository contains the interactive analysis tool associated with the manuscript: **"The Kerr-Salpeter Cosmological Model: Dark Energy as Centrifugal Pressure and Geometric Resolution of the Hubble Tension"**.

## Overview

This Streamlit application is designed to search for specific harmonic signatures within pulsar timing data, particularly the NANOGrav 15-year dataset. The physics engine looks for:
1.  **The Geometric Factor ($\Gamma \approx 1.0833$)**: Representing the global frame-dragging effect.
2.  **The Coupling Factor ($\kappa \approx 30.6$)**: Derived from the Universe's age and the Salpeter timescale.

The application computes the Lomb-Scargle periodogram of the timing residuals, detects peaks, and evaluates the statistical significance of harmonic chains using robust Monte Carlo permutation tests.

## Repository Structure

* `app.py`: The main Streamlit dashboard application.
* `requirements.txt`: List of Python dependencies.
* `data/`: Directory where the `.res` (residuals) files should be placed. *(Note: Due to file size limits, NANOGrav data is not included in this repository. See instructions below).*

## Installation and Setup

It is highly recommended to use a Python virtual environment (Python 3.8+).

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/TokyoOnAcid/Kerr-Salpeter-Cosmology.git](https://github.com/TokyoOnAcid/Kerr-Salpeter-Cosmology.git)
    cd Kerr-Salpeter-Cosmology
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Acquisition (NANOGrav 15yr)

To reproduce the findings, please download the public NANOGrav 15-year Pulsar Timing dataset:
1. Download the residuals from the official NANOGrav site or their Zenodo repository.
2. Place the `.res` files inside a `data/` folder at the root of this repository.

## Running the Dashboard
