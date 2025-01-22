# Network Analysis Project

This project analyzes network datasets and implements the Albert-Barabási model for comparison.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Create a `data` directory in the project root
2. Place your network datasets in the `data` directory with names:
   - `network1.txt`
   - `network2.txt`
   - `network3.txt`

Each dataset should be an edge list file with two columns (source node, target node).
Node IDs should be integers starting from 0.

## Running the Analysis

Run the analysis script:
```bash
python src/network_analysis.py
```

The script will:
1. Analyze the three input networks
2. Generate corresponding Barabási-Albert networks
3. Create plots and statistics for all networks

Results will be saved in the `results` directory. 