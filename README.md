# Sudoku Solver – Constraint Satisfaction Performance Analysis

## Overview
This project was developed as part of my final-year dissertation and investigates Sudoku as a benchmark problem for constraint satisfaction problems (CSPs). The project compares different solving approaches to evaluate how performance changes with puzzle size and difficulty.

## Solving Approaches
- Backtracking with Minimum Remaining Values (MRV) heuristic
- SAT-based solving using PySAT (Glucose3)
- Constraint Programming using MiniZinc

## Experimental Scope
The solvers were evaluated across:
- Puzzle sizes: 9×9, 16×16, and 25×25
- Difficulty levels: easy, medium, hard, and extreme

Performance metrics included solving time and success rate.

## Technologies Used
- Python
- PySAT (Glucose3)
- MiniZinc
- Streamlit
- Constraint Satisfaction Problems (CSP)

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Run the application:
   streamlit run app.py

