import streamlit as st
import subprocess
import tempfile
import json
import time, math, copy, os, re
import random 
import pandas as pd
import shutil
from pysat.solvers import Glucose3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from datetime import datetime

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ——————————————————————————————————————————————
# 1) Cleaned MiniZinc model with only ASCII hyphens
# ——————————————————————————————————————————————
minizinc_model = r"""
%----------------------
% 1) Inline alldifferent
%----------------------
predicate alldifferent(array[int] of var int: a) =
    forall(i in index_set(a), j in index_set(a) where i < j) (
        a[i] != a[j]
    );

%----------------------
% 2) Parameters & variables
%----------------------
int: n;
set of int: NUM = 1..n;
array[1..n,1..n] of var NUM: x;
array[1..n,1..n] of int: clue;

%----------------------
% 3) Pre‑filled clues
%----------------------
constraint
  forall(i in 1..n, j in 1..n where clue[i,j] != 0) (
    x[i,j] = clue[i,j]
  );

%----------------------
% 4) Row/col all‑different
%----------------------
constraint forall(i in 1..n) (
  alldifferent([ x[i,j] | j in 1..n ])
);
constraint forall(j in 1..n) (
  alldifferent([ x[i,j] | i in 1..n ])
);

%----------------------
% 5) Block all‑different
%----------------------
int: block = round(sqrt(n));
constraint forall(br in 0..block-1, bc in 0..block-1) (
  alldifferent([
    x[br*block + di, bc*block + dj]
    | di in 1..block, dj in 1..block
  ])
);

%----------------------
% 6) Redundant SUM constraints
%    (speeds up propagation)
%----------------------
int: target = n*(n+1) div 2;
constraint forall(i in 1..n)(
  sum(j in 1..n)(x[i,j]) = target
);
constraint forall(j in 1..n)(
  sum(i in 1..n)(x[i,j]) = target
);
constraint forall(br in 0..block-1, bc in 0..block-1)(
  sum(di in 1..block, dj in 1..block)(
    x[br*block + di, bc*block + dj]
  ) = target
);

%----------------------
% 7) Search annotation
%    dom_w_deg = domain‑over‑weighted‑degree (strong MRV)
%    indomain_min = try smallest value first
%----------------------
array[1..n*n] of var NUM: VS = [ x[i,j] | i in 1..n, j in 1..n ];
solve :: 
    int_search(VS, dom_w_deg, indomain_min, complete)
    satisfy;
"""

# ——————————————————————————————————————————————
# 2) Updated solve_with_minizinc with UTF-8 writes
# ——————————————————————————————————————————————
def solve_with_minizinc(board, minizinc_cmd="minizinc"):
    n = len(board)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "sudoku.mzn")
        data_path  = os.path.join(tmpdir, "sudoku.dzn")

        # Write model & data in UTF-8
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(minizinc_model)
        flat = [str(cell) for row in board for cell in row]
        dzn_lines = [
            f"n = {n};",
            f"clue = array2d(1..{n},1..{n},[{','.join(flat)}]);"
        ]
        with open(data_path, "w", encoding="utf-8") as f:
            f.write("\n".join(dzn_lines))

        # Invoke MiniZinc
        cmd = [
            minizinc_cmd,
            "--solver", "Chuffed",
            "--time-limit", "60000",
            model_path, data_path
        ]
        start = time.time()
        proc  = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start

        # If it failed or is unsatisfiable, bail out
        if proc.returncode != 0 or "UNSATISFIABLE" in proc.stdout:
            return None, None

        # Parse the [| … |]; block
        m = re.search(r"\[\|([\s\S]*?)\|\];", proc.stdout)
        if not m:
            return None, None

        body = m.group(1).replace("|", "").replace("\n", ",")
        tokens = [tok.strip() for tok in body.split(",") if tok.strip()]
        nums = list(map(int, tokens))

        # Reconstruct 2D solution
        solution = [nums[i*n:(i+1)*n] for i in range(n)]
        return solution, elapsed

# ---------------------------
# Heuristic-Driven Backtracking Solver Functions
# ---------------------------

def get_candidates(board, row, col, n):
    """
    Given the current board state and a position (row, col), this function returns a set of
    candidate numbers (1 to n) that can be legally assigned to that cell.
    It does so by removing numbers already present in the same row, column, and sub-grid.
    """
    # Start with all possible digits 1 to n.
    candidates = set(range(1, n+1))
    # Remove digits present in the current row.
    candidates -= set(board[row])
    # Remove digits present in the current column.
    candidates -= {board[i][col] for i in range(n)}
    
    # Calculate sub-grid (block) dimensions.
    block_size = int(math.sqrt(n))
    start_row = (row // block_size) * block_size
    start_col = (col // block_size) * block_size
    # Remove digits present in the current sub-grid.
    for i in range(start_row, start_row + block_size):
        for j in range(start_col, start_col + block_size):
            candidates.discard(board[i][j])
    
    return candidates

def find_empty_with_MRV(board, n):
    """
    Find the next empty cell (cell with value 0) using the Minimum Remaining Values (MRV) heuristic.
    The function returns a tuple (row, col, candidates) where 'candidates' is the set of possible digits
    that can be placed in that cell.
    If no empty cell is found, returns None.
    """
    min_candidates = None
    best_cell = None
    # Iterate through all cells in the board.
    for i in range(n):
        for j in range(n):
            if board[i][j] == 0:
                # Determine legal candidates for this cell.
                candidates = get_candidates(board, i, j, n)
                # If no candidate is available, this branch is unsolvable.
                if not candidates:
                    return (i, j, set())
                # Update best_cell if this cell has fewer options.
                if min_candidates is None or len(candidates) < len(min_candidates):
                    min_candidates = candidates
                    best_cell = (i, j)
                    # If only one candidate is available, return immediately.
                    if len(min_candidates) == 1:
                        return (best_cell[0], best_cell[1], min_candidates)
    # Return the best candidate found, if any.
    return (best_cell[0], best_cell[1], min_candidates) if best_cell else None

def backtracking_solve(board, n):
    """
    Attempt to solve the Sudoku puzzle using heuristic-driven backtracking.
    Returns True if a solution is found, else False.
    """
    pos = find_empty_with_MRV(board, n)
    # If no empty cell is found, the puzzle is solved.
    if pos is None:
        return True  
    row, col, candidates = pos
    # Dead end when no candidate is available.
    if not candidates:
        return False  
    # Try each candidate value for the cell.
    for candidate in candidates:
        board[row][col] = candidate
        if backtracking_solve(board, n):
            return True
        board[row][col] = 0  # Reset the cell on backtracking
    return False

def solve_with_backtracking(board):
    """
    Wrapper function to solve the Sudoku using backtracking.
    Makes a deep copy of the board to preserve the original.
    Measures and returns the elapsed solving time.
    """
    n = len(board)
    board_copy = copy.deepcopy(board)
    start_time = time.time()
    if backtracking_solve(board_copy, n):
        elapsed_time = time.time() - start_time
        return board_copy, elapsed_time
    else:
        return None, None

# ---------------------------
# SAT Solver Functions using PySAT
# ---------------------------

def varnum(i, j, d, n):
    """
    Map a cell at (i, j) with digit d (0-indexed) to a unique SAT variable.
    Variables are numbered starting from 1.
    """
    return i * n * n + j * n + d + 1

def exactly_one(vars_list):
    """
    Given a list of SAT variables, returns CNF clauses that enforce exactly one variable is True.
    This is done via:
        - One clause ensuring at least one is True.
        - Pairwise clauses ensuring no two are True simultaneously.
    """
    clauses = [vars_list.copy()]  # At least one must be true
    for i in range(len(vars_list)):
        for j in range(i+1, len(vars_list)):
            clauses.append([-vars_list[i], -vars_list[j]])
    return clauses

def encode_sudoku(board, n):
    """
    Encode the entire sudoku puzzle as a list of CNF clauses.
    This includes:
        - Each cell must have exactly one digit.
        - Each row, column, and block must contain each digit exactly once.
        - Pre-filled clues are encoded as unit clauses.
    """
    clauses = []
    block_size = int(math.sqrt(n))
    # Encode cell constraints.
    for i in range(n):
        for j in range(n):
            cell_vars = [varnum(i, j, d, n) for d in range(n)]
            clauses.extend(exactly_one(cell_vars))
    # Encode row constraints.
    for i in range(n):
        for d in range(n):
            row_vars = [varnum(i, j, d, n) for j in range(n)]
            clauses.extend(exactly_one(row_vars))
    # Encode column constraints.
    for j in range(n):
        for d in range(n):
            col_vars = [varnum(i, j, d, n) for i in range(n)]
            clauses.extend(exactly_one(col_vars))
    # Encode block constraints.
    for block_row in range(0, n, block_size):
        for block_col in range(0, n, block_size):
            for d in range(n):
                block_vars = []
                for i in range(block_row, block_row + block_size):
                    for j in range(block_col, block_col + block_size):
                        block_vars.append(varnum(i, j, d, n))
                clauses.extend(exactly_one(block_vars))
    # Encode pre-filled clues as unit clauses.
    for i in range(n):
        for j in range(n):
            if board[i][j] != 0:
                d = board[i][j] - 1
                clauses.append([varnum(i, j, d, n)])
    return clauses

def solve_with_sat(board):
    """
    Solves the Sudoku puzzle using a SAT solver (Glucose3 from PySAT).
    Returns the solved board (if solvable) along with the elapsed solving time.
    """
    n = len(board)
    board_copy = copy.deepcopy(board)
    # Encode the board as CNF clauses.
    clauses = encode_sudoku(board_copy, n)
    solver = Glucose3()
    # Add all clauses to the solver.
    for clause in clauses:
        solver.add_clause(clause)
    
    start_time = time.time()
    solvable = solver.solve()
    elapsed_time = time.time() - start_time

    if not solvable:
        solver.delete()
        return None, None
    # Get the satisfying model and convert it back to the Sudoku board.
    model = solver.get_model()
    solver.delete()
    solution = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for d in range(n):
                v = varnum(i, j, d, n)
                if v in model:
                    solution[i][j] = d + 1  # Adjust back from 0-index
                    break
    return solution, elapsed_time

# ---------------------------
# Random Puzzle Generation Functions (Updated)
# ---------------------------

def generate_complete_board(n):
    """
    Generate a valid completed Sudoku board of size n×n (n must be a perfect square).
    Uses a standard pattern and then applies random transformations.
    """
    # Create initial pattern using the standard formula
    base = int(math.sqrt(n))
    side = n
    board = [
        [((r % base) * base + r // base + c) % side + 1
         for c in range(side)]
        for r in range(side)
    ]
    
    # Apply random transformations to create a truly random board
    board = apply_random_transformations(board, n)
    return board

def apply_random_transformations(board, n):
    """
    Apply a series of valid transformations to create a random but valid Sudoku board.
    Valid transformations preserve the Sudoku constraints.
    """
    base = int(math.sqrt(n))
    result = copy.deepcopy(board)
    
    # 1. Randomly permute the digits (1-n)
    digit_permutation = list(range(1, n+1))
    random.shuffle(digit_permutation)
    digit_map = {i: digit_permutation[i-1] for i in range(1, n+1)}
    
    for i in range(n):
        for j in range(n):
            result[i][j] = digit_map[result[i][j]]
    
    # 2. Randomly swap rows within each block row
    for block_row in range(base):
        for _ in range(base):  # Multiple swaps for more randomness
            # Select two rows within this block row
            r1 = block_row * base + random.randrange(base)
            r2 = block_row * base + random.randrange(base)
            if r1 != r2:
                result[r1], result[r2] = result[r2], result[r1]
    
    # 3. Randomly swap columns within each block column
    for block_col in range(base):
        for _ in range(base):  # Multiple swaps for more randomness
            # Select two columns within this block column
            c1 = block_col * base + random.randrange(base)
            c2 = block_col * base + random.randrange(base)
            if c1 != c2:
                for i in range(n):
                    result[i][c1], result[i][c2] = result[i][c2], result[i][c1]
    
    # 4. Randomly swap block rows
    for _ in range(base):  # Multiple swaps for more randomness
        br1 = random.randrange(base)
        br2 = random.randrange(base)
        if br1 != br2:
            # Swap all rows in these block rows
            for r in range(base):
                r1 = br1 * base + r
                r2 = br2 * base + r
                result[r1], result[r2] = result[r2], result[r1]
    
    # 5. Randomly swap block columns
    for _ in range(base):  # Multiple swaps for more randomness
        bc1 = random.randrange(base)
        bc2 = random.randrange(base)
        if bc1 != bc2:
            # Swap all columns in these block columns
            for c in range(base):
                c1 = bc1 * base + c
                c2 = bc2 * base + c
                for i in range(n):
                    result[i][c1], result[i][c2] = result[i][c2], result[i][c1]
    
    # 6. Random board transposition (with 50% probability)
    if random.random() < 0.5:
        result = [[result[j][i] for j in range(n)] for i in range(n)]
    
    return result

def count_solutions(board, n, limit=2):
    """
    Use MRV backtracking to count up to `limit` solutions.
    Returns the number found (1 = unique, ≥2 = multiple).
    """
    count = 0

    def solve(b):
        nonlocal count
        if count >= limit:
            return
        pos = find_empty_with_MRV(b, n)
        if pos is None:
            count += 1
            return
        r, c, cands = pos
        for d in cands:
            b[r][c] = d
            solve(b)
            b[r][c] = 0
            if count >= limit:
                return

    solve(copy.deepcopy(board))
    return count

def generate_puzzle(n, difficulty):
    """
    Generate a random Sudoku puzzle by first creating a random complete board,
    then removing cells according to the desired difficulty while ensuring uniqueness.
    """
    # Define removal percentages for different difficulties
    removal_percentages = {
        "easy":    0.40,  # Remove 40% of cells
        "medium":  0.50,  # Remove 50% of cells
        "hard":    0.60,  # Remove 60% of cells
        "extreme": 0.70,  # Remove 70% of cells
    }
    
    if difficulty not in removal_percentages:
        raise ValueError(f"Unknown difficulty {difficulty!r}")
    
    # Generate a random complete board
    board = generate_complete_board(n)
    
    # Create a list of all cell positions and shuffle it randomly
    cells = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(cells)
    
    # Calculate how many cells to remove
    cells_to_remove = int(n * n * removal_percentages[difficulty])
    removed = 0
    
    # Try to remove cells while maintaining uniqueness
    for r, c in cells:
        if removed >= cells_to_remove:
            break
        
        backup = board[r][c]
        board[r][c] = 0
        
        # For 9×9 puzzles, ensure uniqueness
        # For larger puzzles, we might skip the uniqueness check for performance
        # or use a faster method to estimate uniqueness
        if n <= 9 and count_solutions(board, n) != 1:
            board[r][c] = backup  # Restore cell if removing it creates multiple solutions
        else:
            removed += 1
    
    return board

def display_board(board, solution=None):
    n     = len(board)
    block = int(math.sqrt(n))

    container_style = (
        "width:80vw; max-width:600px; aspect-ratio:1/1; margin:auto;"
        "overflow:hidden;"
    )
    table_style = (
        "width:100%; height:100%; border-collapse:collapse;"
        "border-spacing:0; background-color:#0d1117;"
    )
    html = f'<div style="{container_style}"><table style="{table_style}">'

    for i in range(n):
        html += "<tr>"
        for j in range(n):
            orig = board[i][j]
            val  = orig
            # default text color: white
            txt_color = "#fff"

            # if we have a solution, and this cell was empty but now filled:
            if solution is not None and orig == 0 and solution[i][j] != 0:
                val       = solution[i][j]
                txt_color = "#00FF00"  # green text

            # build borders exactly as before
            borders = []
            if i == 0:           borders.append("border-top:3px solid #fff")
            if j == 0:           borders.append("border-left:3px solid #fff")
            if i == n - 1:       borders.append("border-bottom:3px solid #fff")
            if j == n - 1:       borders.append("border-right:3px solid #fff")
            if (i + 1) % block == 0 and i != n - 1:
                borders.append("border-bottom:2px solid #fff")
            if (j + 1) % block == 0 and j != n - 1:
                borders.append("border-right:2px solid #fff")

            # font size
            if n <= 9:
                fs = 24
            elif n <= 16:
                fs = 18
            else:
                fs = 14

            style = (
                f"text-align:center; font-size:{fs}px; color:{txt_color}; "
                "background-color:#0d1117; padding:0; margin:0; "
                + ";".join(borders)
            )
            disp = val if val != 0 else ""
            html += f'<td style="{style}">{disp}</td>'
        html += "</tr>"

    html += "</table></div>"
    st.write(html, unsafe_allow_html=True)

def run_automated_analysis(sizes, difficulties, num_puzzles_per_config, output_dir="sudoku_analysis_results"):
    """
    Run automated analysis across multiple configurations and save results
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir)
    
    # Initialize results storage
    all_results = []
    total_tests = len(sizes) * len(difficulties) * num_puzzles_per_config
    current_test = 0
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create results table placeholder
    results_placeholder = st.empty()
    
    # Run tests
    for size in sizes:
        for difficulty in difficulties:
            for puzzle_num in range(num_puzzles_per_config):
                current_test += 1
                progress = current_test / total_tests
                progress_bar.progress(progress)
                status_text.text(f"Testing {size}×{size} {difficulty} - Puzzle {puzzle_num + 1}/{num_puzzles_per_config}")
                
                # Generate puzzle
                try:
                    board = generate_puzzle(size, difficulty)
                    
                    # Test with each algorithm
                    results = {
                        'size': f'{size}×{size}',
                        'difficulty': difficulty.capitalize(),
                        'puzzle_num': puzzle_num + 1
                    }
                    
                    # Backtracking (only for size <= 16)
                    if size <= 16:
                        sol, t = solve_with_backtracking(board)
                        results['Backtracking'] = t if sol else None
                    else:
                        results['Backtracking'] = None
                    
                    # PySAT
                    sol, t = solve_with_sat(board)
                    results['PySAT'] = t if sol else None
                    
                    # MiniZinc (if available)
                    if shutil.which("minizinc"):
                        sol, t = solve_with_minizinc(board)
                        results['MiniZinc'] = t if sol else None
                    else:
                        results['MiniZinc'] = None
                    
                    all_results.append(results)
                    
                    # Update results table
                    df_temp = pd.DataFrame(all_results)
                    results_placeholder.dataframe(df_temp.tail(10))  # Show last 10 results
                    
                except Exception as e:
                    st.warning(f"Error testing {size}×{size} {difficulty} puzzle {puzzle_num + 1}: {str(e)}")
    
    # Convert results to DataFrame for analysis
    df_results = pd.DataFrame(all_results)
    
    # Save raw results
    csv_path = os.path.join(run_dir, "raw_results.csv")
    df_results.to_csv(csv_path, index=False)
    st.success(f"Raw results saved to: {csv_path}")
    
    # Generate and save visualizations
    st.subheader("Generating Analysis Visualizations")
    
    # Prepare data for visualization
    viz_data = []
    for _, row in df_results.iterrows():
        for algo in ['Backtracking', 'PySAT', 'MiniZinc']:
            if row[algo] is not None:
                viz_data.append({
                    'size': row['size'],
                    'difficulty': row['difficulty'],
                    'algorithm': algo,
                    'time': row[algo]
                })
    
    # Create comprehensive visualizations
    create_and_save_all_visualizations(viz_data, run_dir)
    
    # Generate summary report
    generate_summary_report(df_results, run_dir)
    
    st.success(f"Analysis complete! Results saved to: {run_dir}")
    return df_results

def create_and_save_all_visualizations(performance_data, output_dir):
    """
    Create and save all visualization types
    """
    df = pd.DataFrame(performance_data)
    
    # 1. Performance comparison by board size
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle('Performance Comparison by Board Size', fontsize=16, fontweight='bold')
    
    for idx, size in enumerate(['9×9', '16×16', '25×25']):
        ax = axes[idx]
        size_data = df[df['size'] == size]
        
        if not size_data.empty:
            pivot_data = size_data.pivot_table(
                index='difficulty', 
                columns='algorithm', 
                values='time',
                aggfunc='mean'
            )
            pivot_data.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{size} Puzzles')
            ax.set_ylabel('Average Time (seconds)')
            ax.set_yscale('log')
            ax.legend(title='Algorithm')
            ax.grid(True, alpha=0.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'performance_by_size.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Scalability comparison
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Scalability Analysis by Difficulty', fontsize=16, fontweight='bold')
    
    difficulties = ['Easy', 'Medium', 'Hard', 'Extreme']
    size_map = {'9×9': 9, '16×16': 16, '25×25': 25}
    
    for idx, difficulty in enumerate(difficulties):
        ax = axes[idx // 2, idx % 2]
        diff_data = df[df['difficulty'] == difficulty]
        
        for algorithm in df['algorithm'].unique():
            alg_data = diff_data[diff_data['algorithm'] == algorithm].copy()
            alg_data['numeric_size'] = alg_data['size'].map(size_map)
            
            # Calculate mean times for each size
            mean_times = alg_data.groupby('numeric_size')['time'].mean()
            
            if not mean_times.empty:
                ax.plot(mean_times.index, mean_times.values, 
                       marker='o', linewidth=2, markersize=8, 
                       label=algorithm)
        
        ax.set_title(f'{difficulty} Difficulty')
        ax.set_xlabel('Board Size')
        ax.set_ylabel('Average Time (seconds)')
        ax.set_yscale('log')
        ax.set_xticks([9, 16, 25])
        ax.set_xticklabels(['9×9', '16×16', '25×25'])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'scalability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Performance distribution
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    sns.violinplot(data=df, x='algorithm', y='time', ax=ax,
                  inner='quartile', scale='width', cut=0)
    ax.set_yscale('log')
    ax.set_title('Performance Distribution by Algorithm', fontsize=16, fontweight='bold')
    ax.set_ylabel('Solving Time (seconds, log scale)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    for i, algorithm in enumerate(df['algorithm'].unique()):
        data = df[df['algorithm'] == algorithm]['time']
        median = data.median()
        mean = data.mean()
        
        ax.text(i, median, f'Med: {median:.3f}s\nMean: {mean:.3f}s', 
               ha='center', va='bottom', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'performance_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Heatmap
    fig4, ax = plt.subplots(figsize=(10, 8))
    
    pivot_data = df.pivot_table(
        values='time', 
        index=['size', 'difficulty'], 
        columns='algorithm', 
        aggfunc='mean'
    )
    
    # Log transform for better visualization
    pivot_data_log = np.log10(pivot_data + 0.001)
    
    sns.heatmap(pivot_data_log, 
                annot=pivot_data.round(3), 
                fmt='.3f',
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Log10(Time + 0.001)'},
                linewidths=1,
                linecolor='white',
                ax=ax)
    
    ax.set_title('Algorithm Performance Heatmap', fontsize=16, fontweight='bold')
    ax.set_ylabel('Board Size and Difficulty')
    ax.set_xlabel('Algorithm')
    
    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    st.success("All visualizations saved!")

def generate_summary_report(df_results, output_dir):
    """
    Generate a text summary report of the analysis
    """
    report_path = os.path.join(output_dir, "summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("SUDOKU SOLVER PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 20 + "\n")
        
        for algo in ['Backtracking', 'PySAT', 'MiniZinc']:
            times = []
            for _, row in df_results.iterrows():
                if row[algo] is not None:
                    times.append(row[algo])
            
            if times:
                f.write(f"\n{algo}:\n")
                f.write(f"  Min time: {min(times):.4f}s\n")
                f.write(f"  Max time: {max(times):.4f}s\n")
                f.write(f"  Mean time: {np.mean(times):.4f}s\n")
                f.write(f"  Median time: {np.median(times):.4f}s\n")
                f.write(f"  Success rate: {len(times)}/{len(df_results)} ({100*len(times)/len(df_results):.1f}%)\n")
        
        # By configuration
        f.write("\n\nRESULTS BY CONFIGURATION\n")
        f.write("-" * 20 + "\n")
        
        for size in df_results['size'].unique():
            for difficulty in df_results['difficulty'].unique():
                f.write(f"\n{size} - {difficulty}:\n")
                config_data = df_results[(df_results['size'] == size) & 
                                       (df_results['difficulty'] == difficulty)]
                
                for algo in ['Backtracking', 'PySAT', 'MiniZinc']:
                    times = config_data[algo].dropna()
                    if len(times) > 0:
                        f.write(f"  {algo}: mean={np.mean(times):.4f}s, "
                               f"median={np.median(times):.4f}s, "
                               f"success={len(times)}/{len(config_data)}\n")
                    else:
                        f.write(f"  {algo}: No successful solves\n")
    
    st.success(f"Summary report saved to: {report_path}")

def create_performance_visualization(performance_data):
    """
    I'm creating performance visualizations from the collected data
    """
    if not performance_data:
        st.warning("No performance data available for visualization")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(performance_data)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Performance Comparison", "Scalability", "Distribution", "Heatmap"])
    
    with tab1:
        # Performance comparison by board size
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Performance Comparison by Board Size', fontsize=16, fontweight='bold')
        
        for idx, size in enumerate(['9×9', '16×16', '25×25']):
            ax = axes[idx]
            size_data = df[df['size'] == size]
            
            if not size_data.empty:
                pivot_data = size_data.pivot(index='difficulty', columns='algorithm', values='time')
                pivot_data.plot(kind='bar', ax=ax, width=0.8)
                ax.set_title(f'{size} Puzzles')
                ax.set_ylabel('Time (seconds)')
                ax.set_yscale('log')
                ax.legend(title='Algorithm')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        # Scalability comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Scalability Analysis by Difficulty', fontsize=16, fontweight='bold')
        
        difficulties = ['Easy', 'Medium', 'Hard', 'Extreme']
        size_map = {'9×9': 9, '16×16': 16, '25×25': 25}
        
        for idx, difficulty in enumerate(difficulties):
            ax = axes[idx // 2, idx % 2]
            diff_data = df[df['difficulty'] == difficulty]
            
            for algorithm in df['algorithm'].unique():
                alg_data = diff_data[diff_data['algorithm'] == algorithm].copy()
                alg_data['numeric_size'] = alg_data['size'].map(size_map)
                alg_data = alg_data.sort_values('numeric_size')
                
                # Only plot if we have data
                if not alg_data.empty:
                    ax.plot(alg_data['numeric_size'], alg_data['time'], 
                           marker='o', linewidth=2, markersize=8, 
                           label=algorithm)
            
            ax.set_title(f'{difficulty} Difficulty')
            ax.set_xlabel('Board Size')
            ax.set_ylabel('Time (seconds)')
            ax.set_yscale('log')
            ax.set_xticks([9, 16, 25])
            ax.set_xticklabels(['9×9', '16×16', '25×25'])
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        # Performance distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter out None values
        df_clean = df.dropna(subset=['time'])
        
        if not df_clean.empty:
            sns.violinplot(data=df_clean, x='algorithm', y='time', ax=ax,
                          inner='quartile', scale='width', cut=0)
            ax.set_yscale('log')
            ax.set_title('Performance Distribution by Algorithm', fontsize=16, fontweight='bold')
            ax.set_ylabel('Solving Time (seconds, log scale)')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            for i, algorithm in enumerate(df_clean['algorithm'].unique()):
                data = df_clean[df_clean['algorithm'] == algorithm]['time']
                median = data.median()
                mean = data.mean()
                
                ax.text(i, median, f'Median: {median:.3f}s', 
                       ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        st.pyplot(fig)
    
    with tab4:
        # Performance heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data for heatmap
        pivot_data = df.pivot_table(values='time', 
                                   index=['size', 'difficulty'], 
                                   columns='algorithm', 
                                   aggfunc='first')
        
        # Log transform for better visualization
        pivot_data_log = np.log10(pivot_data + 0.001)
        
        sns.heatmap(pivot_data_log, 
                    annot=pivot_data.round(3), 
                    fmt='.3f',
                    cmap='RdYlBu_r',
                    cbar_kws={'label': 'Log10(Time + 0.001)'},
                    linewidths=1,
                    linecolor='white',
                    ax=ax)
        
        ax.set_title('Algorithm Performance Heatmap', fontsize=16, fontweight='bold')
        ax.set_ylabel('Board Size and Difficulty')
        ax.set_xlabel('Algorithm')
        
        st.pyplot(fig)

# ---------------------------
# Main Streamlit App
# ---------------------------

def main():
    st.title("Sudoku Solver")

    # Initialize session state for performance data
    if 'performance_data' not in st.session_state:
        st.session_state.performance_data = []

    # Add seed control in the sidebar
    if 'seed' not in st.session_state:
        st.session_state.seed = random.randint(1, 1000000)
    
    seed_input = st.sidebar.text_input("Random Seed", value=str(st.session_state.seed))
    try:
        seed_value = int(seed_input)
        if seed_value != st.session_state.seed:
            st.session_state.seed = seed_value
            random.seed(seed_value)
    except ValueError:
        st.sidebar.error("Seed must be an integer")

    board_sizes = [9, 16, 25]
    n = st.sidebar.selectbox("Board size", board_sizes)
    difficulty = st.sidebar.selectbox("Difficulty", ["easy","medium","hard","extreme"])
    methods = ["Backtracking","PySAT","MiniZinc"]
    method = st.sidebar.radio("Solving Method", methods)

    # reset on size change
    if ("board" not in st.session_state) or (st.session_state.n != n):
        st.session_state.n        = n
        st.session_state.board    = [[0]*n for _ in range(n)]
        st.session_state.solution = None
        st.session_state.elapsed  = 0.0
        st.session_state.solved   = False

    # Generate Puzzle
    if st.sidebar.button("Generate Puzzle"):
        random.seed(st.session_state.seed)
        st.session_state.board  = generate_puzzle(n, difficulty)
        st.session_state.solved = False
        # Generate a new seed for next time
        st.session_state.seed = random.randint(1, 1000000)

    # Solve Puzzle
    if st.sidebar.button("Solve Puzzle"):
        b = st.session_state.board
        if method == "Backtracking":
            sol, t = solve_with_backtracking(b)
        elif method == "PySAT":
            sol, t = solve_with_sat(b)
        else:
            sol, t = solve_with_minizinc(b)
        if sol is None:
            st.error("No solution found.")
        else:
            st.session_state.solution = sol
            st.session_state.elapsed   = t
            st.session_state.solved    = True
            st.session_state.solution = sol
        st.session_state.elapsed   = t
        st.session_state.solved    = True
        
        # Record performance data
        st.session_state.performance_data.append({
            'size': f'{n}×{n}',
            'difficulty': difficulty.capitalize(),
            'algorithm': method,
            'time': t
        })

    # Display the puzzle (and solution highlights)
    st.subheader("Puzzle")
    display_board(
        st.session_state.board,
        st.session_state.solution if st.session_state.solved else None
    )

    if st.session_state.solved:
        st.write(f"Solving Time: {st.session_state.elapsed:.4f} seconds")

    # ——— Compare Solvers ———
    if st.sidebar.button("Compare Solvers"):
        board = st.session_state.board
        times = {}

        # Backtracking only for n ≤ 16
        if n <= 16:
            _, t_bt = solve_with_backtracking(board)
            times["Backtracking"] = t_bt or float("nan")
        else:
            times["Backtracking"] = float("nan")

        # PySAT
        _, t_sat = solve_with_sat(board)
        times["PySAT"] = t_sat or float("nan")

        # MiniZinc if available
        if shutil.which("minizinc"):
            _, t_mzn = solve_with_minizinc(board)
            times["MiniZinc"] = t_mzn or float("nan")
        else:
            times["MiniZinc"] = float("nan")

        # Build DataFrame
        df = pd.DataFrame.from_dict(times, orient="index", columns=["Time (s)"])

        # Table with "—" for NaN
        display_df = df.copy()
        display_df["Time (s)"] = display_df["Time (s)"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "—"
        )
        st.subheader("Solver Times")
        st.table(display_df)

        # Bar chart of valid times
        st.subheader("Solver Performance (bar chart)")
        st.bar_chart(df.dropna())
        
        for algorithm, time_val in times.items():
          if not math.isnan(time_val):
            st.session_state.performance_data.append({
                'size': f'{n}×{n}',
                'difficulty': difficulty.capitalize(),
                'algorithm': algorithm,
                'time': time_val
            })

        # Performance Analysis Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Performance Analysis")
    
    # Clear performance data button
    if st.sidebar.button("Clear Performance Data"):
        st.session_state.performance_data = []
        st.success("Performance data cleared!")
    
    # Show performance visualizations
    if st.sidebar.button("Show Performance Analysis"):
        if st.session_state.performance_data:
            st.subheader("Performance Analysis")
            create_performance_visualization(st.session_state.performance_data)
        else:
            st.warning("No performance data available. Solve some puzzles first!")
    
    # Export performance data
    if st.sidebar.button("Export Performance Data"):
        if st.session_state.performance_data:
            df_export = pd.DataFrame(st.session_state.performance_data)
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"sudoku_performance_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No performance data to export!")
    
    # Show current performance data
    if st.sidebar.button("Show Performance Data Table"):
        if st.session_state.performance_data:
            st.subheader("Collected Performance Data")
            df_perf = pd.DataFrame(st.session_state.performance_data)
            st.dataframe(df_perf)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            summary_stats = df_perf.groupby('algorithm')['time'].agg(['min', 'max', 'mean', 'median', 'count'])
            st.dataframe(summary_stats)
        else:
            st.info("No performance data collected yet. Solve some puzzles to see statistics!")
    
    # ADD THE AUTOMATED ANALYSIS SECTION HERE
    # Automated Analysis Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Automated Analysis")

    # Configuration for automated testing
    with st.sidebar.expander("Analysis Configuration"):
        selected_sizes = st.multiselect(
            "Board Sizes to Test",
            options=[9, 16, 25],
            default=[9, 16, 25]
        )
        
        selected_difficulties = st.multiselect(
            "Difficulties to Test",
            options=["easy", "medium", "hard", "extreme"],
            default=["easy", "medium", "hard", "extreme"]
        )
        
        num_puzzles = st.number_input(
            "Puzzles per Configuration",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of puzzles to test for each size/difficulty combination"
        )

    # Run automated analysis button
    if st.sidebar.button("Run Complete Analysis", type="primary"):
        st.header("Running Automated Analysis")
        st.info(f"Testing {len(selected_sizes)} sizes × {len(selected_difficulties)} difficulties × {num_puzzles} puzzles = {len(selected_sizes) * len(selected_difficulties) * num_puzzles} total tests")
        
        # Run the analysis
        results = run_automated_analysis(
            sizes=selected_sizes,
            difficulties=selected_difficulties,
            num_puzzles_per_config=num_puzzles
        )
        
        # Show final summary
        st.success("Analysis complete! Check the output directory for saved results and visualizations.")
        
        # Display download links for the generated files
        output_dir = f"sudoku_analysis_results/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if os.path.exists(output_dir):
            st.subheader("Download Results")
            
            # List all generated files
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                with open(file_path, 'rb') as f:
                    st.download_button(
                        label=f"Download {file}",
                        data=f.read(),
                        file_name=file,
                        mime='application/octet-stream'
                    )

if __name__ == "__main__":
    main()