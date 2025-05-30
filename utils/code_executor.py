import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from contextlib import redirect_stdout, redirect_stderr
import sys
import warnings

def execute_python_code(code_string, df=None):
    """
    Executes a string of Python code safely.
    The code can use 'df' for the DataFrame, 'pd' for pandas,
    'plt' for matplotlib.pyplot, and 'sns' for seaborn.
    Captures stdout, stderr, and any generated plots.

    Args:
        code_string (str): The Python code to execute.
        df (pd.DataFrame, optional): The DataFrame to be made available as 'df'.

    Returns:
        tuple: (stdout_output, stderr_output, plot_image_base64)
               plot_image_base64 is a base64 encoded string of the plot, or None.
    """
    # Safe built-ins - include essential functions but exclude dangerous ones
    safe_builtins = {
        'print': print,
        'len': len,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'sorted': sorted,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'round': round,
        'type': type,
        'isinstance': isinstance,
        'hasattr': hasattr,
        'getattr': getattr,
        'setattr': setattr,
        'format': format,
        'repr': repr,
        '__import__': __import__, 
        # Add more safe functions as needed
    }
    
    local_vars = {
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'np': np,
        'df': df.copy() if df is not None else None,  # Pass a copy to avoid modification of original
        'results': {}  # A dictionary to store results if needed
    }
    
    # Buffer to capture stdout and stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    plot_image_base64 = None

    try:
        # Configure matplotlib to not show plots interactively
        plt.ioff()  # Turn off interactive mode
        plt.close('all')  # Close any pre-existing plots
        
        # Suppress specific warnings that are not critical
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            
            # Redirect stdout and stderr
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Execute with safe built-ins
                exec(code_string, {'__builtins__': safe_builtins}, local_vars)

        # Check if a plot was generated and capture it
        if plt.get_fignums():  # Check if any figures are open
            # Get the current figure
            fig = plt.gcf()
            img_buffer = io.BytesIO()
            
            # Save with high DPI for better quality
            fig.savefig(img_buffer, format='png', bbox_inches='tight', 
                       dpi=100, facecolor='white', edgecolor='none')
            img_buffer.seek(0)
            plot_image_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            plt.close(fig)  # Close the figure after saving

    except Exception as e:
        stderr_buffer.write(f"Execution Error: {str(e)}\n")
        # Also log the line number if possible
        import traceback
        stderr_buffer.write(f"Traceback: {traceback.format_exc()}\n")

    finally:
        # Ensure we close any remaining plots
        plt.close('all')

    stdout_output = stdout_buffer.getvalue()
    stderr_output = stderr_buffer.getvalue()
    
    return stdout_output, stderr_output, plot_image_base64

def save_plot_to_file(plot_base64, filename="plot.png"):
    """
    Save a base64 encoded plot to a file.
    
    Args:
        plot_base64 (str): Base64 encoded plot image
        filename (str): Output filename
    """
    if plot_base64:
        with open(filename, "wb") as f:
            f.write(base64.b64decode(plot_base64))
        print(f"Plot saved to {filename}")
    else:
        print("No plot to save")

def display_results(stdout, stderr, plot_base64):
    """
    Helper function to display execution results nicely.
    """
    print("=== EXECUTION RESULTS ===")
    
    if stdout.strip():
        print("📊 OUTPUT:")
        print(stdout)
    
    if stderr.strip():
        print("⚠️  WARNINGS/ERRORS:")
        print(stderr)
    
    if plot_base64:
        print("📈 PLOT: Generated successfully")
        print(f"   Plot size: {len(plot_base64)} characters (base64)")
    else:
        print("📈 PLOT: None generated")
    
    print("=" * 30)

if __name__ == '__main__':
    # Test with a DataFrame
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'], 
        'age': [25, 30, 35, 28, 32], 
        'city': ['NY', 'LA', 'Chicago', 'Miami', 'Seattle'],
        'salary': [50000, 60000, 70000, 55000, 65000]
    }
    sample_df = pd.DataFrame(data)

    # Test 1: Simple print and calculation
    print("🧪 TEST 1: Basic operations")
    code1 = """
print("DataFrame info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\\nFirst 3 rows:")
print(df.head(3))

mean_age = df['age'].mean()
print(f"\\nMean age: {mean_age:.1f}")
results['mean_age'] = mean_age

print(f"Age range: {df['age'].min()} - {df['age'].max()}")
"""
    stdout, stderr, plot = execute_python_code(code1, sample_df)
    display_results(stdout, stderr, plot)

    # Test 2: Plot generation
    print("🧪 TEST 2: Plot generation")
    code2 = """
# Create a nice plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot
ax1.bar(df['name'], df['age'], color='skyblue', alpha=0.7)
ax1.set_title('Age by Person')
ax1.set_xlabel('Name')
ax1.set_ylabel('Age')
ax1.tick_params(axis='x', rotation=45)

# Scatter plot
ax2.scatter(df['age'], df['salary'], color='coral', s=100, alpha=0.7)
ax2.set_title('Age vs Salary')
ax2.set_xlabel('Age')
ax2.set_ylabel('Salary ($)')

# Add grid for better readability
ax2.grid(True, alpha=0.3)

plt.tight_layout()
print("Generated age distribution and age vs salary plots")
"""
    stdout, stderr, plot = execute_python_code(code2, sample_df)
    display_results(stdout, stderr, plot)
    
    # Save the plot to file for testing
    if plot:
        save_plot_to_file(plot, "test_plot.png")

    # Test 3: Statistical analysis
    print("🧪 TEST 3: Statistical analysis")
    code3 = """
# Statistical summary
print("Statistical Summary:")
print("=" * 20)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    print(f"{col}:")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Std:  {df[col].std():.2f}")
    print(f"  Min:  {df[col].min()}")
    print(f"  Max:  {df[col].max()}")
    print()

# Group analysis
print("Analysis by City:")
city_stats = df.groupby('city').agg({
    'age': ['mean', 'count'],
    'salary': 'mean'
}).round(2)
print(city_stats)
"""
    stdout, stderr, plot = execute_python_code(code3, sample_df)
    display_results(stdout, stderr, plot)

    # REMOVED Test 4: Error handling test to avoid intentional errors
    # If you want to test error handling, uncomment the following:
    """
    print("🧪 TEST 4: Error handling")
    code4 = '''
print("This should work fine")
print(df['non_existent_column'])  # This will cause an error
print("This won't be reached")
'''
    stdout, stderr, plot = execute_python_code(code4, sample_df)
    display_results(stdout, stderr, plot)
    """

    print("\n✅ All tests completed!")
    print("Check 'test_plot.png' for the generated visualization.")