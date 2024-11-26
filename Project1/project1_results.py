import nbformat
import re
import argparse

def process_notebook(file_path, sort_results=False):
    try:
        with open(file_path, 'r') as f:
            notebook = nbformat.read(f, as_version=4)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return

    output_lines = []
    for cell in notebook.cells:
        if cell.cell_type == 'code':  # We are only interested in code cells
            for output in cell.get('outputs', []):
                if output.output_type == 'stream':  # Capturing text output
                    output_lines.extend(output.text.splitlines())

    # List to store models with accuracy information
    models = []
    for line in output_lines:
        # Look for lines containing model info and accuracy on the same line
        if line.startswith('Model') and 'ACCURACY' in line:
            model_line = line.strip()

            # Extract the Test Accuracy value from the line using a regex
            match = re.search(r'Test Accuracy\s*([\d\.]+)', model_line)
            if match:
                test_accuracy = float(match.group(1))
                models.append((model_line, test_accuracy))

    # Debug output: Print found models and their accuracy
    if not models:
        print("No models or accuracy information found in the notebook output.")
    else:
        print(f"Found {len(models)} experiments with accuracy information.")

    # Sort the models by Test Accuracy in descending order if required
    if sort_results:
        models = sorted(models, key=lambda x: x[1], reverse=True)  # Sort by Test Accuracy

    # Write the (optionally sorted) models to a text file
    try:
        with open('project1_results.txt', 'w') as f:
            for model, _ in models:
                f.write(f'{model}\n')
        print("Results written to 'project1_results.txt'.")
    except Exception as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process notebook outputs and optionally sort by accuracy.")
    parser.add_argument('-s', '--sort', action='store_true', help="Sort the results by the Test Accuracy in descending order.")
    
    args = parser.parse_args()
    
    # Call the main function with sort option based on the command-line argument
    process_notebook('project1.ipynb', sort_results=args.sort)
