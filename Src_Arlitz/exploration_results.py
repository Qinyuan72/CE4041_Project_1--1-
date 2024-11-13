import nbformat
import re

def process_notebook(file_path):
    with open(file_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    output_lines = []
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            for output in cell.get('outputs', []):
                if output.output_type == 'stream':
                    output_lines.extend(output.text.splitlines())

    models = []
    for i, line in enumerate(output_lines):
        # Look for model description lines
        if line.startswith(('Model1', 'Model2', 'Model3')):
            model_line = line.strip()
            
            # Check the following lines for accuracy information
            for j in range(i + 1, min(i + 5, len(output_lines))):  # Look ahead a few lines for the accuracy
                if 'ACCURACY' in output_lines[j]:
                    accuracy_line = output_lines[j].strip()
                    models.append((model_line, accuracy_line))
                    break  # Stop searching once we've found the accuracy

    # Sort the models by accuracy in decreasing order
    # Extract accuracy value for sorting
    sorted_models = sorted(models, key=lambda x: float(re.search(r'ACCURACY:\s*([0-9\.]+)', x[1]).group(1)), reverse=True)
    
    # Write the sorted lines to a text file
    with open('exploration_results.txt', 'w') as f:
        for model, accuracy_info in sorted_models:
            f.write(f'{model} - {accuracy_info}\n')

# Example usage
process_notebook('exploration_v2.0.ipynb')

