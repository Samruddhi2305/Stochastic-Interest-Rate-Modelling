import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import time

def run_notebook(notebook_path):
    print(f"Loading notebook {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    print("Executing notebook (this may take up to a minute)...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    start_time = time.time()
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
        print(f"Execution completed successfully in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print("Error executing the notebook:")
        print(e)
        raise e
        
    print(f"Saving executed notebook back to {notebook_path}...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print("Done!")

if __name__ == '__main__':
    run_notebook("Stochastic_Interest_Rate_Modelling.ipynb")
