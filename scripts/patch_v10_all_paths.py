import json
from pathlib import Path

def patch_all_paths():
    notebook_path = Path('notebooks/diputraxv10.ipynb')
    if not notebook_path.exists():
        print(f"Error: {notebook_path} not found.")
        return
        
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 1. Patch Cell 11
    # Find DATA_PATH = Path("C:/Users/zigma/Projects/diputrax/data/clean/diputados_20260421_205712.parquet")
    cell_11_src = nb['cells'][11]['source']
    for idx, line in enumerate(cell_11_src):
        if 'DATA_PATH = Path(' in line and 'zigma' in line:
            cell_11_src[idx] = 'PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()\n'
            cell_11_src.insert(idx + 1, 'DATA_PATH = PROJECT_ROOT / "data" / "clean" / "diputados_20260421_205712.parquet"\n')
            print("Patched Cell 11 DATA_PATH")
            break
            
    # 2. Patch Cell 70
    # Find PARQUET = Path("C:/Users/zigma/Projects/diputrax/data/clean/diputados_20260421_205712.parquet")
    cell_70_src = nb['cells'][70]['source']
    for idx, line in enumerate(cell_70_src):
        if 'PARQUET = Path(' in line and 'zigma' in line:
            cell_70_src[idx] = 'PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()\n'
            cell_70_src.insert(idx + 1, 'PARQUET = PROJECT_ROOT / "data" / "clean" / "diputados_20260421_205712.parquet"\n')
            print("Patched Cell 70 PARQUET")
            break

    # 3. Patch Cell 81
    # Find REPORT_DIR = pathlib.Path('C:/Users/zigma/Projects/diputrax/reports/eda')
    cell_81_src = nb['cells'][81]['source']
    for idx, line in enumerate(cell_81_src):
        if 'REPORT_DIR = pathlib.Path(' in line and 'zigma' in line:
            cell_81_src[idx] = 'PROJECT_ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()\n'
            cell_81_src.insert(idx + 1, 'REPORT_DIR = PROJECT_ROOT / "reports" / "eda"\n')
            print("Patched Cell 81 REPORT_DIR")
            break

    # 4. Patch Cell 88 savefig
    cell_88_src = nb['cells'][88]['source']
    for idx, line in enumerate(cell_88_src):
        if 'plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/' in line:
            line_cleaned = line.replace('plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/', 'plt.savefig(REPORT_DIR / "')
            cell_88_src[idx] = line_cleaned
            print("Patched Cell 88 savefig")
            break

    # 5. Patch Cell 91 savefig
    cell_91_src = nb['cells'][91]['source']
    for idx, line in enumerate(cell_91_src):
        if 'plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/' in line:
            line_cleaned = line.replace('plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/', 'plt.savefig(REPORT_DIR / "')
            cell_91_src[idx] = line_cleaned
            print("Patched Cell 91 savefig")
            break

    # 6. Patch Cell 108 savefig
    cell_108_src = nb['cells'][108]['source']
    for idx, line in enumerate(cell_108_src):
        if 'plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/' in line:
            line_cleaned = line.replace('plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/', 'plt.savefig(REPORT_DIR / "')
            cell_108_src[idx] = line_cleaned
            print("Patched Cell 108 savefig")
            break

    # 7. Patch Cell 124 savefig
    cell_124_src = nb['cells'][124]['source']
    for idx, line in enumerate(cell_124_src):
        if 'plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/' in line:
            line_cleaned = line.replace('plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/', 'plt.savefig(REPORT_DIR / "')
            cell_124_src[idx] = line_cleaned
            print("Patched Cell 124 savefig")
            break

    # 8. Patch Cell 141 savefig
    cell_141_src = nb['cells'][141]['source']
    for idx, line in enumerate(cell_141_src):
        if 'plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/' in line:
            line_cleaned = line.replace('plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/', 'plt.savefig(REPORT_DIR / "')
            cell_141_src[idx] = line_cleaned
            print("Patched Cell 141 savefig")
            break

    # 9. Patch Cell 148 savefig
    cell_148_src = nb['cells'][148]['source']
    for idx, line in enumerate(cell_148_src):
        if 'plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/' in line:
            line_cleaned = line.replace('plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/', 'plt.savefig(REPORT_DIR / "')
            cell_148_src[idx] = line_cleaned
            print("Patched Cell 148 savefig")
            break

    # 10. Patch Cell 160 savefig
    cell_160_src = nb['cells'][160]['source']
    for idx, line in enumerate(cell_160_src):
        if 'plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/' in line:
            line_cleaned = line.replace('plt.savefig("C:/Users/zigma/Projects/diputrax/reports/eda/', 'plt.savefig(REPORT_DIR / "')
            cell_160_src[idx] = line_cleaned
            print("Patched Cell 160 savefig")
            break

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print("Notebook path patch complete!")

if __name__ == '__main__':
    patch_all_paths()
