import json
import os
from pathlib import Path

def patch_notebook():
    notebook_path = Path('notebooks/diputraxv10.ipynb')
    if not notebook_path.exists():
        print(f"Error: {notebook_path} not found.")
        return
        
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # We will replace cell 79 source
    new_source = [
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.image as mpimg\n",
        "from pathlib import Path\n",
        "from IPython.display import display\n",
        "\n",
        "# Support running from project root or notebooks/ directory\n",
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
        "IMG_DIR = PROJECT_ROOT / \"reports\" / \"eda\"\n",
        "IMG_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "def show_img(fname, title=None, figsize=(16, 10)):\n",
        "    img = mpimg.imread(IMG_DIR / fname)\n",
        "    fig, ax = plt.subplots(figsize=figsize)\n",
        "    ax.imshow(img)\n",
        "    ax.axis(\"off\")\n",
        "    if title:\n",
        "        ax.set_title(title, fontsize=13, fontweight=\"bold\", pad=10)\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "pd.set_option(\"display.float_format\", \"{:.3f}\".format)\n",
        "print(\"Setup OK\")"
    ]
    
    nb['cells'][79]['source'] = new_source
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print("Successfully patched notebooks/diputraxv10.ipynb cell 79!")

if __name__ == '__main__':
    patch_notebook()
