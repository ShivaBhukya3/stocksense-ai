"""Quick start script — run this to launch StockSense AI."""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

from app import app

if __name__ == '__main__':
    print('StockSense AI - http://localhost:8050')
    app.run(debug=False, port=8050)
