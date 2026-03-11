"""
============================================================================
ROBO-ADVISORY SYSTEM — UNIFIED ENTRY POINT
============================================================================
Usage:
  python main.py                          # Generate portfolio (moderate profile)
  python main.py --profile aggressive     # Use aggressive profile
  python main.py --compare                # Compare 3 risk profiles
  python main.py --capital 50000          # Custom capital ($50K)
  python main.py --metrics                # Generate metrics report only
  python main.py --train-technical        # Retrain technical analysis model

Training (runs separately):
  python "technical analysis/train_technical.py"           # Full training
  python "technical analysis/train_technical.py" --max_stocks 20  # Quick test
============================================================================
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    if '--train-technical' in sys.argv:
        # Run technical model training
        train_script = os.path.join(BASE_DIR, 'technical analysis', 'train_technical.py')
        remaining_args = [a for a in sys.argv[1:] if a != '--train-technical']
        os.system(f'python "{train_script}" {" ".join(remaining_args)}')
    else:
        # Run portfolio generation
        from generate_portfolio import main
        main()