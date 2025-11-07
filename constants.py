from enum import Enum
from pathlib import Path
import os

class Choice(float, Enum):
    LEFT = -1.0
    NULL =  0.0
    RIGHT = 1.0

# Default to project root if env var not set; no trailing slash issues with Path
HOMEDIR = Path(os.getenv(
    "LEARNINGRULES_HOME", 
    default=Path(__file__).resolve().parents[0]
    ))

if __name__ == "__main__":
    print("This is the constants module.")
    print("Choice values: LEFT =", Choice.LEFT.value, ", RIGHT =", Choice.RIGHT.value)
    print("HOMEDIR is set to:", HOMEDIR)