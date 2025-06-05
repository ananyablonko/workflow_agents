from sys import path as p; from pathlib import Path; from dotenv import find_dotenv;
[p.extend([str(d), str(d.parent)]) for f in {find_dotenv("config.py"), __file__} if (d := Path(f).resolve().parent) not in p]

from gather_agent import GatherAgent

__all__ = ['GatherAgent']