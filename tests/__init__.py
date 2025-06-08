from sys import path as p; from pathlib import Path;
[p.extend([str(d), str(d.parent)]) for f in {__file__} if (d := Path(f).resolve().parent) not in p]