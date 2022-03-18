Installation
============

Using pip:

::

    pip install abcvoting

Latest development version from source:

::

    git clone https://github.com/martinlackner/abcvoting/
    python setup.py install

Requirements:

- Python 3.7+

- The following pip packages are required and installed automatically: mip, networkx, ortools, and ruamel.yaml.

Optional requirements:

- `gmpy2 <https://pypi.org/project/gmpy2/>`_: Some functions use fractions (e.g., `compute_seqphragmen`).
  These compute significantly faster if the module gmpy2 is available.
  If gmpy2 is not available, the much slower Python module
  `fractions <https://docs.python.org/2/library/fractions.html>`_ is used.

- `Gurobi (gurobipy) <https://www.gurobi.com/>`_: Most computationally hard rules are also implemented via the ILP
  solver Gurobi. The corresponding functions require
  `gurobipy <https://www.gurobi.com/documentation/8.1/quickstart_mac/the_gurobi_python_interfac.html>`_.
  If Gurobi is not available, the open-source solver `CBC <https://github.com/coin-or/Cbc>`_ is a slower alternative
  (that is installed automatically as part of `mip`).
