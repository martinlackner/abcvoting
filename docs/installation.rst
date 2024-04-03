Installation
============

Using pip:

::

    pip install abcvoting

Latest development version from source:

::

    git clone https://github.com/martinlackner/abcvoting/
    pip install .

Requirements:

- Python 3.8+

- The following pip packages are required and installed automatically: `gurobipy`, `mip`, `networkx`, `numpy`, `ruamel.yaml`, `preflibtools`, and `prefsampling`.

Optional requirements:

- `gmpy2 <https://pypi.org/project/gmpy2/>`_: Some functions use fractions (e.g., `compute_seqphragmen`).
  These compute significantly faster if the module gmpy2 is available.
  If gmpy2 is not available, the much slower Python module
  `fractions <https://docs.python.org/2/library/fractions.html>`_ is used.

::

    pip install gmpy2

- `ortools <https://developers.google.com/optimization/install/python>`_:
  Ortools can be used as an alternative solver for some ABC voting rules (Monroe, CC, Minimax AV).
  Advantage: open source, faster than CBC. Disadvantage: not as reliable as Gurobi (proprietary).

::

    pip install ortools

- `Gurobi (gurobipy) <https://www.gurobi.com/>`_: Most computationally hard rules are also implemented via the ILP
  solver Gurobi. The corresponding functions require
  `gurobipy <https://www.gurobi.com/documentation/quickstart.html>`_.
  While `gurobipy` is installed by default (together with abcvoting), it requires a license to solve larger instances
  (`academic licenses <https://www.gurobi.com/academia/academic-program-and-licenses/>`_ are available).
  If Gurobi is not available, the open-source solver `CBC <https://github.com/coin-or/Cbc>`_ is a slower alternative
  (that is installed automatically as part of `mip`).

Developer tools (unit testing, etc):

::

    pip install .[dev]
