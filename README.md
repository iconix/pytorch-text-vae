A partial reimplementation of "Generating Sentences From a Continuous Space", Bowman, Vilnis, Vinyals, Dai, Jozefowicz, Bengio (<https://arxiv.org/abs/1511.06349>).

Based on code from Kyle Kastner (`@kastnerkyle`) <https://github.com/kastnerkyle/pytorch-text-vae>, adapted to support the [`deephypebot`](https://github.com/iconix/deephypebot) project.

---

Changes in this [detached fork](https://github.com/kastnerkyle/pytorch-text-vae/):
- Update compatibility to Python 3 and PyTorch 0.4
- Add `generate.py` for sampling
- Add special support for JSON reading and thought vector conditioning
- Some code cleanup
- Add `setup.py` for package support as `pytorchtextvae`
- Train/test data split support
