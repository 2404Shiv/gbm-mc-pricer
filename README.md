# GBM Monte Carlo Pricer (C++17)

**What it is:** A single-binary Monte Carlo pricer for European calls/puts under GBM with
**Greeks**, **antithetic** + **control-variate** variance reduction, and a 1-day **P&L/VaR** panel.

**Why it matters:** Demonstrates clean Monte Carlo engineering, numerical Greeks, variance-reduction
craft, and risk linkage—exactly the bread-and-butter of Strats/QR.

## Features
- GBM paths; European call/put payoffs
- Variance reduction: **antithetic** + **control variate** with optimal \(b^\*\) estimated on the fly
- Greeks: **BS closed-form**, **pathwise Δ**, **CRN bump-and-revalue** for Γ/Vega/Θ/ρ
- **Risk mini-panel:** 1-day revaluation VaR95/99 and ES95
- Reproducible (seeded), single portable C++17 binary, no deps

## Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/pricer
Build:

```
clang++ -O3 -std=c++17 -o pricer src/main.cpp
./pricer
```
