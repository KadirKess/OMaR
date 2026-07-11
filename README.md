# Analyse automatique de partitions musicales

Extraction automatique des données d'une partition musicale : reconnaissance
des notes, des clefs et des silences, avec export MIDI possible.

Le projet comporte deux implémentations du même pipeline OMR (Optical Music
Recognition) :

- **`prototype/`** — prototype de référence en Python (NumPy, SciPy,
  scikit-image).
- **`cpp/`** — portage C++ (C++20, OpenCV pour les primitives bas niveau,
  nlohmann/json, GoogleTest).

Les algorithmes cœur (estimation des dimensions de portée, détection et
suppression des lignes de portée, découpage en primitives — hampes, têtes de
notes, ligatures — et reconstruction par graphe) sont implémentés à la main.

## Pipeline

Binarisation → suppression des lignes de portée → découpage en primitives →
reconstruction du graphe de notation → sémantique (hauteurs, durées) → export
MIDI.

## Structure

```
prototype/   prototype Python (src/, tests/, benchmarks/)
cpp/         portage C++ (src/, apps/, tests/)
reports/     rapports LaTeX (état de l'art, méthodes, résultats)
```

## Utilisation

### Python

```sh
cd prototype
uv sync
uv run main.py
uv run pytest          # tests
```

### C++

```sh
cd cpp
cmake -B build
cmake --build build
ctest --test-dir build # tests
```
