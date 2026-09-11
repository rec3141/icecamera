"""Refresh completed combined projection annotations without refitting its layout."""
import importlib.util
from pathlib import Path

portal=Path('/home/cryomics/Downloads/amundsen-ice-2025-2026-tsne')
if (portal/'embedding.json').exists():
    spec=importlib.util.spec_from_file_location('explorer',Path(__file__).resolve().with_name('ice-region-explorer.py'))
    explorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(explorer)
    explorer.render(portal,[],False)
    print('Updated combined 2025–2026 model annotations')
