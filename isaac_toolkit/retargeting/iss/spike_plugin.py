"""Compile an ISE plugin against the user's matching MLonMCU Spike build."""
import json
import os
from pathlib import Path
import pickle
import subprocess

from m2isar.backends.spike.plugin import generate_plugin, identifier


def build_plugin(model_path, core_name, output, install_dir=None, source_dir=None, build_dir=None):
    model_path, output = Path(model_path), Path(output)
    with model_path.open('rb') as handle:
        core = pickle.load(handle).cores[core_name]
    source = generate_plugin(core, custom_only=True, strict=True)
    install_dir = install_dir or os.environ.get('SPIKE_INSTALL_DIR')
    if install_dir is None and os.environ.get('MLONMCU_HOME'):
        install_dir = Path(os.environ['MLONMCU_HOME']) / 'deps/install/spike'
    includes = []
    if install_dir:
        install_dir = Path(install_dir).expanduser().resolve()
        includes += [install_dir / 'include/riscv', install_dir / 'include', install_dir / 'include/fesvr']
    if source_dir:
        source_dir = Path(source_dir).resolve()
        includes += [source_dir, source_dir / 'riscv', source_dir / 'fesvr', source_dir / 'softfloat']
    if build_dir:
        includes.append(Path(build_dir).resolve())
    if not any((path / 'extension.h').is_file() for path in includes):
        raise FileNotFoundError('Matching Spike headers required: use MLonMCU spike.full=true or provide Spike source/build directories')
    output.mkdir(parents=True, exist_ok=True)
    cpp = output / 'extension.cc'
    cpp.write_text(source)
    library = (output / 'libextension.so').resolve()
    result = subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-shared', '-fPIC',
                             *[f'-I{path}' for path in includes], str(cpp), '-o', str(library)],
                            capture_output=True, text=True)
    (output / 'build.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'Spike plugin compilation failed: {output / "build.log"}\n{result.stderr}')
    manifest = {'schema': 1, 'core': core_name, 'extlibs': [str(library)],
                'extension_names': [identifier(core_name)],
                'spike_install_dir': str(install_dir) if install_dir else None}
    (output / 'spike_plugin.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
