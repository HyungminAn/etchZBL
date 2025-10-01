import yaml
import os
import shutil
from ase.io import read, write

def process_yaml_tree(node, path_prefix=[], out_dir="poscars"):
    if isinstance(node, dict):
        for k, v in node.items():
            process_yaml_tree(v, path_prefix + [k], out_dir)
    elif isinstance(node, str):
        src_path = node
        if os.path.exists(src_path):
            # 경로 조합: 'POSCAR_bulk_crystalline_alpha_quartz'
            name_suffix = "_".join(path_prefix)
            filename = f"POSCAR_{name_suffix}"
            dst_path = os.path.join(out_dir, filename)

            os.makedirs(out_dir, exist_ok=True)
            if src_path.endswith('POSCAR'):
                shutil.copy(src_path, dst_path)
                print(f"✅ Copied: {src_path} → {dst_path}")
            elif src_path.endswith('OUTCAR'):
                atoms = read(src_path)
                write(dst_path, atoms, format='vasp')
                print(f"✅ Copied: {src_path} → {dst_path} (converted to POSCAR)")
        else:
            print(f"❌ Not found: {src_path}")

# YAML 파일 읽기
with open("path.yaml", "r") as f:
    config = yaml.safe_load(f)

# 실행
process_yaml_tree(config)

