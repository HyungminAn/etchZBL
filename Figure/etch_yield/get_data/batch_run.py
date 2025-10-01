import os
import yaml
import subprocess


def main():
    path_yaml = "./path.yaml"
    path_bin = "./bin.etchyield"
    interval = 400

    print(f"Batch running with interval = {interval}")
    print(f"Using path yaml: {path_yaml}")
    print(f"Using binary: {path_bin}")

    # 1. read path.yaml
    with open(path_yaml, 'r') as f:
        ion_dict = yaml.safe_load(f)

    # 2. process for each (ion_type-ion_energy) pair
    for ion_type, energy_map in ion_dict.items():
        for energy_str, src_list in energy_map.items():
            ion_E = int(energy_str)

            for src in src_list:
                if not os.path.exists(src):
                    print(f"[skip] {src} not found.")
                    breakpoint()
            src_path = " ".join(src_list)

            dst = f"{ion_type}_{ion_E}"
            if os.path.exists(f'{dst}.dat'):
                print(f"[skip] {dst}.dat already exists.")
                continue

            print(f"\n[run] {ion_type} {ion_E}eV")
            print(f"  src: {src_path}")
            print(f"  dst: {dst}")

            # 3. run the binary for this pair
            commands = [path_bin, str(interval), dst, *src_list]

            print(f"    -> Running: {' '.join(commands)}")
            subprocess.run(commands, env={**os.environ})

    print("\n✅ Batch run finished.")


if __name__ == "__main__":
    main()
