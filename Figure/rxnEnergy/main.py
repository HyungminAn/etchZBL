import sys
from stableconfigfinder import StableConfigFinder
from imageloader import ImageLoader
# from identifier import BulkGasIdentifier
# from uniquerxngenerator import UniqueRxnGenerator


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_src>")
        sys.exit(1)
    path_src = sys.argv[1]

    scf = StableConfigFinder()
    scf.run(path_src)

    ### ---
    # NNP Relax should be done here
    ### ---

    il = ImageLoader()
    is_nnp_relax_done = il.run()
    if is_nnp_relax_done is None:
        print("NNP relaxation is not done. Exiting.")
        sys.exit(1)

    # bgi = BulkGasIdentifier()
    # bgi.run()

    # urg = UniqueRxnGenerator()
    # urg.run()

    ### ---
    # DFT Relax should be done here
    ### ---


if __name__ == "__main__":
    main()
