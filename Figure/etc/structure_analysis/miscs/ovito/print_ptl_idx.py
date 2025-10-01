import sys


def main():
    indices = [f"ParticleIdentifier == {i}" for i in sys.argv[1:]]
    output = " || ".join(indices)
    print(output)


main()
