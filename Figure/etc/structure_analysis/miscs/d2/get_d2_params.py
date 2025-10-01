import numpy as np


def main():
    """
    See: https://doi.org/10.1002/jcc.20495 (DFT-D2 original paper)
    """
    R0_dict = {
        'Si': 1.716,
        'O': 1.342,
        'C': 1.452,
        'H': 1.001,
        'F': 1.287,
    }
    C6_dict = {
        'Si': 9.23,
        'O': 0.70,
        'C': 1.75,
        'H': 0.14,
        'F': 0.75,
    }

    elem_list = ['Si', 'O', 'C', 'H', 'F']
    c6_conv_factor = 10.36565469
    for row in range(5):
        for col in range(row, 5):
            elem_1 = elem_list[row]
            elem_2 = elem_list[col]

            r0_1 = R0_dict[elem_1]
            r0_2 = R0_dict[elem_2]
            R0 = r0_1 + r0_2

            c6_1 = C6_dict[elem_1]
            c6_2 = C6_dict[elem_2]
            c6 = np.sqrt(c6_1 * c6_2) * c6_conv_factor

            line = f"pair_coeff {row+1} {col+1} d2 {c6:.6f} {R0:.3f}"
            print(line)


main()
