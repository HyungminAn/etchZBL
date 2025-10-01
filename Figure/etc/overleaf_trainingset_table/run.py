def main():
    data = {
            'substrate_dominant': {
                '3:6:3:1:3': (74496, None),
                '3:3:3:1:3': (None, 60684),

                '3:6:2:1:1': (70616, None),
                '3:4:2:1:1': (None, 59906),

                '6:12:1:1:1': (81480, None),
                '4:10:1:1:1': (None, 66130),

                '3:3:1:1:1': (76725, None),
                '6:8:1:1:1': (None, 55505),
                },
            'mixed-layer': {
                '1:2:8:1:6': (83808, None),
                '1:1:8:1:6': (None, 51765),

                '1:1:1:12:1': (74496, None),
                '1:1:1:13:1': (None, 66130),

                '1:8:1:1:1': (74208, None),
                '1:1:0:0:3': (77700, None),

                '7:1:7:1:1': (None, 66130)
                },
            'carbon-film_(bulk)': {
                '0:0:1:0:0.2': (47232, 93888),
                '0:0:1:0:0.4': (39168, 77760),
                '0:0:1:0.05:1': (64224, 127872),
                '0:0:1:0.15:1': (64224, 127872),
                '0:0:1:0.1:0.1': (64224, 127872),
                '0:0:1:0.1:0.4': (64224, 127872),
                '0:0:1:0.1:0.6': (64224, 127872),
                '0:0:1:0.1:0.8': (64224, 127872),
                },
            'carbon-film_(slab)': {
                '0:0:1:0:0.2': (28800, None),
                '0:0:1:0:0.4': (28800, None),
                '0:0:1:0.05:1': (27360, None),
                '0:0:1:0.15:1': (27360, None),
                '0:0:1:0.1:0.1': (28800, None),
                '0:0:1:0.1:0.4': (28800, None),
                '0:0:1:0.1:0.6': (27360, None),
                '0:0:1:0.1:0.8': (27360, None),
                },
            }

    for data_type, data_dict in data.items():
        line_count = 0
        for comp, (value1, value2) in data_dict.items():
            line = ''
            if line_count < len(data_type.split('_')):
                header = data_type.split('_')[line_count]
                line = f'{header:20s} & '
            else:
                line = f'{"":20s} & '

            line += f'{comp:20s} & '
            if data_type != 'carbon-film_(slab)':
                value1 = f'{value1:,}' if value1 is not None else '.'
                value2 = f'{value2:,}' if value2 is not None else '.'
                if value1 != '.':
                    value1 = value1.replace(',', ' ')
                if value2 != '.':
                    value2 = value2.replace(',', ' ')
                line += f'{value1:10s} & '
                line += f'{value2:10s} '
            else:
                value1 = f'{value1:,}' if value1 is not None else '.'
                if value1 != '.':
                    value1 = value1.replace(',', ' ')
                line += r'\multicolumn{2}{c}' + f'{{{value1}}} '
            line += r'\\'
            print(line)
            line_count += 1
        print(r'\hline')


if __name__ == '__main__':
    main()
