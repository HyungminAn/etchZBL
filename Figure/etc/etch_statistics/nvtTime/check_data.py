import sys
import pickle


def main():
    src = sys.argv[1]
    with open(src, 'rb') as f:
        data = pickle.load(f)

    data = [(incidence, nvt_temp, nvt_time)
            for incidence, (nvt_temp, nvt_time) in data.items()]
    IDX_NVT_TEMP = 1
    IDX_NVT_TIME = 2
    data_sorted_temp = sorted(data, key=lambda x: x[IDX_NVT_TEMP])
    data_sorted_time = sorted(data, key=lambda x: x[IDX_NVT_TIME], reverse=True)
    PRINT_COUNT = 10

    print('-' * 80)
    for i in range(PRINT_COUNT):
        print(f'Incidence: {data_sorted_temp[i][0]}, '
              f'Temp: {data_sorted_temp[i][IDX_NVT_TEMP]:.2f} K, '
              f'Time: {data_sorted_temp[i][IDX_NVT_TIME]:.2f} ps')
    print('-' * 80)
    for i in range(PRINT_COUNT):
        print(f'Incidence: {data_sorted_time[i][0]}, '
              f'Temp: {data_sorted_time[i][IDX_NVT_TEMP]:.2f} K, '
              f'Time: {data_sorted_time[i][IDX_NVT_TIME]:.2f} ps')
    print('-' * 80)
    breakpoint()


if __name__ == '__main__':
    main()
