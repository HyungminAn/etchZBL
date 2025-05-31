import os

from ase.io import read

from params import PARAMS


class ImageLoader:
    def run(self, src_list, PATTERN='rm_byproduct_str_shoot_', INTERVAL=PARAMS.PLOT.HEIGHT.READ_INTERVAL):
        file_dict = self.get_file_list(src_list, PATTERN)
        keys = sorted(file_dict.keys())
        result = {}
        for key in keys[::INTERVAL]:
            file = file_dict[key]
            atoms = read(file, **PARAMS.LAMMPS.READ_OPTS)
            result[key] = atoms
        return result

    @staticmethod
    def get_file_list(src_list, pattern):
        '''
        Get the file list from the source directories,
        starting with the given pattern.
        '''
        file_dict = {}
        for src in src_list:
            for file in os.listdir(src):
                if not file.startswith(pattern):
                    continue
                key = int(file.split('_')[-1].split('.')[0])
                if file_dict.get(key) is not None:
                    print(f'key {key} already exists')
                    continue
                file_dict[key] = os.path.join(src, file)
        return file_dict

class ImageLoaderExtended(ImageLoader):
    def run(self, src_list, PATTERN=None, INTERVAL=None):
        result = {}
        images = super().run(src_list, PATTERN='rm_byproduct_str_shoot_')
        images_sub = super().run(src_list, PATTERN='save_str_shoot_', INTERVAL=1)
        if not images_sub:
            print('No slab subtract data found')
            return images

        sub_keys = sorted(images_sub.keys(), reverse=True)
        start_of_slab_subtract = min(images_sub.keys())
        for image_idx, image in images.items():
            if image_idx < start_of_slab_subtract:
                result[image_idx] = image
                continue

            for sub_key in sub_keys:
                if sub_key > image_idx:
                    continue
                save = images_sub[sub_key]

                cell = image.get_cell()
                z_shift = save.get_cell()[2, 2]
                cell[2, 2] += z_shift
                image.set_cell(cell)

                pos = image.get_positions()
                pos[:, 2] += z_shift
                image.set_positions(pos)
                image.extend(save)

            result[image_idx] = image

        # for image_idx, image in result.items():
        #     write(f'test_{image_idx}.extxyz', image, format='extxyz')
        #     print(f'{image_idx} written')

        return result

