"""A simple algorithm to identify unseen captchas.
"""

import numpy as np
import os

from matplotlib import pyplot as plt


os.chdir(os.path.abspath(os.path.dirname(__file__)))

class Captcha(object):
    """
    An algorithm to learn patterns from training dataset
    and predict unseen captchas.
    """
            
    def __init__(self):
        filenames_input_raw = sorted('input/'+ filename for filename in 
                                    filter(lambda x: str.endswith(x, '.txt'),
                                           os.listdir('input')))
        filenames_output_raw = sorted('output/'+ filename for filename in 
                                    filter(lambda x: str.endswith(x, '.txt'),
                                           os.listdir('output')))
        in_number = [filename[-6:-4] for filename in filenames_input_raw]
        out_number = [filename[-6:-4] for filename in filenames_output_raw]
        # exclude the data point that has no output file, vice versa
        valid_number = sorted(list(set(in_number).intersection(out_number)))
        self.filenames_input = ['input/input' + v_n + '.txt' for v_n in valid_number]
        self.filenames_output = ['output/output' + v_n + '.txt' for v_n in valid_number]
        sample_image = self._binarize_image(self._read_image(self.filenames_input[0]))
        # get starting and ending row indices to determine the region for actual 
        # character representation in an image
        # obtain index of rows with all 1s, those rows are pure background
        row_indices = list(np.where(sample_image.sum(axis=1) == 60)[0])
        for i, value in enumerate(row_indices):
            if i != value:
                row_index = i
                break
        self._start_row = row_indices[row_index-1]+1
        self._end_row = row_indices[row_index]

        self._pattern = self._find_pattern()

    def _read_image(self, filename: str) -> np.array:
        """
        Read the file and obtain its image.
        args:
            filename: path to image txt.
        returns:
            image_raw: image in the numpy array format.
        """
        with open(filename) as f:
            content = f.readlines()
        # first row contains the image shape
        rows, columns = [int(x) for x in content[0].split(' ')]
        image_raw = np.empty([rows, columns, 3])
        # the rest rows contain the image
        for i, line in enumerate(content[1:]):
            image_raw[i, :, :] = np.array([list(map(int,
                values.split(','))) for values in line.split(' ')])
        return image_raw

    def _binarize_image(self, image_raw: np.array) -> np.array:
        """
        Binarize image to contain zeroes (foreground) and ones (background).
        args:
            image_raw: raw image to binarize, numpy array.
        returns:
            image_binarized: binarized images with only 0s and 1s.
        """
        # calculate brightness based on RGB values
        # formula reference: 
        # https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-6-201506-I!!PDF-E.pdf
        image_raw = 0.2126 * image_raw[:,:,0] + 0.7152 * image_raw[:,:,1] \
                    + 0.0722 * image_raw[:,:,2]
        # if pixel value > 127, pixel value is 1 that stands for white (background),
        # otherwise 0.
        image_binarized = (image_raw > 127).astype(int)
        return image_binarized

    def _find_pattern(self) -> dict:
        """
        Go through training images and find patterns.
        returns:
            pattern: a dictionary that maps characters from the letter/number 
                     to numpy arrays.
        """
        pattern = {}
        for filename_input, filename_output in zip(self.filenames_input,
                                                    self.filenames_output):
            image_raw = self._read_image(filename_input)
            image_binarized = self._binarize_image(image_raw)

            with open(filename_output) as file:
                labels = file.readlines()[0][:-1]
            # return index of columns with all 1s, those columns are pure background
            column_indices = [-1] + list(np.where(image_binarized.sum(axis=0) == 30)[0])
            
            # when adjacent columns change from all 1s to mix of 0s and 1s,
            # it indicates the region for actual letter/number.
            # we extract pixel value in the region with mixed 1s and 0s 
            # and use it as character pattern.
            j = 0
            for start_column, end_column in zip(column_indices[:-1], column_indices[1:]):
                if start_column + 1 != end_column:
                    character_represent = image_binarized[self._start_row:self._end_row,
                                                          start_column + 1:end_column]
                    pattern[labels[j]] = character_represent
                    j += 1
        return pattern
    
    def _match(self, image_binarized: np.array, pattern: dict) -> list:
        """
        Compare each region on the image with the character pattern 
        and record result in a matrix. Matrix index corresponds to the 
        starting row and column index of each region on the image.
        args:
            image_binarized: binarized images with only 0s and 1s.
            pattern: a dictionary that maps characters from the letter/number to numpy arrays.
        returns:
            location: list of index
        """
        match_record = np.zeros([image_binarized.shape[0] - pattern.shape[0] + 1,
                        image_binarized.shape[1] - pattern.shape[1] + 1])
        for i in range(image_binarized.shape[0] - pattern.shape[0] + 1):
            for j in range(image_binarized.shape[1] - pattern.shape[1] + 1):
                image_slice = image_binarized[i:i + pattern.shape[0],j:j + pattern.shape[1]]
                match_record[i, j] = (image_slice == pattern).all()
        # return the starting column index of matching region
        # to arrange identfied characters later according to position
        location = list(np.where(match_record == 1)[1])
        return location

    def __call__(self, im_path: str, save_path: str) -> None:
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        # get the RGM matrix of an image
        image = plt.imread(im_path)
        image = self._binarize_image(image)
        characters, positions_all = [], []
        # one character may appear more than once in the captcha
        for character, pattern in self._pattern.items():
            positions = self._match(image, pattern)
            if positions:
                for position in positions:
                    characters.append(character)
                    positions_all.append(position)
        out = ''.join(np.array(characters)[np.argsort(positions_all)])
        with open(save_path, 'w+') as f:
            f.write(out)


if __name__ == '__main__':
    idf_cap = Captcha()
    idf_cap('input/input100.jpg', 'output/final_result.txt')
