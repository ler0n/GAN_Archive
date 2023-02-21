import os
import glob
import argparse

from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw


class DataGenerator:
    def __init__(self, args):
        self.width = args.width
        self.height = args.height
        self.output_dir = args.output_dir
        self._preprocess(args)
    
    def _preprocess(self, args):
        # 폰트 파일 경로 불러오기
        self.font_list = glob.glob(os.path.join(args.font_dir, '*.ttf'))
        
        # output directory가 존재하지 않으면 폴더 생성
        if not os.path.exists(args.output_dir): os.makedirs(os.path.join(args.output_dir))

        # letter_file에서 글자들 가져와서 하나의 문자열로 생성
        with open(args.letter_file, 'r', encoding='utf-8') as f:
            self.letters = ''.join(f.read().split())

        self.font_size = 48 * min(args.width, args.height) // 64
        

    def generate(self):
        for letter in tqdm(self.letters):
            for j, font_path in enumerate(self.font_list):
                img = Image.new('L', (self.width, self.height), color=0)
                font = ImageFont.truetype(font_path, self.font_size)
                drawing = ImageDraw.Draw(img)
                l, t, r, b = drawing.textbbox((0, 0), letter, font)
                drawing.text((((self.width - r + l) / 2), (self.height - b + t) / 2),
                              letter, fill=(255), font=font)
                img.save(os.path.join(self.output_dir, f'{letter}-{j}.png'), 'PNG')


if __name__== '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--width', type=int, default=64, help='width of generated image')
    args.add_argument('--height', type=int, default=64, help='height of generated image')
    args.add_argument('--letter_file', type=str, default='./data/label/korean_letters.txt', help='file containing letters')
    args.add_argument('--font_dir', type=str, default='./fonts', help='directory of .ttf files')
    args.add_argument('--output_dir', type=str, default='./data/letters', help='directory of saving generated data')

    args = args.parse_args()
    DataGenerator(args).generate()
