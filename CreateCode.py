# coding:utf-8
import random
import os
import string
import json
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont, ImageColor

# 将随机函数赋给变量 rdint
rdint = random.randint


class VerifyCode(object):
    def __init__(self, width, height, bg_color, num, font_path, font_size, save_path):
        self.width = width  # 生成图片宽度
        self.height = height  # 生成图片高度
        self.bg_color = bg_color  # 生成图片背景颜色
        self.num = num  # 验证码字符个数
        self.font_path = font_path  # 字体路径
        self.font_size = font_size  # 字体大小
        self.code = ''  # 验证内容
        self.img = Image.new('RGB', (self.width, self.height), self.bg_color)  # 生成图片对象
        # 生成图片的保存路径
        self.savePath = save_path
        self.file_names = []

    # 获取随机颜色，RGB格式
    def get_random_color(self):
        c1 = rdint(50, 150)
        c2 = rdint(50, 150)
        c3 = rdint(50, 150)
        return (c1, c2, c3)

    # 随机生成1位字符,作为验证码内容
    def get_random_char(self):
        c = ''.join(random.sample(string.ascii_letters + string.digits, 1))
        self.code += c
        return c

    # 生成随机位置(x,y)
    def get_random_xy(self):
        x = rdint(0, int(self.width))
        y = rdint(0, int(self.height))
        return (x, y)

    # 根据字体文件生成字体，无字体文件则生成默认字体
    def get_font(self):
        if self.font_path:
            if os.path.exists(self.font_path):
                if self.font_size and 0 < self.font_size < self.height:
                    size = self.font_size
                else:
                    size = rdint(int(self.height / 1.5), int(self.height - 10))
                font = ImageFont.truetype(self.font_path, size)
                return font
            raise Exception('字体文件不存在或路径错误', self.font_path)
        return ImageFont.load_default().font

    # 图片旋转
    def rotate(self):
        deg = int(self.height / 3)  # 旋转角度
        self.img = self.img.rotate(rdint(0, deg), expand=0)

    # 画n条干扰线
    def draw_line(self, n):
        draw = ImageDraw.Draw(self.img)
        for i in range(n):
            draw.line([self.get_random_xy(), self.get_random_xy()],
                      self.get_random_color())
        del draw

    # 画n个干扰点
    def draw_point(self, n):
        draw = ImageDraw.Draw(self.img)
        for i in range(n):
            draw.point([self.get_random_xy()], self.get_random_color())
        del draw

    # 写验证码内容
    def draw_text(self, position, char, fill_color):
        draw = ImageDraw.Draw(self.img)
        draw.text(position, char, font=self.get_font(), fill=fill_color)
        del draw

    # 生成验证码图片，并返回图片对象
    def get_verify_img(self):
        x_start = 2
        y_start = 0
        self.img = Image.new('RGB', (self.width, self.height), self.bg_color)  # 生成图片对象
        for i in range(self.num):
            x = x_start + i * int(self.width / self.num)
            y = rdint(y_start, int(self.height / 3))
            self.draw_text((x, y), self.get_random_char(), self.get_random_color())
        self.file_names = list(self.code)
        self.code = ""
        self.draw_line(5)
        self.draw_point(200)
        return self.img

    # 将图片保存到内存,便于前台点击刷新
    # 将验证码保存到session中，返回内存中的图片数据
    def save_in_memory(self, request):
        img = self.get_verify_img()
        request.session['code'] = self.code.lower()
        f = BytesIO()  # 开辟内存空间
        img.save(f, 'png')
        return f.getvalue()

    # 将图片保存在本地，并以json格式返回验证码内容
    def save_in_local(self):
        img = self.get_verify_img()
        try:
            print(os.path.join(self.savePath, "".join(self.file_names) + ".png"), end="\n")
            img.save(os.path.join(self.savePath, "".join(self.file_names) + ".png"))
        except:
            raise NotADirectoryError('保存路径错误或不存在:' + self.savePath)
        return json.dumps({'code': self.code})


# v = vertifyCode(90, 50, ImageColor.colormap["white"], 4, 'C:\\Windows\\Fonts\\AdobeArabic-BoldItalic.otf', 40, savePath='img\\yzm')
# v.saveInLocal()

if not os.path.exists("codeDemo"):
    os.mkdir("codeDemo")

if not os.path.exists("codeDemo\\test"):
    os.mkdir("codeDemo\\test")

if not os.path.exists("codeDemo\\train"):
    os.mkdir("codeDemo\\train")


# 生成验证码
train = VerifyCode(100, 60, ImageColor.colormap["white"], 4, 'C:\\Windows\\Fonts\\AdobeArabic-BoldItalic.otf', 40, save_path='codeDemo\\train')
for i in range(20000):
    print(i, end="\t")
    train.save_in_local()
#
print("test===============")
test = VerifyCode(100, 60, ImageColor.colormap["white"], 4, 'C:\\Windows\\Fonts\\AdobeArabic-BoldItalic.otf', 40, save_path='codeDemo\\test')
for i in range(2000):
    print(i, end="\t")
    test.save_in_local()


# 生成标签
for root, dir_, files in os.walk("codeDemo"):
    for d in dir_:
        result = []
        file_name = os.path.join(root, d)
        for fname in os.listdir(file_name):
            result.append([os.path.join(file_name, fname), fname.split(".")[0]])
        with open(os.path.join(root, d+".txt"), "w") as f_:
            for r in result:
                f_.write(" ".join(r) + "\n")
