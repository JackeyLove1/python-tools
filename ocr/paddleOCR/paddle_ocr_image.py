from paddleocr import PaddleOCR
import time
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
def ocr(image_file_path):
    t = time.time()
    print("start:", t)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory
    result = ocr.ocr(image_file_path, cls=True)
    files = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line[1][0])
            files.append(line[1][0])
    return files

image_file_path = "./math1.jpg"
print(ocr(image_file_path))