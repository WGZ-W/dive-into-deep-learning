
from tqdm import tqdm

def progress_bar(current, total, msg=None):
    """ 使用tqdm封装进度显示 """
    bar = tqdm(total=total, desc='Train',
              bar_format='{l_bar}{bar:20}{r_bar}',
              postfix=msg,
              ncols=100)  # 控制进度条总宽度
    bar.update(current - bar.n)
    if current + 1 >= total:
        bar.close()