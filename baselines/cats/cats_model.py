# from .models import cats as c
from baselines.cats.models.cats import CATs
from baselines.cats.utils_training.utils import parse_list

def Cats(datasets):
        if datasets == 'pfpascal':
                hyperpixel_ids = [2,17,21,22,25,26,28]
        elif datasets == "spair":
                hyperpixel_ids[0,8,20,21,26,28,29,30]
        cats = CATs(hyperpixel_ids=hyperpixel_ids)
        return cats