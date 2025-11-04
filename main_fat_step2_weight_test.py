import torch
import os
import utility
import data
import model
import loss
from option_fat_step2_weight import args
from trainer_fat_step2_weight import Trainer
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        utility.get_parameter_number(model)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():

            t.test()
        checkpoint.done()

