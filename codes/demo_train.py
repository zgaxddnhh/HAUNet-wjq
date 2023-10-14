from option import args
import torch
import numpy as np
import random
import data
import model
import utils
import loss
import trainer
import os 
import datetime
import wandb

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch(args.seed)
    wandb.init(project=args.project_name, name = args.save)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    checkpoint = utils.checkpoint(args)
    if checkpoint.ok:
        dataloaders = data.create_dataloaders(args)   # dataloaders为一个dict
        sr_model = model.Model(args, checkpoint)
        total = sum([param.nelement() for param in sr_model.parameters()])
        print("Number of parameters: %.4fM" % (total / 1e6))
        sr_loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = trainer.Trainer(args, dataloaders, sr_model, sr_loss, checkpoint)
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        t.ckp.write_log("start_time:" + now + '\n')
        

        while not t.terminate():
            t.train()
            t.test()
            wandb.log({'epoch':t.scheduler.last_epoch, 'L1 loss': t.loss.log[-1].numpy(), 'psnr':t.ckp.log[-1].numpy()})

    end = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    t.ckp.write_log("end_time:" + end + '\n')
    checkpoint.done()
    wandb.finish()
