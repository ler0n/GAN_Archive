import wandb
import neptune.new as neptune


class BaseLogger:
    def __init__(self):
        pass

    def write_log(self, log_dict):
        pass

    def write_figure(self, epoch, fig):
        pass

    def finish(self):
        pass


class NeptuneLogger(BaseLogger):
    def __init__(self, secrets, args):
        super().__init__()
        self.run = neptune.init_run(
            project=secrets['NEPTUNE']['PROJECT_NAME'],
            api_token=secrets['NEPTUNE']['API_TOKEN'],
        )
        self.run['parameters'] = args

    def write_log(self, log_dict):
        for key, val in log_dict.items():
            self.run[f'{key}'].append(val)
    
    def write_figure(self, epoch, fig):
        self.run[f'result-Epoch {epoch}'].upload(fig)
    
    def finish(self):
        self.run.stop()
    

class WandbLogger(BaseLogger):
    def __init__(self, secrets, args):
        super().__init__()
        wandb.login(key=secrets['WANDB']['API_TOKEN'])
        self.run = wandb.init(project=secrets['WANDB']['PROJECT_NAME'])
        self.run.config.update(args)
    
    def write_log(self, log_dict):
        self.run.log(log_dict)

    def write_figure(self, epoch, fig):
        self.run.log({'generate_result': fig})
    
    def finish(self):
        self.run.finish()