
import evaluation

evaluation.evalrank('model_best.pth.tar',
                    data_path='',
                    split='test', fold5=True)
evaluation.evalrank('model_best.pth.tar',
                    data_path='',
                    split='test', fold5=False)
