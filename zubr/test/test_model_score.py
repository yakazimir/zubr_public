from zubr import get_config
from zubr.Learner import ModelProbs 

if __name__ == "__main__":
    config = get_config('zubr.Optimizer')
    config.dir = 'experiments/technical_documentation/runs/model_scores'

    m = ModelProbs(3)

    m.global_ranks[0] = 10.0
    m.global_ranks[1] = 1.0
    m.global_ranks[2] = 0.3
    m.global_ranks[3] = 0.003
    m.global_counts = 100.0
    
    m.normalize(it=10,wdir=config.dir,lang_map={"java":0,"python":1})

