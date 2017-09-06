import sys
import shutil
import os

from zubr.Paraphrase import PivotParaphraser

def merge_models(new_loc,model_list):
    """Merge all paraphrase models into a single model

    :param new_loc: the location to put final model
    :param model_list: the list of models to merge
    """
    os.makedirs(new_loc)
    main_model_name = model_list[0].split('/')[-1]
    first_model_path  = os.path.join(model_list[0],"paraphrase.p")
    main_model = PivotParaphraser.from_model(first_model_path)

    ## copy over initial model
    main_subdir = os.path.join(new_loc,main_model_name)
    shutil.copytree(model_list[0],main_subdir)
    original_model = os.path.join(main_subdir,"paraphrase.p")

    ## go through the other models
    for model_dir in model_list[1:]:
        main_name = model_dir.split('/')[-1]
        paraphraser = os.path.join(model_dir,"paraphrase.p")
        try: 
            p = PivotParaphraser.from_model(paraphraser)
        except IOError:
            continue

        main_model.merge_models(p)

        ## make a copy of data 
        new_subdir = os.path.join(new_loc,main_name)
        shutil.copytree(model_dir,new_subdir)

        ## remove copy of single model
        os.remove(os.path.join(new_subdir,"paraphrase.p"))

        ## remove original directory (after backing up) 
        shutil.removetree(model_dir)

    new_model_path = os.path.join(new_loc,"paraphrase.p")
    main_model.save_model(new_model_path)
    os.remove(original_model)


if __name__ == "__main__":
    details = sys.argv[1:]
    new_loc = details[0]
    models = details[1:]
    
    ## merge the models into a single model
    merge_models(new_loc,models)
