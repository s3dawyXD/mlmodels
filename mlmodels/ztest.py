# -*- coding: utf-8 -*-
import copy
import math
import os
from collections import Counter, OrderedDict
import json

import numpy as np
import pandas as pd
# import scipy as sci
# import sklearn as sk

####################################################################################################

# import autogluon
# import gluonts
####################################################################################################
import mlmodels

from mlmodels.util import get_recursive_files, log, os_package_root_path, model_get_list, os_get_file




def os_file_current_path():
  import inspect
  val = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  # return current_dir + "/"
  # Path of current file
  # from pathlib import Path

  # val = Path().absolute()
  val = str(os.path.join(val, ""))
  # print(val)
  return val



####################################################################################################  
def test_model_structure():
  print("os.getcwd", os.getcwd())
  print(mlmodels) 

  path = mlmodels.__path__[0]
  
  print("############Check structure ############################")
  cmd = f"ztest_structure.py"
  os.system( cmd )


def test_import() :
    import tensorflow as tf
    import torch as torch

    print(np, np.__version__)
    print(tf, tf.__version__)
    print(torch, torch.__version__)
    print(mlmodels)


    from importlib import import_module

    block_list = ["raw"]

    file_list = os_get_file(folder=None, block_list=[], pattern= r"/*.py")
    for f in file_list  :
        try :
            f = f.replace("\\", ".").replace(".py", "")
            import_module(f)
            print(f)
        except Exception as e:
            print(f, e)



def test_all():
  print("os.getcwd", os.getcwd())

  path = mlmodels.__path__[0]
  print("############Check model ################################")
  model_list = model_get_list(folder=None, block_list=[])
  print(model_list)


  ## Block list
  root = os_package_root_path()
  cfg = json.load(open(root + "config/test_config.json", mode='r'))['test_all']
  block_list = cfg['block_model_list']
  model_list = [ t for t in model_list if t not in block_list]
  print("Used", model_list)


  path = path.replace("\\", "//")  
  test_list =[ f"python {path}/"  + t.replace(".","//").replace("//py", ".py") for t in model_list ] 


  for cmd in test_list :
    print("\n\n\n", flush=True)
    print(cmd, flush=True)
    os.system( cmd )




def test_list(mlist):
  print( "os.getcwd", os.getcwd() )
  print("############Check model ################################")
  path  = mlmodels.__path__[0]
  #mlist = str_list.split(",")
  test_list =[   f"python {path}/{model}"  for model in  mlist ]  
   
  for cmd in test_list :
    print("\n\n\n", flush=True)
    print(cmd, flush=True)
    os.system( cmd )



def test_custom():
  test_list0 =[    
   ### Tflow
   f"model_tf/namentity_crm_bilstm.py",
 
  
   ### Keras
   f"model_keras/01_deepctr.py",
   f"model_keras/charcnn.py",


   f"model_keras/01_deepctr.py",
   f"model_keras/textcnn.py",


   ### SKLearn
   f"model_sklearn/sklearn.py",


   ### Torch
   f"model_tch/03_nbeats.py",
   f"model_tch/textcnn.py",
   f"model_tch/transformer_classifier.py",


   ### Glueon
   f"model_gluon/gluon_deepar.py",
   f"model_glufon/gluon_ffn.py",
   ]
  test_list(test_list0)




#################################################################################################################
#################################################################################################################
def cli_load_arguments(config_file= None):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    import argparse
    # from util import load_config
    #if config_file is None  :
    #  cur_path = os.path.dirname(os.path.realpath(__file__))
    #  config_file = os.path.join(cur_path, "template/test_config.json")
    # print(config_file)

    
    p = argparse.ArgumentParser()
    def add(*w, **kw) :
       p.add_argument(*w, **kw)
    
    add("--config_file", default=config_file, help="Params File")
    add("--config_mode", default="test", help="test/ prod /uat")
    add("--log_file", help="log.log")
    add("--do", default="test_all", help="test")
    add("--folder", default=None, help="test")
    
    ##### model pars



    ##### data pars




    ##### compute pars



    ##### out pars
    add("--save_folder", default="ztest/",  help=".")
    

    arg = p.parse_args()
    # arg = load_config(arg, arg.config_file, arg.config_mode, verbose=0)
    return arg


def main():
  arg = cli_load_arguments()
  print(arg.do)


  if ".py" in arg.do   :  #list all models in the repo
    s = arg.do
    test_list(  s.split(",") )


  else :
    globals()[arg.do]()





if __name__ == "__main__":
   main()
  


  



