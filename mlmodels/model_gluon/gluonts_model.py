#https://github.com/arita37/mlmodels/blob/dev/mlmodels/model_gluon/gluonts_model.py
# -*- coding: utf-8 -*-

#### New version
new = True



"""
Advanded GlutonTS models

"""
import os, copy
import pandas as pd, numpy as np

import importlib
import matplotlib.pyplot as plt
from pathlib import Path
import json


from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.seq2seq import Seq2SeqEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.simple_feedforward import  SimpleFeedForwardEstimator
from gluonts.model.wavenet import WaveNetEstimator, WaveNetSampler, WaveNet



from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor
from gluonts.distribution.neg_binomial import NegativeBinomialOutput
from tqdm.autonotebook import tqdm
#### Only for SeqtoSeq
from gluonts.block.encoder import (
    HierarchicalCausalConv1DEncoder,
    RNNCovariateEncoder,
    MLPEncoder,
    Seq2SeqEncoder,  # Buggy, not implemented
)


####################################################################################################
from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri, json_norm

from mlmodels.util import load_function_uri


VERBOSE = False
#MODEL_URI = get_model_uri(__file__)


MODELS_DICT = {
"deepar"         : DeepAREstimator
,"deepstate"     : DeepStateEstimator
,"deepfactor"    : DeepFactorEstimator
,"gp_forecaster" : GaussianProcessEstimator
,"seq2seq"       : Seq2SeqEstimator
,"feedforward"   : SimpleFeedForwardEstimator
,"transformer"   : TransformerEstimator
,"wavenet"       : WaveNetEstimator
}


####################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None,  compute_pars=None, **kwargs):
        self.compute_pars = compute_pars
        self.model_pars   = model_pars
        self.data_pars    = data_pars
        

        ##### Empty model for Seiialization
        if model_pars is None :
            self.model = None

        else:
            mpars = json_norm(model_pars['model_pars'] )      #"None" to None
            cpars = json_norm(compute_pars['compute_pars'])
            
            if model_pars["model_name"] == "seq2seq" :
                mpars['encoder'] = MLPEncoder()   #bug in seq2seq

            if model_pars["model_name"] == "deepar" :
                #print(str(mpars['distr_output']))
                #mpars['distr_output'] 
                

                #### distr_output – Distribution to use to evaluate observations and sample predictions (default: StudentTOutput())
                mpars['distr_output'] = NegativeBinomialOutput()
                #### NOT WOrking    ########## Needs   ()  to have an instance of class
                #mpars['distr_output'] =_load_function(mpars['distr_output'])()  # "gluonts.distribution.neg_binomial:NegativeBinomialOutput" 
                print(mpars['distr_output'])


                ### cardinality – Number of values of each categorical feature. This must be set if use_feat_static_cat == True (default: None)
                ### self.train_ds,self.test_ds, self.cardinalities = get_dataset2(data_pars)             
                ### mpars["cardinality"]  =list(self.cardinalities)  ## later with with get_dataset()                
                ### Need to put manually in JSON Before


                print( mpars["cardinality"] )
                # mpars["cardinality"]  =  [ 3,4,6,7]  ## later with with get_dataset()    


            ### Setup the compute
            trainer = Trainer( **cpars  )

            ### Setup the model
            self.model = MODELS_DICT[model_pars["model_name"]]( **mpars, trainer=trainer )



def get_params(choice="", data_path="dataset/timeseries/", config_mode="test", **kw):
    if choice == "json":
      data_path = path_norm( data_path )
      config    = json.load(open(data_path))
                                 #, encoding='utf-8'))
      config    = config[config_mode]
      
      return config["model_pars"], config["data_pars"], config["compute_pars"], config["out_pars"]
  
    else :
        raise Exception("Error no JSON FILE") 



def get_dataset2(data_pars):    
    from mlmodels.preprocess.timeseries import pandas_to_gluonts, pd_clean_v1
    from mlmodels.preprocess.timeseries import (  gluonts_create_dynamic,  gluonts_create_static,
      gluonts_create_timeseries, create_startdate,
      pandas_to_gluonts_multiseries )

    d = data_pars
     ###### New code                                 ########################################
    df_timeseries, df_static, df_dynamic, n_timeseries    = get_features(data_pars)
    
    ##Set parameters of dataset
    single_pred_length     = d['single_pred_length']  # 28
    submission_pred_length = single_pred_length * 2
    startdate              = d['startdate']  #  "2011-01-29"
    freq                   = d["freq"]
   
  
    submission             =data_pars['submission']
    pars                   = {'submission':submission,'single_pred_length':single_pred_length,
                            'submission_pred_length':submission_pred_length,
                            'n_timeseries':n_timeseries   ,
                            'start_date':startdate ,'freq':freq}
    
    if data_pars['train'] :
       train_ds, test_ds, cardinalities   = pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static,pars) 
      
       return train_ds, test_ds, cardinalities


    else :
       ### Submission mode
       _, test_ds, cardinalities   = pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static,pars) 
       return None, test_ds, cardinalities
    ############################################################################################################



def get_features(data_pars):
    data_folder            = data_pars[ "data_path"]
    
    """"
    calendar               = pd.read_csv(data_folder+'/calendar.csv')
    sales_train_val        = pd.read_csv(data_folder+'/sales_train_validation.csv.zip')
    sample_submission      = pd.read_csv(data_folder+'/sample_submission.csv.zip')
    sell_prices            = pd.read_csv(data_folder+'/sell_prices.csv.zip')
    cal_feat = calendar.drop( ['date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'event_name_1', 'event_name_2', 'd'],  axis=1 )
    cal_feat['event_type_1'] = cal_feat['event_type_1'].apply(lambda x: 0 if str(x)=="nan" else 1)
    cal_feat['event_type_2'] = cal_feat['event_type_2'].apply(lambda x: 0 if str(x)=="nan" else 1)

    df_dynamic    = cal_feat
    df_static     = sales_train_val[["item_id","dept_id","cat_id","store_id","state_id"]]
    df_timeseries = sales_train_val.drop(["id","item_id","dept_id","cat_id","store_id","state_id"], axis=1) 
    return df_timeseries,df_static,df_dynamic, len(sales_train_val)

    """

    ##### Need to  save MANUALLY Before   
    """
      We impose some dataframe format

    """ 
    df_dynamic    = pd.read_csv(data_folder+'/df_dynamic.csv')
    df_static     = pd.read_csv(data_folder+'/df_static.csv')
    df_timeseries = pd.read_csv(data_folder+'/df_timeseries.csv')
   
    return df_timeseries,df_static,df_dynamic, len(df_timeseries)




### OLD CODE#####
def get_dataset_old(data_pars):    
    from mlmodels.preprocess.timeseries import pandas_to_gluonts, pd_clean_v1
    print(data_pars)
    data_path=data_pars['data_path']
    ### Old Codes
    df = pd.read_csv(data_path)
    df = df.set_index( data_pars['col_date'] )
    df = pd_clean_v1(df)

    # start_date = pd.Timestamp( data_pars['start'], freq=data_pars['freq'])
    pars = { "start" : data_pars['start'], 
             "cols_target" : data_pars['col_ytarget'],
             "freq"        : data_pars['freq'],
             "cols_cat"    : data_pars["cols_cat"],
             "cols_num"    : data_pars["cols_num"]
        }    
    gluonts_ds = pandas_to_gluonts(df, pars=pars) 
 
    if VERBOSE:
        entry        = next(iter(gluonts_ds))
        train_series = to_pandas(entry)
        train_series.plot()
        save_fig     = data_pars.get('save_fig', "save_fig.png")
        # plt.savefig(save_fig)
    return gluonts_ds



def fit(model, sess=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None, **kwargs):
        """
          Classe Model --> model,   model.model contains thte sub-model
        ### OLD CODE
        print(data_pars,model_pars)
        data_pars['train'] = True
        
        gluont_ds          = get_dataset(data_pars)
        predictor          = model_gluon.train(gluont_ds)

        #### New version
        if data_pars['new'] = True :
          """

        data_pars['train'] = 1        
        train_ds, test_ds, cardinalities = get_dataset2(data_pars)
   
        model_gluon        = model.model
        
        predictor          = model_gluon.train(train_ds)
       
        #predictor          = model_gluon.train(model.train_ds)
        model.model        = predictor
        return model


def predict(model, sess=None, data_pars=None, compute_pars=None, out_pars=None, **kw):
    
    data_pars['train'] = 0 
    _, test_ds, cardinalities = get_dataset2(data_pars) 
    # test_ds            = model.test_ds
    model_gluon        = model.model
    
    forecast_it, ts_it = make_evaluation_predictions(
            dataset     = test_ds,      # test dataset
            predictor   = model_gluon,  # predictor
            num_samples = model.compute_pars['num_samples'],  # number of sample paths we want for evaluation
        )
    tss = list(tqdm(ts_it, total=len(model.test_ds)))

    forecasts = list(tqdm(forecast_it, total=len(model.test_ds)))

    #forecasts, tss = list(forecast_it), list(ts_it)
    forecast_entry, ts_entry = forecasts[0], tss[0]

    ### External benchmark.py evaluation
    if kw.get("return_ytrue") :
        ypred, ytrue = forecasts, tss
        return ypred, ytrue

    if VERBOSE:
        print(f"Number of sample paths: {forecast_entry.num_samples}")
        print(f"Dimension of samples: {forecast_entry.samples.shape}")
        print(f"Start date of the forecast window: {forecast_entry.start_date}")
        print(f"Frequency of the time series: {forecast_entry.freq}")
        print(f"Mean of the future window:\n {forecast_entry.mean}")
        print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")

    dd = {"forecasts": forecasts, "tss": tss}
    return dd



def metrics(model,ypred, data_pars, compute_pars=None, out_pars=None, **kw):
        ## load test dataset
       
        test_ds=model.test_ds

        forecasts = ypred["forecasts"]
        tss = ypred["tss"]

        ## Evaluate
        evaluator = Evaluator(quantiles=model.out_pars['quantiles'])
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
        metrics_dict = json.dumps(agg_metrics, indent=4)
        return metrics_dict, item_metrics



def fit_metrics(model,ypred, data_pars, compute_pars=None, out_pars=None, **kw):
        ### load test dataset
      
        test_ds=model.test_ds
        forecasts = ypred["forecasts"]
        tss = ypred["tss"]

        ### Evaluate
        evaluator = Evaluator(quantiles=out_pars['quantiles'])
        agg_metrics, item_metrics = evaluator(tss,forecasts, num_series=len(test_ds))
        metrics_dict = json.dumps(agg_metrics, indent=4)
        return metrics_dict, item_metrics

def save(model, path):
    import pickle
    path = path_norm(path + "/gluonts_model/")
    os.makedirs(path, exist_ok = True)

    model.model.serialize(Path(path) )   
    d = {"model_pars"  :  model.model_pars, 
         "compute_pars":  model.compute_pars,
         "data_pars"   :  model.data_pars
        }
    pickle.dump(d, open(path + "/glutonts_model_pars.pkl", mode="wb"))
    log(os.listdir(path))

"""    
def save_local(model, path):
    import pickle
    os.makedirs(path, exist_ok = True)

    model.model.serialize(Path(path) )   
    d = {"model_pars"  :  model.model_pars, 
         "compute_pars":  model.compute_pars,
         "data_pars"   :  model.data_pars
        }
    pickle.dump(d, open(path + "/glutonts_model_pars.pkl", mode="wb"))
    log(os.listdir(path))
"""    

def load(path):
    import pickle
    path = path_norm(path  + "/gluonts_model/" )

    predictor_deserialized = Predictor.deserialize(Path(path))
    d = pickle.load( open(path + "/glutonts_model_pars.pkl", mode="rb")  )
    
    ### Setup Model
    model = Model(model_pars= d['model_pars'], compute_pars= d['compute_pars'],
                  data_pars= d['data_pars'])  

    model.model = predictor_deserialized

    return model

"""    
def load_local(path):
    import pickle
   

    predictor_deserialized = Predictor.deserialize(Path(path))
    d = pickle.load( open(path + "/glutonts_model_pars.pkl", mode="rb")  )
    
    ### Setup Model
    model = Model(model_pars= d['model_pars'], compute_pars= d['compute_pars'],
                  data_pars= d['data_pars'])  

    model.model = predictor_deserialized

    return model
"""


def plot_prob_forecasts(ypred, out_pars=None):
    forecast_entry = ypred["forecasts"][0]
    ts_entry = ypred["tss"][0]
   
    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()


def plot_predict(item_metrics, out_pars=None):
    item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
    plt.grid(which="both")
    outpath = out_pars['path']
    os.makedirs(outpath, exist_ok=True)
    plt.savefig(outpath)
    plt.clf()
    print('Saved image to {}.'.format(outpath))



####################################################################################################
def test_single(data_path="dataset/", choice="", config_mode="test"):
    model_uri = MODEL_URI
    log("#### Loading params   ##############################################")
    log( model_uri)
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=choice, data_path=data_path, config_mode=config_mode)
    print(model_pars, data_pars, compute_pars, out_pars)

    log("#### Loading dataset   #############################################")
    #gluonts_ds = get_dataset(data_pars)
    
    log("#### Model init     ################################################")
    from mlmodels.models import module_load_full
    module, model = module_load_full(model_uri, model_pars, data_pars, compute_pars)
    print(module, model)


    log("#### Model fit     #################################################")
    #model=Model(model_pars, data_pars, compute_pars)
    model = fit(model, sess=None, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)    
    print(model)


    log("#### Save the trained model  ######################################")
    save(model, out_pars["path"])


    log("#### Load the trained model  ######################################")
    model = load(out_pars["path"])


    log("#### Predict   ####################################################")
    ypred = predict(model, sess=None, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    # print(ypred)


    log("#### metrics   ####################################################")
    metrics_val, item_metrics = metrics(model,ypred, data_pars, compute_pars, out_pars)
    print(metrics_val)


    log("#### Plot   #######################################################")
    if VERBOSE :
      plot_prob_forecasts(ypred, out_pars)
      plot_predict(item_metrics, out_pars)



def test() :
    ll = [ "deepar" , "deepfactor" , "transformer"  ,"wavenet", "feedforward",
           "gp_forecaster", "deepstate" ]

    ## Not yet  Implemented, error in Glutonts
    ll2 = [   "seq2seq"  ]
    
    for t in ll  :
      test_single(data_path="model_gluon/gluonts_model.json", choice="json", config_mode= t )
      #test_single(data_path="gluonts_model.json", choice="json", config_mode= t )


""" """
if __name__ == '__main__':
    VERBOSE = False

    #test()
