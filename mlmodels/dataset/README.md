#### Datasets



```
dataset/timeseries/
   Time series related



dataset/text/
   Time series related


dataset/tabular/
   Tabular related


dataset/recommender/
   Recommender


```



Hello,

Thanks.

I have some questions on the interview format :

   Phone call only or video ?

   Does it include live coding test ?
     (Do we need to connect to platform ?)


These are the dates ( I am taking some time off for this):




Wednesday April 1st (Japan time):

   At 0:00 AM (0:00) Tokyo Time = 8:00 AM (8:00) Previous Day San Francisco Time

   From  9:00 AM (9:00) Tokyo Time =   until anytime 
      (5:00 PM (17:00) Previous Day San Francisco Time)

   From  10 AM (Tokyo tme),  (6pm previous day SF time).



Thursday April 2nd (Japan time:

   From 1:00 AM (1:00) Tokyo Time (  9:00 AM (9:00) Previous Day San Francisco Time )


   From  9:00 AM (9:00) Tokyo Time =   until anytime 
      (5:00 PM (17:00) Previous Day San Francisco Time)

   From  10 AM (Tokyo tme),  (6pm, previous day SF time).



AFTER April 2th, it is difficult since I have some project assignments,
before it's better.


Thank you
Kevin











https://github.com/arita37/ml_sampler











    to login with a token:
        http://localhost:8889/?token=


ded64cfa8d4cf6d08ad190a74431341157eac09e73d36fda&token=ded64cfa8d4cf6d08ad190a74431341157eac09e73d36fda



```

        data_pars ={
            "path"            : 
            "path_type"   :  "local/absolute/web"


            "data_type"   :   "text" / "recommender"  / "timeseries" /"image"
            "data_split"  : {"istrain" :  1   , "split_size" : 0.5, "randomseed" : 1   },


            "data_loader"      :   "mlmodels.data:pd_reader",
            "data_loader_pars" :   { "sequential" : 1  },


            "data_preprocessor" : "mlmodels.model_keras.prepocess:process",
            "data_preprocessor_pars" :  {  },


            "size" : [0,1,2],
            "output_size": [0, 6]            
          }




```




