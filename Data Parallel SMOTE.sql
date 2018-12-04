--
-- create a user defined type for returning result set
--

drop type if exists balancedset cascade;
create type balancedset as (
  avgpkts double precision,
  stdpkts double precision,
  avgbytes double precision,
  stdbytes double precision,
  avgdur double precision,
  stddur double precision,
  avgbps double precision,
  stdbps double precision,
  Label integer
);

--
-- PL/Python code to SMOTE (Synthetic Minority Over Sampling Technique Estimator)
--

DROP FUNCTION IF EXISTS smote(float[], int, int[],int,int);
CREATE OR REPLACE FUNCTION smote(
features_matrix_linear float[],
num_features int,
labels int[],
r0 int,
r1 int
)
RETURNS  setof balancedset
AS
$$
	from imblearn.over_sampling import SMOTE
	import pandas as pd
	import numpy as np
	y = np.asarray(labels)
	plpy.info("length of y is %s"%(len(y)))
	
	#decomposing array of arrays to a numpy matrix (Rows,Columns) of table 
 
	x_mat = np.array(features_matrix_linear).reshape(len(features_matrix_linear)/num_features, num_features)
	plpy.info("shape of mat is %s x %s" %(x_mat.shape[0], x_mat.shape[1]))
	xt=pd.DataFrame(x_mat[:]) 
	yt=pd.DataFrame(y[:]) 
	
	play.info("Calling SMOTE function...")
	sm = SMOTE(random_state=12,ratio={0:r0,1:r1})
	x_train_res, y_train_res = sm.fit_sample(xt, yt)
	xt=pd.DataFrame(x_train_res[:],columns=[ 'avgpkts',
							'stdpkts',
							'avgbytes',
							'stdbytes',
							'avgdur',
							'stddur',
							'avgbps',
							'stdbps'
							])
	yt=pd.DataFrame(y_train_res[:],columns=['label'])
	train_set=pd.concat([xt,yt],axis=1)
	
	#convert python pandas dataframe to list of lists so that GPDB can convert list of lists to GPDB composite type
	
	ts=list(list(x) for x in zip(*(train_set[x].values.tolist() for x in train_set.columns)))
	plpy.info('returning now')
        return(ts)
$$ LANGUAGE PLPYTHONU;

--
-- create user defined aggregate function array_agg_array to make up Array of Arrays i.e. {{c1,...c104},{c1,...c104},...n}
--

drop aggregate if exists array_agg_array(anyarray) cascade;
create ordered aggregate array_agg_array(anyarray)
(
SFUNC = array_cat,
STYPE = anyarray
);

--
-- call PL/Python SMOTE function to create balanced dataset
--

drop table if exists balanced_trainset;
  create table balanced_trainset as 
  Select * from smote(
  (select   array_agg_array(array[avgpkts,stdpkts,avgbytes,stdbytes,avgdur,stddur,avgbps,stdbps]) from packetdata order by id), -- Features passed as Array of Arrays 
         8 , -- number of features
  (select array_agg(label) from packet data order by id) -- class label array
          )
  ;
