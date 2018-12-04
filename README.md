# Data Parallel SMOTE using Greenplum PL/Python

A dataset is imbalanced if the classification categories are not approximately equally represented. Often real-world datasets are predominantly composed of normal examples with only a small percentage of abnormal or ‘interesting’ examples. To overcome the challenge of learning imbalanced class datasets SMOTE (Synthetic Minority Over Sampling Technique Estimator) has been proposed by [1].

At a high level, SMOTE creates synthetic observations of the minority class by:

* Finding the k-nearest-neighbors for minority class observations (finding similar observations)
* Randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observation.

SMOTE on large big datasets is not computationally feasible, To be able apply SMOTE on large datasets we can use capability of  Greenplum’s data parallel execution of PL/Python routines.

### What is Data Parallelism?
* Little or no effort is required to break up the problem into a number of parallel tasks, and there exists no dependency (or communication) between those parallel tasks.
* Also known as ‘data parallel’ or ‘embarrassingly parallel’ tasks
* Examples:
  * Have each person in this room weigh themselves: Measure each person’s weight in parallel
  * Count a deck of cards by dividing it up between people in this room: Count in parallel
  * MapReduce
  * apply() family of functions in R.
  
  
  
  
  ![](https://github.com/pnagula/Data-Parallel-SMOTE/blob/master/Picture1.png)
  
  #### Data Parallelism through Procedural Language Extensions (PL/X)
  
  * Allows users to write SQL functions in the R, Python, Java, Perl, pgsql or C languages
  * The interpreter/VM of the language ‘X’ is installed on each node of the Greenplum Database Cluster
  * Data Parallelism:PL/X piggybacks on Greenplum MPP architecture

Let’s comeback to creating PL/Python function *smote* and examine different parts of the function.
* First, we need to create composite user defined type *balancedset* before creating function *smote*

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
                label integer
              );

* Create function *smote* with input parameters 
  * features_matrix - This is input rows packed as array of arrays
  * Num_features - number of input features 
  * Labels - class labels
  * Balancedset - This is a result set from *smote* function which will have balanced set of rows and this is a composite user     defined type created just before creating *smote* function.
##### Here is the body of function *smote*, SMOTE algorithm is available in python package imblearn.over_sampling and we are trying to make use of that.

```python
    CREATE OR REPLACE FUNCTION smote(
     features_matrix_linear float[],
     num_features int,
     labels int[]
     )
     RETURNS  setof balancedset
     AS
     $$
        from imblearn.over_sampling import SMOTE
        import pandas as pd
        import numpy as np
        y = np.asarray(labels)
        plpy.info("length of y is %s"%(len(y)))
        #decomposing input rows which are represented as array of arrays to a numpy matrix (Rows,Columns) of table.  
        x_mat=np.array(features_matrix_linear).reshape(len(features_matrix_linear)/num_features, num_features)
        #print the shape of matrix.
        plpy.info("shape of mat is %s x %s" %(x_mat.shape[0], x_mat.shape[1]))

        #create a dataframe xt from numpy matrix x_mat
        xt=pd.DataFrame(x_mat[:]) 
        #create a dataframe yt from numpy matrix y
        yt=pd.DataFrame(y[:]) 

        play.info("Calling SMOTE function...")
        #create object sm from SMOTE.
        sm = SMOTE(random_state=12)
        #call the fit_sample function of object sm to execute SMOTE on the dataset.
        x_train_res, y_train_res = sm.fit_sample(xt, yt)

        #name the columns of dataframe so that they match composite type column names
        xt=pd.DataFrame(x_train_res[:],columns= ['avgpkts','stdpkts','avgbytes','stdbytes','avgdur','stddur','avgbps','stdbps'])
        yt=pd.DataFrame(y_train_res[:],columns=['label'])

        #concatenate the xt and yt dataframes to create a complete SMOTEd dataset
        train_set=pd.concat([xt,yt],axis=1)

        #This is tricky part, convert python pandas dataframe to list of lists so that GPDB can convert list of lists to GPDB         user defined composite type 
        ts=list(list(x) for x in zip(*(train_set[x].values.tolist() for x in train_set.columns)))

        #return the user defined composite type
        plpy.info('returning now')
        return(ts)
     $$ LANGUAGE PLPYTHONU;
```
    
 * Create a user defined aggregate function to encode rows of a table as Array of Arrays
    
    Let Row1 = {C11,C12,...,C1n}, Row2={C21,C22,...,C2n} be row arrays
    then Array of Arrays will be 
     
     
        {
          {C11,...,C1n}, Row1
          {C21,...,C2n}, Row2
          …
          {Cn1,...,Cnn}  Rown
        }
        
        drop aggregate if exists array_agg_array(anyarray) cascade;
        create ordered aggregate array_agg_array(anyarray)
        (
        SFUNC = array_cat,
        STYPE = anyarray
        );

* Finally, Calling *smote* PL/Python function, Observe the function *smote* is called from the 'table' position of a Select statement as the *smote* function is returning a result set. The query is distributed as statement has a function in the FROM clause returning a set of rows, the statement can run on the segments

        drop table if exists balanced_trainset;
        create table balanced_trainset as 
        Select * from smote(
        (select   array_agg_array(array[avgpkts,stdpkts,avgbytes,stdbytes,avgdur,stddur,avgbps,stdbps]) from packetdata order by id), -- Features passed as Array of Arrays 
               8 , -- number of features
        (select array_agg(label) from packet data order by id) -- class label array
                )
        ;
