# Predicting Tanzania's water pumps maintenance
This project is an example of applying survival analyses concepts for maintenance. The purpose is to build a model to predict the survival probability and hazard function of a water pump, providing a tool to evaluate predictive maintenance and thus creating both an economic and social benefit. 
## Dataset
https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/
https://s3.amazonaws.com/drivendata/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv (for download)
https://s3.amazonaws.com/drivendata/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv (for download)
## Description
Data wrangling and visualization techniques are used to prepare the dataset for the model and to gain insight on the available data. Some regions have less functional water pumps than others, and in general it looks like not much maintenance has been performed over the years.
Three algorithms have been considered for modeling, i.e. Cox regression, SVM and Random Survival Forest. The latter is the best performing one, with a score of 0.797 on a test set.
A full report and a slide deck are available:


## Source code and requirements
The code is written in Python and uses some common packages plus the scikit-survival library.
Environment requirements:
https://github.com/aspds18/Springboard_capstone2/blob/master/tanzania.yml 

Complete code:
https://github.com/aspds18/Springboard_capstone2/blob/master/tanzaniawp.ipynb
## References:
1.	Pölsterl, S., Navab, N., and Katouzian, A., Fast Training of Support Vector Machines for Survival Analysis. Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2015, Porto, Portugal, Lecture Notes in Computer Science, vol. 9285, pp. 243-259 (2015)
2.	Pölsterl, S., Navab, N., and Katouzian, A., An Efficient Training Algorithm for Kernel Survival Support Vector Machines. 4th Workshop on Machine Learning in Life Sciences, 23 September 2016, Riva del Garda, Italy
3.	Pölsterl, S., Gupta, P., Wang, L., Conjeti, S., Katouzian, A., and Navab, N., Heterogeneous ensembles for predicting survival of metastatic, castrate-resistant prostate cancer patients. F1000Research, vol. 5, no. 2676 (2016).

